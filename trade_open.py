# trade_open.py
import os, sys, math, time, pytz, yaml
import numpy as np, pandas as pd
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
import yfinance as yf
from strategy import train_lgbm  # reuse your trainer
from notifier import notify_slack
import csv
from pathlib import Path

NY = pytz.timezone("America/New_York")

LOG_FILE = Path("trade_log.csv")

def append_log(date_str, dry_run, orders):
    header = ["date", "dry_run", "n_orders", "symbols", "details"]
    row = [
        date_str,
        dry_run,
        len(orders),
        ",".join([o[0] for o in orders]),
        "; ".join([f"{o[0]} qty={o[1]} stop={o[3]:.2f} take={o[4]:.2f}" for o in orders])
    ]
    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_client():
    key = os.getenv("APCA_API_KEY_ID")
    sec = os.getenv("APCA_API_SECRET_KEY")
    paper = os.getenv("APCA_PAPER", "true").lower() == "true"
    if not key or not sec:
        raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your environment.")
    return TradingClient(key, sec, paper=paper)

def market_is_open_now(client):
    clock = client.get_clock()
    return clock.is_open

def fetch_prices(tickers, start="2022-01-01", end=None):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    closes = df["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame()
    return closes.dropna(how="all")

def compute_features_for_training(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in prices.columns:
        s = prices[t].dropna()
        r = s.pct_change()
        df = pd.DataFrame(index=s.index)
        df["ret_1d"]   = r.shift(1)
        df["ret_5d"]   = s.pct_change(5).shift(1)
        df["vol_5d"]   = r.rolling(5).std().shift(1)
        df["sma5_rel"] = s.rolling(5).mean().shift(1) / s
        df["sma20_rel"]= s.rolling(20).mean().shift(1) / s
        df["y_next"]   = s.pct_change().shift(-1)
        df["ticker"]   = t
        rows.append(df)
    data = pd.concat(rows).dropna().reset_index().rename(columns={"index":"date"})
    return data

def compute_atr(prices: pd.DataFrame, lookback=14):
    # simple close-to-close ATR proxy
    r = prices.pct_change().abs()
    dollar_move = (r * prices).rolling(lookback).mean()
    return dollar_move

def rank_today(cfg, prices):
    data = compute_features_for_training(prices)
    # train using all but the last day; predict on the last day’s row for each ticker
    last_day = data["date"].max()
    X_full = pd.get_dummies(
        data[["ret_1d","ret_5d","vol_5d","sma5_rel","sma20_rel","ticker"]],
        columns=["ticker"], drop_first=True
    ).astype(float)
    y_full = data["y_next"].astype(float).values
    dates  = pd.to_datetime(data["date"])

    train_mask = dates < last_day
    test_mask  = dates == last_day

    X_train, y_train = X_full[train_mask], y_full[train_mask]
    X_test  = X_full[test_mask]

    model = train_lgbm(X_train, y_train)
    preds = model.predict(X_test)

    test_rows = data.loc[test_mask, ["ticker","date"]].copy()
    test_rows["pred"] = preds
    test_rows = test_rows.sort_values("pred", ascending=False).reset_index(drop=True)
    return test_rows  # top rows = top picks

def build_orders(cfg, client, prices):
    tickers = cfg["tickers"]
    top_n   = int(cfg["top_n"])
    budget  = float(cfg["per_trade_budget"])
    stop_k  = float(cfg["stop_atr_mult"])
    take_k  = float(cfg["take_atr_mult"])
    look    = int(cfg["atr_lookback"])

    # rank by model
    ranked = rank_today(cfg, prices).head(top_n)

    # use last available close for sizing and brackets
    last_close = prices.iloc[-1]
    atr = compute_atr(prices, lookback=look).iloc[-1].fillna(0.0)

    orders = []
    for _, row in ranked.iterrows():
        t = row["ticker"]
        px = float(last_close.get(t, np.nan))
        a  = float(atr.get(t, 0.0))
        if not np.isfinite(px) or px <= 0:
            continue

        qty = max(1, int(math.floor(budget / px)))
        if qty == 0:
            continue

        stop_price = round(px - stop_k * a, 2)
        take_price = round(px + take_k * a, 2)
        if stop_price <= 0 or take_price <= stop_price:
            continue

        mo = MarketOrderRequest(
            symbol=t,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,   # sending right after the open
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_price),
            stop_loss=StopLossRequest(stop_price=stop_price)
        )
        orders.append((t, qty, px, stop_price, take_price, mo))
    return orders

def main():
    cfg = load_cfg()

    # --- DRY_RUN override from env (GitHub Secret) ---
    env_dry = os.getenv("DRY_RUN")
    if env_dry is not None:
        cfg["dry_run"] = env_dry.lower() in ("1", "true", "yes", "on")
    print(f"DRY_RUN = {cfg['dry_run']} (source={'env' if env_dry is not None else 'config.yaml'})")

    client = get_client()

    # Guard: only run when regular session is open
    if not market_is_open_now(client):
        msg = "Market is closed — skipping open script."
        print(msg)
        notify_slack(msg)
        return

    # Use data through yesterday (no peeking into today)
    tickers = cfg["tickers"]
    today_ny = datetime.now(NY).date()
    prices = fetch_prices(tickers, start="2022-01-01", end=str(today_ny))

    # Build orders (sizes + bracket levels)
    orders = build_orders(cfg, client, prices)

    today_str = datetime.now(NY).strftime("%Y-%m-%d")
    append_log(today_str, cfg.get("dry_run", True), orders)

    # --- Slack pre-trade summary ---
    if orders:
        lines = [f"DRY_RUN={cfg.get('dry_run', True)} | Top {len(orders)} orders for today:"]
        for sym, qty, px, stp, tk, _ in orders:
            lines.append(f"• {sym}: qty {qty} | ref {px:.2f} | stop {stp} | take {tk}")
        notify_slack("\n".join(lines))
    else:
        notify_slack(f"DRY_RUN={cfg.get('dry_run', True)} | No valid orders generated today.")
        print("No orders to place.")
        return

    # Dry run? just log/notify and exit
    if cfg.get("dry_run", True):
        print("DRY RUN = True → not submitting orders.")
        return

    # --- Submit orders ---
    submitted, failed = [], []
    for sym, qty, px, stp, tk, req in orders:
        try:
            o = client.submit_order(req)
            submitted.append((sym, qty, o.id))
            time.sleep(0.4)  # polite rate limiting
        except Exception as e:
            failed.append((sym, qty, str(e)))
            print(f"Failed {sym}: {e}")

    # --- Slack post-submit summary ---
    parts = []
    if submitted:
        parts.append("Submitted orders:")
        parts += [f"• {s} x{q} (id={oid})" for s, q, oid in submitted]
    if failed:
        parts.append("Failures:")
        parts += [f"• {s} x{q} → {err}" for s, q, err in failed]
    notify_slack("\n".join(parts) if parts else "No orders submitted.")
    

if __name__ == "__main__":
    main()
