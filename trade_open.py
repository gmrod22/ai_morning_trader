# trade_open.py
import os, sys, math, time, pytz, yaml
import numpy as np, pandas as pd
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
import yfinance as yf
from strategy import train_lgbm  # reuse your trainer

NY = pytz.timezone("America/New_York")

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

    # Optional guard: only trade on regular sessions
    if not market_is_open_now(client):
        print("Market is closed now — not placing orders.")
        return

    tickers = cfg["tickers"]
    # Use data through **yesterday** so we never peek
    today_ny = datetime.now(NY).date()
    prices = fetch_prices(tickers, start="2022-01-01", end=str(today_ny))

    orders = build_orders(cfg, client, prices)

    print("Planned orders:")
    for sym, qty, px, stp, tk, _ in orders:
        print(f"  BUY {sym} x{qty} @ mkt  (stop {stp}, take {tk})  [ref close {px:.2f}]")

    if cfg.get("dry_run", True):
        print("\nDRY RUN = True → not sending orders. Set dry_run: false in config.yaml to trade.")
        return

    # Place orders
    for sym, qty, px, stp, tk, req in orders:
        try:
            o = client.submit_order(req)
            print(f"Submitted: {sym} x{qty} (stop {stp}, take {tk}) → id={o.id}")
            time.sleep(0.4)  # be gentle with rate limits
        except Exception as e:
            print(f"Failed {sym}: {e}")

if __name__ == "__main__":
    main()
