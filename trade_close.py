# trade_close.py
import os, pytz, yaml, csv
from datetime import datetime
from pathlib import Path

import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from notifier import notify_slack

NY = pytz.timezone("America/New_York")

# -------- Logging helpers --------
LOG_FILE = Path("trade_log.csv")

def append_pnl_and_return(date_str, total_pnl):
    """Update today's P&L ($) and daily_return (= pnl / notional) in trade_log.csv."""
    if not LOG_FILE.exists():
        return
    with open(LOG_FILE, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header = rows[0]
    if "pnl" not in header:
        header += ["pnl", "daily_return"]
        rows[0] = header
    for i in range(1, len(rows)):
        r = rows[i]
        if r and r[0] == date_str:
            try:
                idx_notional = header.index("notional")
                notional = float(r[idx_notional]) if r[idx_notional] else 0.0
            except Exception:
                notional = 0.0
            daily_ret = (total_pnl / notional) if notional > 0 else 0.0
            while len(r) < len(header):
                r.append("")
            r[header.index("pnl")] = f"{total_pnl:.2f}"
            r[header.index("daily_return")] = f"{daily_ret:.6f}"
            rows[i] = r
            break
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerows(rows)

# -------- Config / client helpers --------
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

# -------- Data helper --------
def _fetch_daily_prices(tickers, end_date, lookback=60):
    """Fetch daily adjusted closes for the universe up to end_date; return last `lookback` rows."""
    if not tickers:
        return yf.download("SPY", period="3mo", auto_adjust=True, progress=False)[["Close"]].tail(1)  # dummy
    df = yf.download(tickers, period="max", end=str(end_date), auto_adjust=True, progress=False)
    closes = df["Close"]
    if isinstance(closes, yf.pandas.Series):  # rare single-ticker shape
        closes = closes.to_frame()
    return closes.dropna(how="all").tail(lookback)

# -------- Main hybrid close --------
def main():
    # --- DRY_RUN guard (skip closing in dry mode) ---
    dry_env = os.getenv("DRY_RUN", "")
    dry_run = dry_env.lower() in ("1", "true", "yes", "on")
    if dry_run:
        msg = "DRY RUN = True → skipping cancel/close actions."
        print(msg)
        try:
            notify_slack(msg)
        except Exception:
            pass
        return

    cfg = load_cfg()
    hold_cfg = cfg.get("hold", {})
    hold_enabled = bool(hold_cfg.get("enabled", False))

    client = get_client()
    ny_now = datetime.now(NY)
    today_str = ny_now.strftime("%Y-%m-%d")

    # Get current positions
    try:
        positions = client.get_all_positions()
    except Exception as e:
        print("Fetching positions failed:", e)
        positions = []

    # Pre-close P&L snapshot (log + Slack)
    try:
        total_pnl = sum(float(p.unrealized_pl) for p in positions)
        append_pnl_and_return(today_str, total_pnl)
        try:
            notify_slack(f"Today's P&L snapshot (pre-close): ${total_pnl:.2f}")
        except Exception:
            pass
    except Exception as e:
        print("P&L logging error:", e)

    # Cancel any open orders first
    try:
        for o in client.get_orders(status="open"):
            try:
                client.cancel_order_by_id(o.id)
                print(f"Canceled order {o.id}")
            except Exception as ce:
                print(f"Cancel failed {o.id}: {ce}")
    except Exception as e:
        print("Fetching open orders failed:", e)

    if not positions:
        print("No positions to manage.")
        try:
            notify_slack("Close script finished: no positions.")
        except Exception:
            pass
        return

    # If hybrid hold is disabled OR it's Friday and we skip Friday holds → flatten everything
    if (not hold_enabled) or (ny_now.weekday() == 4 and hold_cfg.get("skip_friday", True)):
        try:
            for p in positions:
                client.close_position(p.symbol)
                print(f"Closed {p.symbol}")
            notify_slack("Close script finished: flattened all positions.")
        except Exception as e:
            print("Flatten error:", e)
        return

    # ---------- Hybrid logic: decide hold vs close ----------
    symbols = [p.symbol for p in positions]
    lookback = max(60, int(hold_cfg.get("atr_lookback", 14)) + 30)
    prices = _fetch_daily_prices(symbols, end_date=ny_now.date(), lookback=lookback)

    # Indicators
    atr_look = int(hold_cfg.get("atr_lookback", 14))
    # ATR proxy (close-to-close in $)
    atr = (prices.pct_change().abs() * prices).rolling(atr_look).mean().iloc[-1].fillna(0.0)
    sma5 = prices.rolling(5).mean().iloc[-1]
    sma20 = prices.rolling(20).mean().iloc[-1]
    last_close = prices.iloc[-1]
    prev_close = prices.iloc[-2] if len(prices) >= 2 else last_close
    today_ret = (last_close / prev_close - 1.0).fillna(0.0)

    # Thresholds / caps
    max_hold_positions = int(hold_cfg.get("max_overnight_positions", 0))
    max_hold_notional = float(hold_cfg.get("max_overnight_notional", 0.0))
    min_today_ret = float(hold_cfg.get("min_today_ret", 0.0))
    require_trend = bool(hold_cfg.get("require_sma_trend", True))
    stop_k = float(hold_cfg.get("stop_atr_mult", 1.2))

    to_hold, to_close = [], []
    held_notional = 0.0

    for p in positions:
        sym = p.symbol
        try:
            qty = abs(int(float(p.qty)))
        except Exception:
            qty = 0
        cur_px = float(last_close.get(sym, 0.0))
        if qty == 0 or cur_px <= 0:
            to_close.append(sym)
            continue

        mom_ok = float(today_ret.get(sym, 0.0)) >= min_today_ret
        if require_trend:
            s5 = float(sma5.get(sym, cur_px))
            s20 = float(sma20.get(sym, cur_px))
            trend_ok = (cur_px > s5 > s20)
        else:
            trend_ok = True
        pnl_ok = float(p.unrealized_pl) >= 0.0  # prefer winners

        candidate_notional = qty * cur_px

        if mom_ok and trend_ok and pnl_ok \
           and len(to_hold) < max_hold_positions \
           and (held_notional + candidate_notional) <= max_hold_notional:
            to_hold.append((sym, qty, cur_px))
            held_notional += candidate_notional
        else:
            to_close.append(sym)

    # Execute closes
    for sym in to_close:
        try:
            client.close_position(sym)
            print(f"Closed {sym}")
        except Exception as ce:
            print(f"Close failed {sym}: {ce}")

    # For holds: place a GTC protective stop
    for sym, qty, cur_px in to_hold:
        a = float(atr.get(sym, 0.0))
        stop_price = round(max(0.01, cur_px - stop_k * a), 2)
        try:
            stop_req = StopOrderRequest(
                symbol=sym,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=stop_price
            )
            client.submit_order(stop_req)
            print(f"Holding {sym} overnight with GTC stop @ {stop_price}")
        except Exception as e:
            print(f"GTC stop placement failed for {sym}: {e}")

    # Slack summary
    try:
        msg = (
            "Hybrid close:\n"
            + (f"• Held ({len(to_hold)}): " + ", ".join([h[0] for h in to_hold]) + "\n" if to_hold else "• Held: none\n")
            + (f"• Closed ({len(to_close)}): " + ", ".join(to_close) if to_close else "• Closed: none")
        )
        notify_slack(msg)
    except Exception:
        pass

if __name__ == "__main__":
    main()
