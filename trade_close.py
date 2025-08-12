# trade_close.py
import os
import math
import time
import pytz
import yaml
import csv
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderStatus

from notifier import notify_slack

NY = pytz.timezone("America/New_York")
LOG_FILE = Path("trade_log.csv")


# -------------------- Config & Clients --------------------
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


# -------------------- Logging helpers --------------------
def append_pnl_and_return(date_str, total_pnl, notional_guess=0.0):
    """
    Update today's P&L ($) and daily_return (= pnl / notional) in trade_log.csv, if file/date exists.
    """
    if not LOG_FILE.exists():
        return
    with open(LOG_FILE, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header = rows[0]
    # ensure columns exist
    if "pnl" not in header:
        header += ["pnl", "daily_return"]
        rows[0] = header
    # find today and write values
    for i in range(1, len(rows)):
        r = rows[i]
        if r and r[0] == date_str:
            # prefer log notional if present
            try:
                idx_notional = header.index("notional")
                notional = float(r[idx_notional]) if r[idx_notional] else notional_guess
            except Exception:
                notional = notional_guess
            daily_ret = (total_pnl / notional) if notional > 0 else 0.0
            while len(r) < len(header):
                r.append("")
            r[header.index("pnl")] = f"{total_pnl:.2f}"
            r[header.index("daily_return")] = f"{daily_ret:.6f}"
            rows[i] = r
            break
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerows(rows)


# -------------------- Data helpers --------------------
def _fetch_daily_prices(tickers, end_date, lookback=60):
    """
    Pulls daily OHLC from yfinance. Returns (close_df, open_df)
    """
    start = (pd.to_datetime(end_date) - pd.Timedelta(days=int(lookback*2.2))).date().isoformat()
    df = yf.download(tickers, start=start, end=str(end_date), auto_adjust=False, progress=False)
    # yfinance returns MultiIndex cols for multiple tickers, normal df for single
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy()
        openp = df["Open"].copy()
    else:
        # single ticker -> promote to 2D
        close = df[["Close"]].copy()
        openp = df[["Open"]].copy()
        # name the single column with the ticker
        colname = tickers[0] if isinstance(tickers, (list, tuple)) and tickers else "TICKER"
        close.columns = [colname]
        openp.columns = [colname]
    # ensure sorted and clean
    close = close.sort_index().dropna(how="all")
    openp = openp.sort_index().dropna(how="all")
    return close, openp

def _atr_proxy(prices: pd.DataFrame, lookback=14) -> pd.Series:
    """
    Close-to-close ATR proxy in $ for a single symbol series.
    """
    r = prices.pct_change().abs()
    dollar = (r * prices).rolling(lookback).mean()
    return dollar.iloc[-1]  # last row (per symbol)


# -------------------- Hybrid hold filter --------------------
def _hybrid_hold_ok(cfg_hold, prices_close: pd.DataFrame, day, sym, entry_px_today):
    """
    Decide whether to hold 'sym' overnight using momentum/trend filters.
    Returns (ok: bool)
    """
    enabled = bool(cfg_hold.get("enabled", False))
    if not enabled:
        return False
    skip_friday = bool(cfg_hold.get("skip_friday", True))
    if skip_friday and pd.Timestamp(day).tz_localize(NY).weekday() == 4:
        return False

    # require price history
    if day not in prices_close.index:
        return False
    idx = prices_close.index.get_loc(day)
    if idx < 1:
        return False

    last_c = float(prices_close.iloc[idx].get(sym, np.nan))
    prev_c = float(prices_close.iloc[idx-1].get(sym, np.nan))
    if not (np.isfinite(last_c) and np.isfinite(prev_c)) or last_c <= 0 or prev_c <= 0:
        return False

    min_today_ret = float(cfg_hold.get("min_today_ret", 0.0))
    today_ret = last_c / prev_c - 1.0
    if today_ret < min_today_ret:
        return False

    if bool(cfg_hold.get("require_sma_trend", True)):
        s5 = prices_close[sym].rolling(5).mean().iloc[idx]
        s20 = prices_close[sym].rolling(20).mean().iloc[idx]
        if not np.isfinite(s5) or not np.isfinite(s20) or not (last_c > s5 > s20):
            return False

    # Favor winners on the day vs entry price (approx)
    if entry_px_today is not None and (last_c - entry_px_today) < 0:
        return False

    return True


# -------------------- Main Close Flow --------------------
def main():
    # DRY_RUN guard
    dry_env = os.getenv("DRY_RUN", "")
    dry_run = dry_env.lower() in ("1", "true", "yes", "on")

    cfg = load_cfg()
    hold_cfg = cfg.get("hold", {})

    client = get_client()
    ny_now = datetime.now(NY)
    today_str = ny_now.strftime("%Y-%m-%d")

    # --- Pull current positions & PnL snapshot
    try:
        positions = client.get_all_positions()
        total_unreal = sum(float(p.unrealized_pl) for p in positions)
        # best-effort notionals for logging
        notional_guess = sum(float(p.market_value) for p in positions if hasattr(p, "market_value"))
        append_pnl_and_return(today_str, total_unreal, notional_guess=notional_guess)
        try:
            notify_slack(f"[Close] Pre-close snapshot: positions={len(positions)} | Unrealized P&L ${total_unreal:.2f}")
        except Exception:
            pass
    except Exception as e:
        print("P&L snapshot error:", e)
        positions = []

    # If no positions, we can still cancel any stray open orders and exit
    symbols = sorted({p.symbol for p in positions})

    # --- Cancel any OPEN orders, but only for symbols we'll FLATTEN (decide below).
    # For now, just fetch open orders; we'll filter after we compute hold list.
    try:
        open_orders = []
        try:
            req = GetOrdersRequest(status=OrderStatus.OPEN)
            open_orders = list(client.get_orders(filter=req))
        except Exception as e:
            print("Fetching open orders failed:", e)
    except Exception as e:
        print("Open orders retrieval error:", e)
        open_orders = []

    # --- Build price history for hybrid decision (only if we have positions)
    close_df, open_df = (pd.DataFrame(), pd.DataFrame())
    if symbols:
        lookback = int(hold_cfg.get("atr_lookback", cfg.get("atr_lookback", 14))) + 40
        try:
            close_df, open_df = _fetch_daily_prices(symbols, end_date=ny_now.date(), lookback=lookback)
        except Exception as e:
            print("Price fetch error:", e)

    # --- Decide hold vs flatten for each position
    hold_list = []     # [(symbol, qty)]
    flatten_list = []  # [symbol]
    held_notional_cap = float(hold_cfg.get("max_overnight_notional", 0.0))
    held_pos_cap = int(hold_cfg.get("max_overnight_positions", 0))

    held_notional_sum = 0.0

    if symbols and not close_df.empty:
        last_day = close_df.index[-1]
        for p in positions:
            sym = p.symbol
            qty = int(float(p.qty))
            # Use today's entry ref if available (approx: today's open)
            entry_px_today = None
            try:
                entry_px_today = float(open_df.iloc[-1].get(sym, np.nan))
            except Exception:
                entry_px_today = None

            ok = _hybrid_hold_ok(hold_cfg, close_df, last_day, sym, entry_px_today)
            # enforce caps
            last_close = float(close_df.iloc[-1].get(sym, np.nan)) if sym in close_df.columns else None
            notional = qty * last_close if (last_close and np.isfinite(last_close)) else 0.0

            if ok and (held_pos_cap <= 0 or len(hold_list) < held_pos_cap) and \
               (held_notional_cap <= 0 or (held_notional_sum + notional) <= held_notional_cap):
                hold_list.append((sym, qty))
                held_notional_sum += notional
            else:
                flatten_list.append(sym)
    else:
        # no price data or no positions -> just flatten everything
        flatten_list = symbols[:]

    # --- Cancel open orders only for symbols we plan to FLATTEN
    if open_orders:
        for o in open_orders:
            try:
                if flatten_list and getattr(o, "symbol", None) in flatten_list:
                    client.cancel_order_by_id(o.id)
                    print(f"Canceled order {o.id} ({o.symbol})")
            except Exception as ce:
                print(f"Cancel failed {getattr(o, 'id', '?')}: {ce}")

    # --- DRY_RUN handling: log decision & exit
    if dry_run:
        msg = f"[Close] DRY_RUN=True → will flatten: {flatten_list or 'none'}; hold: {[s for s,_ in hold_list] or 'none'}"
        print(msg)
        try:
            notify_slack(msg)
        except Exception:
            pass
        return

    # --- Execute flatten operations
    closed_ok, close_err = [], []
    for sym in flatten_list:
        try:
            client.close_position(sym)
            closed_ok.append(sym)
            # be polite to API
            time.sleep(0.3)
        except Exception as ce:
            close_err.append((sym, str(ce)))

    # --- Summary & Slack
    parts = [f"[Close] Done. Closed {len(closed_ok)} positions."]
    if hold_list:
        parts.append(f"Holding overnight: {', '.join([s for s,_ in hold_list])}")
    if close_err:
        parts.append("Close errors:")
        parts += [f"• {s} → {err}" for s, err in close_err]
    msg = "\n".join(parts)
    print(msg)
    try:
        notify_slack(msg)
    except Exception:
        pass


if __name__ == "__main__":
    main()
