# backtest.py
import os, math, pytz, yaml, warnings
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from strategy import train_lgbm  # use the same trainer as live
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

NY = pytz.timezone("America/New_York")


# -------------------- Data helpers --------------------
def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_daily_ohlc(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns MultiIndex (field, ticker) for fields:
    ['Open','High','Low','Close'].
    Index is DatetimeIndex (trading days).
    """
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        # Expected shape
        pass
    else:
        # single ticker -> promote to MultiIndex shape
        df = pd.concat({tickers[0]: df}, axis=1).swaplevel(axis=1)
    # Reorder to (field, ticker)
    df = df.reindex(columns=pd.MultiIndex.from_product([['Open','High','Low','Close'], tickers]))
    # forward-fill missing to keep continuity minimal (avoid filling Open/High/Low aggressively)
    return df.dropna(how="all")

def closes_from_ohlc(ohlc: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted({c[1] for c in ohlc.columns})
    close = ohlc['Close'].copy()
    close.columns = tickers
    return close

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

def compute_atr_proxy(prices: pd.DataFrame, lookback=14) -> pd.DataFrame:
    """Close-to-close ATR proxy in $."""
    r = prices.pct_change().abs()
    dollar_move = (r * prices).rolling(lookback).mean()
    return dollar_move


# -------------------- Sim engine --------------------
@dataclass
class Position:
    qty: int
    entry_px: float
    entry_date: pd.Timestamp
    stop: float
    take: float
    symbol: str

def daily_signal_rank(prices: pd.DataFrame, rank_date: pd.Timestamp, top_n: int) -> List[str]:
    data = compute_features_for_training(prices.loc[:rank_date])
    if data.empty:
        return []
    last_day = data["date"].max()
    if pd.to_datetime(last_day) != pd.to_datetime(rank_date):
        # not enough data for this date
        return []

    X_full = pd.get_dummies(
        data[["ret_1d","ret_5d","vol_5d","sma5_rel","sma20_rel","ticker"]],
        columns=["ticker"], drop_first=True
    ).astype(float)
    y_full = data["y_next"].astype(float).values
    dates  = pd.to_datetime(data["date"])

    train_mask = dates < last_day
    test_mask  = dates == last_day
    if train_mask.sum() < 100 or test_mask.sum() == 0:
        return []

    X_train, y_train = X_full[train_mask], y_full[train_mask]
    X_test  = X_full[test_mask]

    model = train_lgbm(X_train, y_train)
    preds = model.predict(X_test)

    test_rows = data.loc[test_mask, ["ticker","date"]].copy()
    test_rows["pred"] = preds
    test_rows = test_rows.sort_values("pred", ascending=False).reset_index(drop=True)
    picks = test_rows["ticker"].tolist()[:top_n]
    return picks

def simulate_backtest(cfg: Dict, ohlc: pd.DataFrame, start_buffer_days=60) -> pd.DataFrame:
    """
    Simulates:
      - Buy at OPEN with ATR-based bracket
      - Intraday exit using High/Low (stop first if both touched)
      - Hybrid close: hold winners based on config (momentum/trend/caps)
      - Overnight GTC-like stops honored with gap logic
    Returns equity curve DataFrame with columns: ['equity', 'cash', 'positions'].
    """
    tickers = sorted({c[1] for c in ohlc.columns})
    close = ohlc['Close'][tickers]
    high  = ohlc['High'][tickers]
    low   = ohlc['Low'][tickers]
    openp = ohlc['Open'][tickers]

    atr_lb = int(cfg.get("atr_lookback", 14))
    stop_k = float(cfg.get("stop_atr_mult", 1.5))
    take_k = float(cfg.get("take_atr_mult", 3.0))
    per_trade_budget = float(cfg.get("per_trade_budget", 500))
    top_n = int(cfg.get("top_n", 3))

    hold_cfg = cfg.get("hold", {})
    hold_enabled = bool(hold_cfg.get("enabled", False))
    hold_skip_friday = bool(hold_cfg.get("skip_friday", True))
    hold_max_pos = int(hold_cfg.get("max_overnight_positions", 0))
    hold_max_notional = float(hold_cfg.get("max_overnight_notional", 0.0))
    hold_min_today_ret = float(hold_cfg.get("min_today_ret", 0.0))
    hold_require_trend = bool(hold_cfg.get("require_sma_trend", True))
    hold_stop_k = float(hold_cfg.get("stop_atr_mult", 1.2))
    hold_atr_lb = int(hold_cfg.get("atr_lookback", atr_lb))

    prices = close.copy()
    atr_proxy = compute_atr_proxy(prices, lookback=atr_lb)

    dates = prices.index
    if len(dates) < start_buffer_days + 60:
        raise RuntimeError("Not enough history for backtest.")

    cash = 10000.0  # virtual starting cash (doesn't affect % returns except sizing if you change budget logic)
    equity_series = []
    positions: Dict[str, Position] = {}
    held_gtc_stop: Dict[str, float] = {}  # symbol -> stop price if held overnight

    # helper for hybrid hold checks
    def hybrid_hold_filter(day: pd.Timestamp, sym: str, qty: int) -> Tuple[bool, float]:
        if not hold_enabled:
            return False, 0.0
        if hold_skip_friday and day.weekday() == 4:
            return False, 0.0
        # momentum & trend from daily bars
        if day not in prices.index:
            return False, 0.0
        idx = prices.index.get_loc(day)
        if idx < 1:
            return False, 0.0
        last_c = prices.iloc[idx][sym]
        prev_c = prices.iloc[idx-1][sym]
        if not (np.isfinite(last_c) and np.isfinite(prev_c)) or last_c <= 0 or prev_c <= 0:
            return False, 0.0
        today_ret = last_c / prev_c - 1.0
        if today_ret < hold_min_today_ret:
            return False, 0.0
        if hold_require_trend:
            s5 = prices[sym].rolling(5).mean().iloc[idx]
            s20 = prices[sym].rolling(20).mean().iloc[idx]
            if not (last_c > s5 > s20):
                return False, 0.0
        # favor winners: require unrealized > 0 on the day (approx: last_c - entry_px > 0)
        pos = positions.get(sym)
        pnl_ok = True
        if pos is not None:
            pnl_ok = (last_c - pos.entry_px) * pos.qty >= 0.0
        # caps
        notional = qty * last_c
        return pnl_ok, notional

    # main loop: from the day after warmup to end-1 (since we buy at open of D+1)
    start_i = start_buffer_days
    for i in range(start_i, len(dates)-1):
        d_prev = dates[i-1]   # last fully known day
        d = dates[i]          # ranking day (we will enter at d+1 open)
        d_next = dates[i+1]   # execution/open and intraday range

        # --- Morning selection for next day ---
        history_up_to_d = prices.loc[:d]
        picks = daily_signal_rank(history_up_to_d, d, top_n)
        # determine ATRs & reference (use yesterday close ATR to size)
        atr_row = atr_proxy.loc[d] if d in atr_proxy.index else atr_proxy.iloc[i]
        # Intraday execution on d_next
        if d_next not in openp.index:
            continue

        day_open = openp.loc[d_next]
        day_high = high.loc[d_next]
        day_low  = low.loc[d_next]
        day_close= close.loc[d_next]

        # --- Enter positions at open with bracket levels
        todays_trades = []
        for sym in picks:
            px = float(day_open.get(sym, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue
            a  = float(atr_row.get(sym, 0.0))
            qty = max(1, int(math.floor(per_trade_budget / px)))
            if qty <= 0:
                continue
            stop = round(px - stop_k * a, 2)
            take = round(px + take_k * a, 2)
            if stop <= 0 or take <= stop:
                continue
            positions[sym] = Position(qty=qty, entry_px=px, entry_date=d_next, stop=stop, take=take, symbol=sym)
            todays_trades.append(sym)
            # cash is not tracked tightly (we're using per-trade budget); keep as informational

        # --- Simulate intraday exits on d_next (bracket logic) ---
        to_remove = []
        for sym, pos in positions.items():
            # only handle positions opened TODAY; held ones are managed via overnight logic below
            if pos.entry_date != d_next:
                continue
            hi = float(day_high.get(sym, np.nan))
            lo = float(day_low.get(sym, np.nan))
            cl = float(day_close.get(sym, np.nan))
            if not (np.isfinite(hi) and np.isfinite(lo) and np.isfinite(cl)):
                continue
            # If both stop & take touch, assume stop hits first (conservative)
            exited = False
            if lo <= pos.stop:
                # exit at stop
                cash += (pos.stop - pos.entry_px) * pos.qty
                to_remove.append(sym)
                exited = True
            elif hi >= pos.take:
                cash += (pos.take - pos.entry_px) * pos.qty
                to_remove.append(sym)
                exited = True
            if not exited:
                # neither hit; will decide at close whether to hold based on hybrid rules
                pass

        for sym in to_remove:
            positions.pop(sym, None)

        # --- At close of d_next: hybrid close vs flatten ---
        # Build lists for decision using caps
        hold_list: List[Tuple[str,int,float]] = []
        close_list: List[str] = []
        held_notional = 0.0

        for sym, pos in list(positions.items()):
            if pos.entry_date != d_next:
                # a previously held position continues; evaluate hold again
                qty = pos.qty
            else:
                qty = pos.qty

            # momentum/trend & caps
            ok, notional = hybrid_hold_filter(d_next, sym, qty)
            if not hold_enabled or (hold_skip_friday and d_next.weekday() == 4):
                ok = False
            if ok and len(hold_list) < hold_max_pos and (held_notional + notional) <= hold_max_notional:
                hold_list.append((sym, qty, float(close.loc[d_next][sym])))
                held_notional += notional
            else:
                close_list.append(sym)

        # Execute closes at close price for those not held
        for sym in close_list:
            if sym in positions:
                exit_px = float(day_close.get(sym, positions[sym].entry_px))
                cash += (exit_px - positions[sym].entry_px) * positions[sym].qty
                positions.pop(sym, None)
                held_gtc_stop.pop(sym, None)

        # For held positions: assign/refresh GTC-like stops based on hold_stop_k * ATR (use d_next ATR proxy)
        if hold_enabled and len(hold_list) > 0:
            atr_row_hold = compute_atr_proxy(prices, lookback=hold_atr_lb).loc[d_next]
            for sym, qty, cur_px in hold_list:
                a = float(atr_row_hold.get(sym, 0.0))
                held_gtc_stop[sym] = round(max(0.01, cur_px - hold_stop_k * a), 2)
                # keep the position, but update entry if it was not already present (it is)
                # we keep original entry for P&L tracking

        # --- Overnight gap handling for GTC stops on NEXT day open ---
        if i+2 < len(dates):
            next_open_day = dates[i+2]
            if next_open_day in openp.index and held_gtc_stop:
                op = openp.loc[next_open_day]
                lo = low.loc[next_open_day]
                # If gap below stop, fill at next open (worst case through)
                to_exit_overnight = []
                for sym, stop_px in list(held_gtc_stop.items()):
                    if sym not in positions:
                        held_gtc_stop.pop(sym, None)
                        continue
                    o = float(op.get(sym, np.nan))
                    l = float(lo.get(sym, np.nan))
                    if not (np.isfinite(o) and np.isfinite(l)):
                        continue
                    if o <= stop_px or l <= stop_px:
                        fill = o if o <= stop_px else stop_px  # gap-through -> open; otherwise stop
                        cash += (fill - positions[sym].entry_px) * positions[sym].qty
                        to_exit_overnight.append(sym)
                for sym in to_exit_overnight:
                    positions.pop(sym, None)
                    held_gtc_stop.pop(sym, None)

        # --- Mark-to-market equity end of day (d_next close) ---
        mtm = 0.0
        for sym, pos in positions.items():
            px = float(close.loc[d_next].get(sym, pos.entry_px))
            mtm += (px - pos.entry_px) * pos.qty
        equity = cash + mtm
        equity_series.append((d_next, equity, cash, len(positions)))

    eq = pd.DataFrame(equity_series, columns=["date", "equity", "cash", "positions"]).set_index("date")
    return eq


# -------------------- Metrics --------------------
def perf_metrics(eq: pd.DataFrame) -> Dict[str, float]:
    ret = eq["equity"].pct_change().dropna()
    if ret.empty:
        return {"CAGR":0, "Vol":0, "Sharpe":0, "MaxDD":0}
    # CAGR
    days = (eq.index[-1] - eq.index[0]).days
    yrs = max(1e-9, days/365.25)
    cagr = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (1/yrs) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / (ret.std() + 1e-12)
    # Max drawdown
    roll_max = eq["equity"].cummax()
    dd = 1 - eq["equity"]/roll_max
    maxdd = dd.max()
    return {
        "CAGR": float(cagr),
        "Vol": float(vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(maxdd)
    }


# -------------------- Main --------------------
def main():
    cfg = load_cfg()
    tickers = cfg["tickers"]
    end = datetime.now(NY).date()
    start = (end - timedelta(days=365*5)).isoformat()  # 5 years by default

    print(f"Loading OHLC for {len(tickers)} tickers from {start} to {end} …")
    ohlc = load_daily_ohlc(tickers, start=start, end=end.isoformat())
    eq = simulate_backtest(cfg, ohlc, start_buffer_days=80)

    # Save and report
    eq.to_csv("backtest_equity.csv")
    m = perf_metrics(eq)
    print(f"CAGR: {m['CAGR']*100:.2f}% | Vol: {m['Vol']*100:.2f}% | Sharpe: {m['Sharpe']:.2f} | MaxDD: {m['MaxDD']*100:.2f}%")
    print("Saved equity curve to backtest_equity.csv")

    # --- Equity curve plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(eq.index, eq["equity"], label="Equity", color="blue")
    plt.title("Backtest Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("backtest_equity.png", dpi=150)
    print("Saved equity curve plot to backtest_equity.png")
    plt.show()

if __name__ == "__main__":
    main()
