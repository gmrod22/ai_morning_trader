# weekly_summary.py
import csv, math, os
from pathlib import Path
from datetime import datetime, timedelta
from statistics import mean, pstdev  # population std is fine for our short window
from notifier import notify_slack

LOG = Path("trade_log.csv")

def read_rows():
    if not LOG.exists():
        return [], []
    with open(LOG, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    header, data = rows[0], rows[1:]
    return header, data

def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def compute_metrics(header, data):
    # Parse dates, pnl, returns
    idx_date = header.index("date")
    idx_pnl  = header.index("pnl") if "pnl" in header else None
    idx_ret  = header.index("daily_return") if "daily_return" in header else None

    records = []
    for r in data:
        if not r or len(r) <= idx_date:
            continue
        d = r[idx_date]
        pnl = to_float(r[idx_pnl]) if idx_pnl is not None and idx_pnl < len(r) else 0.0
        ret = to_float(r[idx_ret]) if idx_ret is not None and idx_ret < len(r) else 0.0
        try:
            dt = datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            continue
        records.append((dt, pnl, ret))

    if not records:
        return None

    records.sort(key=lambda x: x[0])

    # Last 5 trading entries (not strictly calendar days)
    last5 = records[-5:] if len(records) >= 5 else records[:]
    week_pnl = sum(p for _, p, _ in last5)
    week_hit = sum(1 for _, p, _ in last5 if p > 0)
    week_trading_days = len(last5)
    hit_rate = (week_hit / week_trading_days) if week_trading_days else 0.0

    # Rolling Sharpe on daily returns (use up to last 20 entries)
    lastN = records[-20:] if len(records) >= 20 else records[:]
    daily_returns = [r for _, _, r in lastN if r != 0.0]
    if len(daily_returns) >= 5 and pstdev(daily_returns) > 0:
        sharpe = (mean(daily_returns) * 252) / (pstdev(daily_returns) * math.sqrt(252))
    else:
        sharpe = 0.0

    # Cumulative P&L and max drawdown in $
    cum = []
    running = 0.0
    max_peak = 0.0
    max_dd = 0.0
    for _, pnl, _ in records:
        running += pnl
        cum.append(running)
        max_peak = max(max_peak, running)
        dd = (max_peak - running)
        max_dd = max(max_dd, dd)

    return {
        "week_days": week_trading_days,
        "week_pnl": week_pnl,
        "hit_rate": hit_rate,
        "rolling_sharpe": sharpe,
        "cum_pnl": cum[-1] if cum else 0.0,
        "max_dd": max_dd,
    }

def main():
    header, data = read_rows()
    if not data:
        notify_slack("Weekly summary: no data yet.")
        return
    m = compute_metrics(header, data)
    if not m:
        notify_slack("Weekly summary: insufficient data.")
        return

    text = (
        "*Weekly Performance Summary*\n"
        f"• Trading days: {m['week_days']}\n"
        f"• Week P&L: ${m['week_pnl']:.2f}\n"
        f"• Hit rate: {m['hit_rate']*100:.1f}%\n"
        f"• Rolling Sharpe (≤20d): {m['rolling_sharpe']:.2f}\n"
        f"• Cumulative P&L: ${m['cum_pnl']:.2f}\n"
        f"• Max Drawdown (since start): ${m['max_dd']:.2f}"
    )
    notify_slack(text)

if __name__ == "__main__":
    main()
