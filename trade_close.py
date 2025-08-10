# trade_close.py
import os, pytz
from datetime import datetime
from alpaca.trading.client import TradingClient
import csv
from pathlib import Path
from notifier import notify_slack

NY = pytz.timezone("America/New_York")

LOG_FILE = Path("trade_log.csv")

def append_pnl_and_return(date_str, total_pnl):
    # Update today's P&L ($) and daily_return (= pnl / notional) in trade_log.csv
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

def get_client():
    key = os.getenv("APCA_API_KEY_ID")
    sec = os.getenv("APCA_API_SECRET_KEY")
    paper = os.getenv("APCA_PAPER", "true").lower() == "true"
    if not key or not sec:
        raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your environment.")
    return TradingClient(key, sec, paper=paper)

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

    client = get_client()

    # If the market is closed, we still perform a cleanup/flatten
    try:
        today_str = datetime.now(NY).strftime("%Y-%m-%d")
        positions = client.get_all_positions()
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

    # Liquidate positions
    try:
        positions = client.get_all_positions()
        if not positions:
            print("No positions to close.")
        for p in positions:
            try:
                client.close_position(p.symbol)
                print(f"Closed {p.symbol}")
            except Exception as ce:
                print(f"Close failed {p.symbol}: {ce}")
    except Exception as e:
        print("Fetching positions failed:", e)

    # End-of-day notification
    try:
        notify_slack("Close script finished: canceled open orders and attempted to flatten positions.")
    except Exception:
        pass

if __name__ == "__main__":
    main()
