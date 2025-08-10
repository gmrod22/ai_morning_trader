# trade_close.py
import os, pytz
from datetime import datetime
from alpaca.trading.client import TradingClient
from notifier import notify_slack

NY = pytz.timezone("America/New_York")

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
        notify_slack(msg)
        return

    client = get_client()

    # Cancel any open orders
    open_orders = client.get_orders(status="open")
    for o in open_orders:
        try:
            client.cancel_order(o.id)
        except Exception as e:
            print(f"Failed to cancel order {o.id}: {e}")

    # Close all positions
    positions = client.get_all_positions()
    for p in positions:
        try:
            client.close_position(p.symbol)
        except Exception as e:
            print(f"Failed to close {p.symbol}: {e}")

    # --- Slack summary after work is done ---
    notify_slack("Close script finished: canceled open orders and attempted to flatten positions.")

if __name__ == "__main__":
    main()
