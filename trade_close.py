# trade_close.py
import os, pytz
from datetime import datetime
from alpaca.trading.client import TradingClient

NY = pytz.timezone("America/New_York")

def get_client():
    key = os.getenv("APCA_API_KEY_ID")
    sec = os.getenv("APCA_API_SECRET_KEY")
    paper = os.getenv("APCA_PAPER", "true").lower() == "true"
    if not key or not sec:
        raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your environment.")
    return TradingClient(key, sec, paper=paper)

def main():
    client = get_client()
    clock = client.get_clock()
    if not clock.is_open:
        print("Market is closed — nothing to do.")
        return

    # Cancel any open orders first
    for o in client.get_orders(status="open"):
        try:
            client.cancel_order_by_id(o.id)
            print(f"Canceled order {o.id}")
        except Exception as e:
            print(f"Cancel failed {o.id}: {e}")

    # Liquidate positions
    positions = client.get_all_positions()
    if not positions:
        print("No positions to close.")
        return

    for p in positions:
        try:
            side = "sell" if float(p.qty) > 0 else "buy"
            client.close_position(p.symbol)
            print(f"Closed {p.symbol}")
        except Exception as e:
            print(f"Close failed {p.symbol}: {e}")

if __name__ == "__main__":
    main()