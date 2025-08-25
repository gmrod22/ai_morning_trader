"""
HOOG dip/mean-reversion poller (5-minute cadence via GitHub Actions)
- Uses APCA_* env var names for credentials + paper toggle.
"""

import os, math, datetime as dt
import numpy as np
import pytz

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.common.exceptions import APIError

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ---------- CONFIG ----------
SYMBOL = os.getenv("SYMBOL", "HOOG")
FEED = os.getenv("DATA_FEED", "iex")  # 'iex' avoids SIP permission issues
PAPER = os.getenv("APCA_PAPER", "true").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

ROLLING_MINUTES = int(os.getenv("ROLLING_MINUTES", "40"))
BUY_Z_LADDERS = [-1.2, -2.0, -3.0]
BUY_SIZES      = [1,     2,    3   ]
SELL_Z_TRIMS = [0.0, 0.8, 1.5]
SELL_PCTS    = [0.50, 0.30, 0.20]

DOLLARS_PER_UNIT = float(os.getenv("DOLLARS_PER_UNIT", "150"))
MAX_POSITION_SHARES = int(os.getenv("MAX_POSITION_SHARES", "120"))
USE_MARKETABLE_LIMITS = True

NY_TZ = pytz.timezone("America/New_York")

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
if not API_KEY or not API_SECRET:
    raise SystemExit("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY env vars.")

trading = TradingClient(API_KEY, API_SECRET, paper=PAPER)
data = StockHistoricalDataClient(API_KEY, API_SECRET)

# ---------- UTIL ----------
def ny_now():
    return dt.datetime.now(NY_TZ)

def log(msg: str):
    print(f"[{ny_now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def account_equity() -> float:
    a = trading.get_account()
    return float(a.equity)

def buying_power() -> float:
    a = trading.get_account()
    return float(a.buying_power)

def position_qty_and_basis(symbol: str):
    try:
        pos = trading.get_open_position(symbol)
        return float(pos.qty), float(pos.avg_entry_price)
    except APIError:
        return 0.0, 0.0

def submit(side: str, qty: int, ref_price: float | None = None):
    if qty <= 0:
        return None
    if DRY_RUN:
        log(f"DRY_RUN: {side.upper()} {qty} {SYMBOL}")
        return None

    if USE_MARKETABLE_LIMITS and ref_price:
        limit_price = ref_price * (1.001 if side.lower()=="buy" else 0.999)
        req = LimitOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.BUY if side.lower()=="buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2)
        )
    else:
        req = MarketOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.BUY if side.lower()=="buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
    try:
        order = trading.submit_order(req)
        log(f"Placed {side.upper()} {qty} {SYMBOL} (id={order.id})")
        return order
    except Exception as e:
        log(f"Order error: {e}")
        return None

def shares_for_dollars(dollars: float, last_price: float) -> int:
    if last_price <= 0:
        return 0
    return max(1, int(dollars // last_price))

def zscore(series: np.ndarray):
    if series.size < 10:
        return None, None, None
    mu = series.mean()
    sd = series.std(ddof=1)
    if sd <= 1e-9:
        return None, mu, sd
    return (series[-1] - mu) / sd, mu, sd

def market_is_open() -> bool:
    clk = trading.get_clock()
    return bool(clk.is_open)

# ---------- MAIN ----------
def main():
    if not market_is_open():
        log("Market is closed; exiting.")
        return

    end = ny_now()
    start = end - dt.timedelta(minutes=ROLLING_MINUTES + 10)
    req = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=FEED,
        limit=ROLLING_MINUTES + 10
    )
    bars = data.get_stock_bars(req)
    if SYMBOL not in bars.data or len(bars.data[SYMBOL]) == 0:
        log("No bars returned; exiting.")
        return

    def _bar_close(b):
    # be tolerant across alpaca-py versions
    return getattr(b, "close", getattr(b, "c", None))

    series = [ _bar_close(b) for b in bars.data[SYMBOL] ]
    series = [ float(x) for x in series if x is not None ]
    closes = np.array(series[-ROLLING_MINUTES:])
    last_price = float(closes[-1])
    z, mu, sd = zscore(closes)

    mu_str = f"{mu:.2f}" if mu is not None else "NA"
    sd_str = f"{sd:.2f}" if sd is not None else "NA"
    z_str  = f"{z:.2f}"  if z  is not None else "NA"
    log(f"{SYMBOL} last={last_price:.2f} z={z_str} mu={mu_str} sd={sd_str}")

    qty, basis = position_qty_and_basis(SYMBOL)
    basis_str = f"{basis:.2f}" if basis else "0.00"
    log(f"Position: {qty:.0f} @ {basis_str} | BP ${buying_power():,.0f} | Eq ${account_equity():,.0f}")

    if z is None:
        return

    # --- BUY
    if qty < MAX_POSITION_SHARES:
        to_units = 0
        for lvl, units in zip(BUY_Z_LADDERS, BUY_SIZES):
            if z <= lvl:
                to_units = units
        if to_units > 0:
            est_qty = shares_for_dollars(DOLLARS_PER_UNIT * to_units, last_price)
            est_qty = max(0, min(est_qty, MAX_POSITION_SHARES - int(qty)))
            if buying_power() >= max(100, last_price * est_qty * 1.1) and est_qty > 0:
                submit("buy", est_qty, ref_price=last_price)

    # --- SELL
    if qty > 0:
        remaining = int(qty)
        sold_any = False
        for lvl, pct in zip(SELL_Z_TRIMS, SELL_PCTS):
            if z >= lvl and remaining > 0:
                to_sell = max(0, int(math.floor(qty * pct)))
                if to_sell > 0:
                    submit("sell", to_sell, ref_price=last_price)
                    remaining -= to_sell
                    sold_any = True
        if sold_any:
            log("Trimmed on reversion.")

if __name__ == "__main__":
    main()
