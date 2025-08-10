
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from strategy import load_prices, build_dataset, train_lgbm

# ----- Config -----
TICKERS = ["AAPL","MSFT","NVDA","AMZN","META","GOOG","TSLA","AMD","NFLX","INTC"]
START   = "2022-01-01"
TOP_N   = 5
CAPITAL = 10_000
TRAIN_DAYS = 252  # ~1 trading year

def main():
    prices = load_prices(TICKERS, start=START)
    data   = build_dataset(prices)

    # Safety: confirm 'date'
    if "date" not in data.columns:
        raise RuntimeError(f"No 'date' column in dataset. Columns: {list(data.columns)}")

    # One-hot encode ticker so model can learn per-name effects
    X_full = pd.get_dummies(
        data[["ret_1d","ret_5d","vol_5d","sma5_rel","sma20_rel","ticker"]],
        columns=["ticker"],
        drop_first=True
    ).astype(float)
    y_full = data["y_next"].astype(float).values
    dates  = pd.to_datetime(data["date"]).values

    unique_days = np.unique(dates)
    equity      = [CAPITAL]
    daily_rets  = []

    for i in range(TRAIN_DAYS, len(unique_days) - 1):
        day_trade = unique_days[i]   # train up to this day, predict on this day
        train_mask = dates < day_trade
        test_mask  = dates == day_trade

        if test_mask.sum() == 0 or train_mask.sum() < 1000:
            continue

        X_train, y_train = X_full[train_mask], y_full[train_mask]
        X_test           = X_full[test_mask]
        y_test           = y_full[test_mask]  # realized next-day returns for those names

        model = train_lgbm(X_train, y_train)
        preds = model.predict(X_test)

        order = np.argsort(-preds)[:TOP_N]
        picks_y = y_test[order]
        day_ret = picks_y.mean() if len(picks_y) > 0 else 0.0

        daily_rets.append(day_ret)
        equity.append(equity[-1] * (1.0 + day_ret))

    # Results
    # Results
    eq_trade_days = pd.to_datetime(unique_days[TRAIN_DAYS:TRAIN_DAYS + len(daily_rets)])
    eq_index_full = [pd.to_datetime(unique_days[TRAIN_DAYS - 1])] + list(eq_trade_days)  # include initial point
    eq = pd.Series(equity, index=pd.DatetimeIndex(eq_index_full))

    # Stats
    dr = np.array(daily_rets)
    cagr = (eq.iloc[-1]/eq.iloc[0])**(252/len(dr)) - 1 if len(dr)>0 else 0
    vol  = dr.std()*np.sqrt(252) if len(dr)>1 else 0
    sharpe = (dr.mean()*252)/vol if vol>0 else 0
    mdd = 1 - (eq/eq.cummax()).min()

    print(f"CAGR: {cagr:.2%} | Vol: {vol:.2%} | Sharpe: {sharpe:.2f} | MaxDD: {mdd:.2%}")

    plt.figure()
    eq.plot(title="AI Morning Trader — Equity Curve")
    plt.xlabel("Date"); plt.ylabel("Equity ($)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
