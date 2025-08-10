
import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb

def load_prices(tickers, start="2022-01-01", end=None):
    """Download adjusted close prices for a list of tickers."""
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = df["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    return prices.dropna(how="all")

def build_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Build per-ticker features and next-day return label."""
    rows = []
    for t in prices.columns:
        s = prices[t].dropna()
        if s.empty:
            continue
        df = pd.DataFrame(index=s.index)
        r = s.pct_change()
        df["ret_1d"]   = r.shift(1)
        df["ret_5d"]   = s.pct_change(5).shift(1)
        df["vol_5d"]   = r.rolling(5).std().shift(1)
        df["sma5_rel"] = s.rolling(5).mean().shift(1) / s
        df["sma20_rel"]= s.rolling(20).mean().shift(1) / s
        # label: next-day close-to-close return
        df["y_next"]   = s.pct_change().shift(-1)
        df["ticker"]   = t
        rows.append(df.dropna())

    data = pd.concat(rows)
    data.index.name = "date"
    data = data.reset_index()          # now we have a 'date' column
    return data

def train_lgbm(X, y):
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9
    )
    model.fit(X, y)
    return model
