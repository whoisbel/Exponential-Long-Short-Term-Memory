import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def pull_latest_data_from_yahoo():
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    yesterday = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    try:
        df = yf.download("AAPL", start=start_date, end=today, interval="1d")
        print(df)
        close_prices = df["Close"].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(close_prices)
        return df, scaled_prices, scaler
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None


print(pull_latest_data_from_yahoo())

dat = yf.Ticker("AI.PA")
tz = dat._fetch_ticker_tz(proxy=None, timeout=30)
valid = yf.utils.is_valid_timezone(tz)
print(f"{"AI sa"}: tz='{tz}', valid={valid}")
