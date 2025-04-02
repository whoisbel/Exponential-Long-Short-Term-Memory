import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def pull_latest_data_from_yahoo():
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    yesterday = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    try:
        df = yf.download("AI.PA", start=start_date, end=yesterday, interval="1d")

        close_prices = df["Close"].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(close_prices)
        return df, scaled_prices, scaler
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None


df, scaled_prices, scaler = pull_latest_data_from_yahoo()
if df is not None:
    data_as_lists = df.reset_index().values.tolist()
    print(data_as_lists)
