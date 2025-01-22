import yfinance as yf
import os

def download_and_save_data(tickers, start_date, end_date=None, save_folder="data"):
    """
    Downloads and saves data for the specified tickers and date range.
    
    :param tickers: List of ticker symbols to download data for.
    :param start_date: Start date for downloading data (format: 'YYYY-MM-DD').
    :param end_date: End date for downloading data (optional, default is None which means up to the current date).
    :param save_folder: Folder to save the downloaded CSV files (default is 'data').
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for ticker in tickers:
        file_name = os.path.join(save_folder, f"{ticker}.csv")
        if os.path.exists(file_name):
            print(f"Data for {ticker} already exists. Skipping download.")
            continue

        print(f"Downloading data for {ticker} from {start_date} to {end_date or 'current'}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            data.to_csv(file_name)
            print(f"Data for {ticker} saved to {file_name}.")
        else:
            print(f"No data available for ticker {ticker}.")

if __name__ == "__main__":
    tickers = ['AAPL', 'JPM', 'XOM', 'PG', 'TSLA', 'DIS']
    start_date = "2015-01-01"
    end_date = "2023-09-01"  # Or None for the current date
    download_and_save_data(tickers, start_date, end_date)
    print("Data download and save process completed.")
