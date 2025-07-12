import os
import sys
import torch
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from src.models.custom_lstm import LSTMModel
import yfinance as yf
import pytz
from pathlib import Path
from src.config.settings import (
    CONFIG_PATH,
    ELU_MODEL_PATH,
    TANH_MODEL_PATH,
    DEFAULT_TICKER,
    DEFAULT_TIMEZONE,
    DEVICE_TYPE,
    BACKTEST_DAYS,
)
from src.utils.helpers import (
    setup_device,
    validate_dataframe,
    handle_multiindex_columns,
    compute_technical_features,
    inverse_transform_predictions,
)

# Setup device
DEVICE = setup_device(DEVICE_TYPE)


def load_config():
    """Load configuration from final_config.json"""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found at {CONFIG_PATH}")
        # Fallback to default values
        return {
            "dataset": {"sequence_length": 28},
            "model_architecture": {
                "input_size": 3,
                "hidden_sizes": [124],
                "dropout": 0.3,
            },
        }


# Load configuration
config = load_config()

# Extract configuration values
SEQ_LEN = config["dataset"]["sequence_length"]
INPUT_SIZE = config["model_architecture"]["input_size"]
HIDDEN_SIZE = config["model_architecture"]["hidden_sizes"][0]
DROPOUT = config["model_architecture"]["dropout"]
PRED_DAYS = 3  # Keep this as default

# Load Models with configuration parameters
elu_lstm = LSTMModel(
    input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, dropout=DROPOUT, activation_fn="elu"
).to(DEVICE)
elu_lstm.load_state_dict(torch.load(ELU_MODEL_PATH, map_location=DEVICE))
elu_lstm.eval()

tanh_lstm = LSTMModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    dropout=DROPOUT,
    activation_fn="tanh",
).to(DEVICE)
tanh_lstm.load_state_dict(torch.load(TANH_MODEL_PATH, map_location=DEVICE))
tanh_lstm.eval()


def compute_features(df):
    """Wrapper for backward compatibility"""
    return compute_technical_features(df)


def load_data():
    df = pd.read_csv("datasets/air_liquide.csv")
    features = compute_features(df)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    return df, scaled, scaler


def pull_latest_data_from_yahoo(days_to_pull=None, ticker=DEFAULT_TICKER):
    days_to_pull = days_to_pull if days_to_pull is not None else SEQ_LEN
    local_tz = pytz.timezone(DEFAULT_TIMEZONE)
    today = (pd.Timestamp.today()).strftime("%Y-%m-%d")
    days = days_to_pull * 2
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        df = yf.download(ticker, start=start_date, end=today, interval="1d")
        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'.")

        # Handle MultiIndex columns
        df = handle_multiindex_columns(df)

        # Validate required columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if not validate_dataframe(df, required_columns):
            return None, None, None

        # Handle timezone
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(local_tz)

        features = compute_features(df)
        print(f"features shape: {len(features)}")
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)
        return df, scaled, scaler
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None


def predict_last_week_internal(full_df, full_scaled, full_scaler):
    local_tz = pytz.timezone(DEFAULT_TIMEZONE)
    today = pd.Timestamp.today()
    # Use BACKTEST_DAYS instead of hardcoded 7 days
    backtest_start = (today - pd.Timedelta(days=BACKTEST_DAYS)).strftime("%Y-%m-%d")
    yesterday = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        if full_df is None or full_df.empty:
            return {"error": "Failed to fetch data from Yahoo Finance."}

        if "Close" not in full_df.columns:
            return {"error": "Required column 'Close' is missing from the data"}

        actual_df = full_df[backtest_start:yesterday]

        if len(actual_df) == 0:
            return []  # Return empty array instead of error

        days_to_predict = len(actual_df)
        print(days_to_predict, "days to predict")
        # Prepare for predictions
        future_predictions = []

        # For each day in the backtest period, create a separate prediction window
        for i in range(days_to_predict):
            predict_date = actual_df.index[i]

            # Get historical data up to the day before the prediction date
            history_end_date = predict_date - pd.Timedelta(days=1)
            historical_df = full_df[:history_end_date]

            if len(historical_df) < SEQ_LEN:
                continue  # Skip if not enough history

            # Compute features and scale
            features = compute_features(historical_df)

            # Check if features is empty
            if len(features) == 0:
                continue

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(features)

            # Prepare input sequence - last SEQ_LEN days before prediction date
            if len(scaled) < SEQ_LEN:
                continue  # Skip if not enough scaled data

            input_sequence = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 3)
            X_input = torch.tensor(input_sequence, dtype=torch.float32).to(DEVICE)

            # Make predictions
            with torch.no_grad():
                elu_pred_scaled = elu_lstm(X_input).cpu().numpy().flatten()[0]

            # Inverse transform only the Close price
            elu_close = scaler.inverse_transform([[elu_pred_scaled, 0, 0]])[0][0]

            # Get actual value - safely access 'Close' column
            actual_date = predict_date.strftime("%Y-%m-%d")
            try:
                actual_close = actual_df.iloc[i]["Close"]
            except (KeyError, IndexError) as e:
                print(f"Error accessing Close value for date {actual_date}: {e}")
                actual_close = None

            # Store predictions
            future_predictions.append(
                {
                    "date": actual_date,
                    "elu": elu_close,
                    "actual": actual_close,
                }
            )

        return future_predictions
    except Exception as e:
        print(f"Error in predict_last_week_internal: {str(e)}")
        return []  # Return empty array on error instead of error object


def predict_next_months(seq_length=None):
    # Use custom sequence length if provided, or default to SEQ_LEN
    # Ensure we pull enough data for both predictions and backtest validation
    base_days = seq_length if seq_length is not None else SEQ_LEN
    days_to_pull = base_days + BACKTEST_DAYS + 10  # Extra buffer for weekends/holidays

    df, scaled, scaler = pull_latest_data_from_yahoo(days_to_pull=days_to_pull)

    # Determine which sequence length to use for prediction input
    input_seq_len = SEQ_LEN  # Always use the model's expected sequence length for input

    if scaled is None or len(scaled) < input_seq_len:
        return {"error": "Not enough data."}

    # Take only the last SEQ_LEN elements regardless of how many were pulled
    latest_sequence = scaled[-input_seq_len:].reshape(1, input_seq_len, 3)
    X_input = torch.tensor(latest_sequence, dtype=torch.float32).to(DEVICE)

    future_predictions = []

    for _ in range(PRED_DAYS):
        with torch.no_grad():
            elu_pred_scaled = elu_lstm(X_input).cpu().numpy().flatten()[0]
            tanh_pred_scaled = tanh_lstm(X_input).cpu().numpy().flatten()[0]

        # Inverse transform only the Close price
        elu_close = scaler.inverse_transform([[elu_pred_scaled, 0, 0]])[0][0]
        tanh_close = scaler.inverse_transform([[tanh_pred_scaled, 0, 0]])[0][0]

        future_predictions.append({"elu": elu_close})

        # Simulate next feature vector (placeholder returns)
        next_row = np.array([[elu_pred_scaled, 0.0, 0.0]])
        latest_sequence = np.append(latest_sequence[:, 1:, :], [next_row], axis=1)
        X_input = torch.tensor(latest_sequence, dtype=torch.float32).to(DEVICE)

    # Get backtest predictions using the same data (now with enough historical data)
    last_week_data = predict_last_week_internal(df, scaled, scaler)

    return {
        "predicted_values": future_predictions,
        # Return the right number of data points from the pulled data
        "base_data": df.reset_index().values.tolist()[-min(base_days, len(df)) :],
        "last_week_data": last_week_data,
    }


def predict_last_week(seq_length=None):
    local_tz = pytz.timezone(DEFAULT_TIMEZONE)
    today = pd.Timestamp.today()
    backtest_start = (today - pd.Timedelta(days=BACKTEST_DAYS)).strftime("%Y-%m-%d")
    yesterday = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        # Get more data than needed to ensure we have enough history
        days_to_pull = (
            seq_length if seq_length is not None else SEQ_LEN + BACKTEST_DAYS + 10
        )
        full_df, full_scaled, full_scaler = pull_latest_data_from_yahoo(
            days_to_pull=days_to_pull
        )

        if full_df is None or full_df.empty:
            return {"error": "Failed to fetch data from Yahoo Finance."}

        # Check if 'Close' column exists
        if "Close" not in full_df.columns:
            print("Warning: 'Close' column not found in DataFrame")
            # Check for possible alternative column names
            close_columns = [col for col in full_df.columns if "close" in col.lower()]
            if close_columns:
                print(f"Using column '{close_columns[0]}' instead of 'Close'")
                full_df["Close"] = full_df[close_columns[0]]
            else:
                return {"error": "Required column 'Close' is missing from the data"}

        # Get actual data for the last BACKTEST_DAYS days
        actual_df = full_df[backtest_start:yesterday]

        if len(actual_df) == 0:
            return {"error": f"No data available for the last {BACKTEST_DAYS} days."}

        # Prepare for predictions
        future_predictions = []

        # For each day in the backtest period
        for i in range(len(actual_df)):
            predict_date = actual_df.index[i]

            # Get historical data up to the day before the prediction date
            history_end_date = predict_date - pd.Timedelta(days=1)
            historical_df = full_df[:history_end_date]

            if len(historical_df) < SEQ_LEN:
                continue  # Skip if not enough history

            # Compute features and scale
            features = compute_features(historical_df)

            # Check if features is empty
            if len(features) == 0:
                continue

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(features)

            # Prepare input sequence - last SEQ_LEN days before prediction date
            if len(scaled) < SEQ_LEN:
                continue  # Skip if not enough scaled data

            input_sequence = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 3)
            X_input = torch.tensor(input_sequence, dtype=torch.float32).to(DEVICE)

            # Make predictions
            with torch.no_grad():
                elu_pred_scaled = elu_lstm(X_input).cpu().numpy().flatten()[0]
                tanh_pred_scaled = tanh_lstm(X_input).cpu().numpy().flatten()[0]

            # Inverse transform only the Close price
            elu_close = scaler.inverse_transform([[elu_pred_scaled, 0, 0]])[0][0]
            tanh_close = scaler.inverse_transform([[tanh_pred_scaled, 0, 0]])[0][0]

            # Get actual value for comparison - safely access 'Close' column
            actual_date = predict_date.strftime("%Y-%m-%d")
            try:
                actual_close = actual_df.iloc[i]["Close"]
            except (KeyError, IndexError) as e:
                print(f"Error accessing Close value: {e}")
                actual_close = None

            # Store predictions
            future_predictions.append(
                {
                    "date": actual_date,
                    "elu": elu_close,
                    "tanh": tanh_close,
                    "actual": actual_close,
                }
            )

        # Get base data (data before the prediction period starts)
        if future_predictions and len(future_predictions) > 0:
            first_pred_date = pd.Timestamp(future_predictions[0]["date"])
            base_data_end = first_pred_date - pd.Timedelta(days=1)
            base_data_df = full_df[:base_data_end]
            base_data = base_data_df.reset_index().values.tolist()[-SEQ_LEN:]
        else:
            base_data = []

        return {
            "predicted_values": future_predictions,
            "base_data": base_data,
            "prediction_period": {"start": backtest_start, "end": yesterday},
        }
    except Exception as e:
        return {"error": f"Error predicting backtest data: {str(e)}"}


def predict_with_dataset():
    df, scaled, scaler = load_data()
    ochlv = df.values

    if len(scaled) < SEQ_LEN + PRED_DAYS:
        return {"error": "Not enough data."}

    train_data = scaled[:-PRED_DAYS]
    latest_sequence = train_data[-SEQ_LEN:].reshape(1, SEQ_LEN, 3)
    X_input = torch.tensor(latest_sequence, dtype=torch.float32).to(DEVICE)

    future_predictions = []
    for i in range(PRED_DAYS):
        with torch.no_grad():
            elu_pred_scaled = elu_lstm(X_input).cpu().numpy().flatten()[0]
            tanh_pred_scaled = tanh_lstm(X_input).cpu().numpy().flatten()[0]

        elu_close = scaler.inverse_transform([[elu_pred_scaled, 0, 0]])[0][0]
        tanh_close = scaler.inverse_transform([[tanh_pred_scaled, 0, 0]])[0][0]
        actual_close = ochlv[-PRED_DAYS + i][1]

        future_predictions.append(
            {
                "elu": elu_close,
                "actual": actual_close,
            }
        )

        next_row = np.array([[elu_pred_scaled, 0.0, 0.0]])
        latest_sequence = np.append(latest_sequence[:, 1:, :], [next_row], axis=1)
        X_input = torch.tensor(latest_sequence, dtype=torch.float32).to(DEVICE)

    return {
        "predicted_values": future_predictions,
        "base_data": ochlv[-SEQ_LEN - PRED_DAYS : -PRED_DAYS].tolist(),
    }
