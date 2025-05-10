import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.models.custom_lstm import LSTMModel
import yfinance as yf
import pytz
from pathlib import Path

parent_folder = str(Path(__file__).resolve().parents[3])
sys.path.append(parent_folder)

ELU_MODEL_PATH = "saved_models/baseline-top-result/model_elu.pth"
TANH_MODEL_PATH = "saved_models/baseline-top-result/model_tanh.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 60
PRED_DAYS = 3

# Load Models
elu_lstm = LSTMModel(input_size=3, activation_fn="elu").to(DEVICE)
elu_lstm.load_state_dict(torch.load(ELU_MODEL_PATH, map_location=DEVICE))
elu_lstm.eval()

tanh_lstm = LSTMModel(input_size=3, activation_fn="tanh").to(DEVICE)
tanh_lstm.load_state_dict(torch.load(TANH_MODEL_PATH, map_location=DEVICE))
tanh_lstm.eval()


def compute_features(df):
    df["return_1d"] = df["Close"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["return_5d"] = df["Close"].pct_change(5).replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    return df[["Close", "return_1d", "return_5d"]].values


def load_data():
    df = pd.read_csv("datasets/air_liquide.csv")
    features = compute_features(df)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    return df, scaled, scaler


def pull_latest_data_from_yahoo(days_to_pull=None, ticker="AI.PA"):
    local_tz = pytz.timezone("Asia/Manila")
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Use provided days_to_pull or default to SEQ_LEN
    days = days_to_pull if days_to_pull is not None else SEQ_LEN

    start_date = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        df = yf.download(ticker, start=start_date, end=today, interval="1d")
        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'.")

        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(local_tz)

        features = compute_features(df)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)

        return df, scaled, scaler
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None


def predict_next_months(seq_length=None):
    # Use custom sequence length if provided, or default to SEQ_LEN
    days_to_pull = seq_length if seq_length is not None else SEQ_LEN

    df, scaled, scaler = pull_latest_data_from_yahoo(days_to_pull=days_to_pull)

    # Determine which sequence length to use for prediction input
    input_seq_len = SEQ_LEN  # Always use the model's expected sequence length for input

    if scaled is None or len(scaled) < input_seq_len:
        return {"error": "Not enough data."}

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

    return {
        "predicted_values": future_predictions,
        "base_data": df.reset_index().values.tolist()[-days_to_pull:],
    }


def predict_last_week(seq_length=None):
    local_tz = pytz.timezone("Asia/Manila")
    today = pd.Timestamp.today()
    one_week_ago = (today - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    yesterday = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        # Get the full dataset using the existing function with custom sequence length if provided
        days_to_pull = seq_length if seq_length is not None else SEQ_LEN
        full_df, full_scaled, full_scaler = pull_latest_data_from_yahoo(
            days_to_pull=days_to_pull
        )

        if full_df is None or full_df.empty:
            return {"error": "Failed to fetch data from Yahoo Finance."}

        # Get actual data for the week we want to validate
        actual_df = full_df[one_week_ago:yesterday]

        if len(actual_df) == 0:
            return {"error": "No actual data found for the past week."}

        # Calculate number of days to predict (from one week ago to yesterday)
        days_to_predict = len(actual_df)

        # Prepare for predictions
        future_predictions = []

        # For each day in the week, create a separate prediction window
        for i in range(days_to_predict):
            predict_date = actual_df.index[i]

            # Get historical data up to the day before the prediction date
            history_end_date = predict_date - pd.Timedelta(days=1)
            historical_df = full_df[:history_end_date]

            if len(historical_df) < SEQ_LEN:
                continue  # Skip if not enough history

            # Compute features and scale
            features = compute_features(historical_df)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(features)

            # Prepare input sequence - last SEQ_LEN days before prediction date
            input_sequence = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 3)
            X_input = torch.tensor(input_sequence, dtype=torch.float32).to(DEVICE)

            # Make predictions
            with torch.no_grad():
                elu_pred_scaled = elu_lstm(X_input).cpu().numpy().flatten()[0]
                tanh_pred_scaled = tanh_lstm(X_input).cpu().numpy().flatten()[0]

            # Inverse transform only the Close price
            elu_close = scaler.inverse_transform([[elu_pred_scaled, 0, 0]])[0][0]
            tanh_close = scaler.inverse_transform([[tanh_pred_scaled, 0, 0]])[0][0]

            # Get actual value
            actual_date = predict_date.strftime("%Y-%m-%d")
            actual_close = actual_df.iloc[i]["Close"]

            # Store predictions
            future_predictions.append(
                {
                    "date": actual_date,
                    "elu": elu_close,
                    "tanh": tanh_close,
                    "actual": actual_close,
                }
            )

        # Use the earliest prediction's training data as base data
        if future_predictions:
            first_pred_date = pd.Timestamp(future_predictions[0]["date"])
            base_data_end = first_pred_date - pd.Timedelta(days=1)
            base_data_df = full_df[:base_data_end]
            base_data = base_data_df.reset_index().values.tolist()[-SEQ_LEN:]
        else:
            base_data = []

        return {
            "predicted_values": future_predictions,
            "base_data": base_data,
            "prediction_period": {"start": one_week_ago, "end": yesterday},
        }
    except Exception as e:
        return {"error": f"Error predicting last week's data: {str(e)}"}


def predict_with_dataset():
    df, scaled, scaler = load_data()
    ochlv = df.values
    print(df)
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
        "tanh": [],
    }
