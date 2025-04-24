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


def pull_latest_data_from_yahoo():
    local_tz = pytz.timezone("Asia/Manila")
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")

    try:
        df = yf.download("AI.PA", start=start_date, end=today, interval="1d")
        if df.empty:
            raise ValueError("No data found for ticker 'AI.PA'.")

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


def predict_next_months():
    df, scaled, scaler = pull_latest_data_from_yahoo()
    if scaled is None or len(scaled) < SEQ_LEN:
        return {"error": "Not enough data."}

    latest_sequence = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 3)
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
        "base_data": df.reset_index().values.tolist()[-SEQ_LEN:],
    }


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
