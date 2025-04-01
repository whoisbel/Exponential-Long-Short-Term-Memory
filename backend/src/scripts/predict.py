import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.models.custom_lstm import CustomLSTM

# Configurations
ELU_MODEL_PATH = "saved_models/TEST3/model_elu.pth"
TANH_MODEL_PATH = "saved_models/Original_model/model_tanh.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 60
PRED_DAYS = 10  # Number of days to predict


# Define Model Class (Same as training)
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=50, activation_fn="elu"):
        super(LSTMModel, self).__init__()
        self.lstm1 = CustomLSTM(
            input_size,
            hidden_size,
            num_layers=1,
            hidden_activation=activation_fn,
            cell_activation=activation_fn,
        )
        self.dropout1 = torch.nn.Dropout(0.2)
        self.lstm2 = CustomLSTM(
            hidden_size,
            hidden_size,
            num_layers=1,
            hidden_activation=activation_fn,
            cell_activation=activation_fn,
        )
        self.dropout2 = torch.nn.Dropout(0.2)
        self.lstm3 = CustomLSTM(
            hidden_size,
            hidden_size,
            num_layers=1,
            hidden_activation=activation_fn,
            cell_activation=activation_fn,
        )
        self.dropout3 = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.dropout3(out)
        last_output = out[-1]
        return self.fc(last_output)


# Load Model
elu_lstm = LSTMModel(activation_fn="elu").to(DEVICE)
elu_lstm.load_state_dict(torch.load(ELU_MODEL_PATH, map_location=DEVICE))
elu_lstm.eval()

tanh_lstm = LSTMModel(activation_fn="tanh").to(DEVICE)
tanh_lstm.load_state_dict(torch.load(TANH_MODEL_PATH, map_location=DEVICE))
tanh_lstm.eval()


# Initialize Scaler (Dummy example, replace with actual scaler state)
scaler = MinMaxScaler()
scaler.fit(np.random.rand(100, 1))  # Placeholder fit


def predict(past_values):
    if len(past_values) != SEQ_LEN:
        return {"error": f"Expected {SEQ_LEN} past values, got {len(past_values)}"}

    # Preprocess input
    input_sequence = np.array(past_values).reshape(-1, 1)
    scaled_input = scaler.transform(input_sequence)
    X_input = torch.tensor(scaled_input.reshape(1, SEQ_LEN, 1), dtype=torch.float32).to(
        DEVICE
    )

    # Predict
    with torch.no_grad():
        pred_scaled = elu_lstm(X_input).cpu().numpy().flatten()[0]
    pred_rescaled = scaler.inverse_transform([[pred_scaled]])[0][0]

    return {"predicted_value": float(pred_rescaled)}


def load_data():
    df = pd.read_csv("datasets/air_liquide.csv")
    close_prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(close_prices)

    return df, scaled_prices, scaler


def predict_from_csv():
    _, scaled_prices, scaler = load_data()

    # Get the last 60 closing prices
    if len(scaled_prices) < SEQ_LEN:
        return {"error": f"Not enough data. Need at least {SEQ_LEN} days."}

    latest_sequence = scaled_prices[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    X_input = torch.tensor(latest_sequence, dtype=torch.float32).to(DEVICE)

    # Predict
    with torch.no_grad():
        pred_scaled = elu_lstm(X_input).cpu().numpy().flatten()[0]
    pred_rescaled = scaler.inverse_transform([[pred_scaled]])[0][0]

    return {"predicted_value": float(pred_rescaled)}


def predict_next_month():
    df, scaled_prices, scaler = load_data()
    df = pd.read_csv("datasets/air_liquide.csv")
    ochlv_prices = df.values

    # Get the last 60 closing prices
    if len(scaled_prices) < SEQ_LEN:
        return {"error": f"Not enough data. Need at least {SEQ_LEN} days."}

    latest_sequence = scaled_prices[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    X_input = torch.tensor(latest_sequence, dtype=torch.float32).to(DEVICE)

    future_predictions = []
    # get the closing prices from the last 60 days

    for _ in range(PRED_DAYS):
        with torch.no_grad():
            elu_pred_scaled = elu_lstm(X_input).cpu().numpy().flatten()[0]
            tanh_pred_scaled = tanh_lstm(X_input).cpu().numpy().flatten()[0]

        elu_pred_rescaled = scaler.inverse_transform([[elu_pred_scaled]])[0][0]
        tanh_pred_rescaled = scaler.inverse_transform([[tanh_pred_scaled]])[0][0]

        future_predictions.append(
            {"elu": elu_pred_rescaled, "tanh": tanh_pred_rescaled}
        )

        # Update input sequence by removing the oldest value and adding the new prediction
        latest_sequence = np.append(
            latest_sequence[:, 1:, :],
            [[[scaler.transform([[elu_pred_rescaled]])[0][0]]]],
            axis=1,
        )
        X_input = torch.tensor(latest_sequence, dtype=torch.float32).to(DEVICE)

    return {
        "predicted_values": future_predictions,
        "base_data": ochlv_prices[-SEQ_LEN:].tolist(),
    }


def predict_with_dataset():
    """
    Leave the latest 10 days for testing and predict the rest
    """

    df, scaled_prices, scaler = load_data()
    df = pd.read_csv("datasets/air_liquide.csv")
    ochlv_prices = df.values

    # Get the last 60 closing prices
    if len(scaled_prices) < SEQ_LEN:
        return {"error": f"Not enough data. Need at least {SEQ_LEN} days."}

    # Leave the last 10 days for testing
    train_data = scaled_prices[:-10]

    # Initialize input sequence with the training data
    latest_sequence = train_data[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    X_input = torch.tensor(latest_sequence, dtype=torch.float32).to(DEVICE)

    future_predictions = []
    # get the closing prices from the last 60 days

    for _ in range(PRED_DAYS):
        with torch.no_grad():
            elu_pred_scaled = elu_lstm(X_input).cpu().numpy().flatten()[0]
            tanh_pred_scaled = tanh_lstm(X_input).cpu().numpy().flatten()[0]

        elu_pred_rescaled = scaler.inverse_transform([[elu_pred_scaled]])[0][0]
        tanh_pred_rescaled = scaler.inverse_transform([[tanh_pred_scaled]])[0][0]

        future_predictions.append(
            {
                "elu": elu_pred_rescaled,
                "actual": (ochlv_prices[-PRED_DAYS + _].tolist()[1]),
            }
        )

        # Update input sequence by removing the oldest value and adding the new prediction
        latest_sequence = np.append(
            latest_sequence[:, 1:, :],
            [[[scaler.transform([[elu_pred_rescaled]])[0][0]]]],
            axis=1,
        )
        X_input = torch.tensor(latest_sequence, dtype=torch.float32).to(DEVICE)

    return {
        "predicted_values": future_predictions,
        "base_data": ochlv_prices[-SEQ_LEN - PRED_DAYS : -PRED_DAYS].tolist(),
    }
