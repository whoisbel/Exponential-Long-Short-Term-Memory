import os

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator
import json

from src.models.custom_lstm import CustomLSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------ Config ------------------
# model architecture config
HIDDEN_SIZES = 50
INPUT_SIZE = 4
OUTPUT_SIZE = 1
NUM_LAYERS = 1

# training config
SEQ_LEN = 60
BATCH_SIZE = 32
PATIENCE = 15
DROPOUT = 0.2

# change for testing; refer to folders in saved_models readme.txt for epochs and learning rate
EPOCHS = 55
LEARNING_RATE = 0.01

# system config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# change for training; refer to folders in saved_models for folder names
SAVE_DIR = "saved_models/TEST6/"
os.makedirs(SAVE_DIR, exist_ok=True)

# save config to json
config_dict = {
    "model_architecture": {
        "hidden_sizes": HIDDEN_SIZES,
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "num_layers": NUM_LAYERS,
    },
    "training": {
        "sequence_length": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
    },
    "system": {"device": str(DEVICE), "save_dir": SAVE_DIR},
}

with open(f"{SAVE_DIR}/model_config.json", "w") as f:
    json.dump(config_dict, f, indent=4)


# ------------------ Load and preprocess data ------------------
df = pd.read_csv("../../datasets/air_liquide.csv")
close_prices = df["Close"].values.reshape(-1, 1)  # Use SMA20 for training
df["SMA20"] = df["Close"].rolling(window=20).mean()
df["SMA50"] = df["Close"].rolling(window=50).mean()

df.dropna(inplace=True)  # Drop rows with NaN values
features = df[["Close", "SMA20", "SMA50", "Volume"]].values
print(features.shape)
# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(features)

# Calculate SMA20 and SMA50

# Prepare data for LSTM
X, y = [], []
for i in range(SEQ_LEN, len(scaled_prices)):
    X.append(scaled_prices[i - SEQ_LEN : i])
    y.append(scaled_prices[i, 0])

X = np.array(X)
y = np.array(y)

train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

print(X_train.shape, y_train.shape)


# ------------------ Define Model ------------------
class LSTMModel(nn.Module):
    def __init__(
        self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZES, activation_fn="tanh"
    ):
        super(LSTMModel, self).__init__()
        self.lstm1 = CustomLSTM(
            input_size,
            hidden_size,
            num_layers=NUM_LAYERS,
            hidden_activation=activation_fn,
            cell_activation=activation_fn,
        )
        self.dropout1 = nn.Dropout(DROPOUT)

        self.lstm2 = CustomLSTM(
            hidden_size,
            hidden_size,
            num_layers=NUM_LAYERS,
            hidden_activation=activation_fn,
            cell_activation=activation_fn,
        )
        self.dropout2 = nn.Dropout(DROPOUT)

        self.lstm3 = CustomLSTM(
            hidden_size,
            hidden_size,
            num_layers=NUM_LAYERS,
            hidden_activation=activation_fn,
            cell_activation=activation_fn,
        )
        self.dropout3 = nn.Dropout(DROPOUT)

        self.fc = nn.Linear(hidden_size, OUTPUT_SIZE)

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


# ------------------ Training Function ------------------
def train_and_save(activation_fn="tanh"):
    print(f"\n=== Training with activation: {activation_fn.upper()} ===")

    model = LSTMModel(activation_fn=activation_fn).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                output = model(X_val).squeeze()
                loss = criterion(output, y_val.squeeze())
                val_loss += loss.item()
        val_loss /= len(test_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
        )

    best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    filename = f"{SAVE_DIR}/model_{activation_fn.lower()}.pth"
    torch.save(model.state_dict(), filename)
    print(f"✅ Saved best model with {activation_fn.upper()} activation to: {filename}")

    return model, train_losses, val_losses


# ------------------ Train Models and Plot Loss Curves ------------------
model_tanh, tanh_train_losses, tanh_val_losses = train_and_save("tanh")
model_elu, elu_train_losses, elu_val_losses = train_and_save("elu")


# ------------------ Plot Predictions vs Actual ------------------
def plot_predictions(model, X_data, title, color):
    model.eval()
    with torch.no_grad():
        preds = model(X_data.to(DEVICE)).squeeze().cpu().numpy()
    preds = scaler.inverse_transform(preds.reshape(-1, 1))
    return preds


actual = scaler.inverse_transform(y_test.squeeze().numpy().reshape(-1, 1))
actual_train = scaler.inverse_transform(y_train.squeeze().numpy().reshape(-1, 1))

# Get dates from the dataset
dates = pd.to_datetime(df["Date"])
train_dates = dates[SEQ_LEN : SEQ_LEN + len(actual_train)]
test_dates = dates[
    SEQ_LEN + len(actual_train) : SEQ_LEN + len(actual_train) + len(actual)
]

# Plot 1: Actual vs TANH vs ELU with SMA20 and SMA50
plt.figure(figsize=(12, 6))
plt.plot(train_dates, actual_train, alpha=1, color="#404040")
plt.plot(
    test_dates,
    actual,
    label="Actual Air Liquide Close Price",
    alpha=1,
    color="#404040",
)
plt.plot(
    train_dates,
    plot_predictions(model_tanh, X_train, "TANH", "blue"),
    label="Predicted Air Liquide Close Price (TANH Train Set)",
    alpha=0.9,
    color="#4169E1",
)  # royal blue
plt.plot(
    test_dates,
    plot_predictions(model_tanh, X_test, "TANH", "orange"),
    label="Predicted Air Liquide Close Price (TANH Test Set)",
    alpha=0.9,
    color="#FF7F50",
)  # coral
plt.plot(
    train_dates,
    plot_predictions(model_elu, X_train, "ELU", "red"),
    label="Predicted Air Liquide Close Price (ELU Train Set)",
    alpha=0.9,
    color="#CD5C5C",
)  # indian red
plt.plot(
    test_dates,
    plot_predictions(model_elu, X_test, "ELU", "green"),
    label="Predicted Air Liquide Close Price (ELU Test Set)",
    alpha=0.9,
    color="#3CB371",
)  # medium sea green

# Plotting SMA20 and SMA50
plt.plot(dates, df["SMA20"], label="SMA 20", color="orange", linestyle="--")
plt.plot(dates, df["SMA50"], label="SMA 50", color="green", linestyle="--")

plt.title("Actual vs LSTM Predictions (TANH vs ELU) with SMA20 & SMA50")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
# format x-axis to show years
plt.gca().xaxis.set_major_locator(YearLocator())
plt.gcf().autof
