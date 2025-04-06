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


def evaluate_model(model, X_data, y_true_scaled, model_name="Model"):
    model.eval()
    with torch.no_grad():
        preds = model(X_data.to(DEVICE)).squeeze().cpu().numpy()

    preds_rescaled = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_true_rescaled = target_scaler.inverse_transform(
        y_true_scaled.cpu().numpy().reshape(-1, 1)
    ).flatten()

    mae = mean_absolute_error(y_true_rescaled, preds_rescaled)
    rmse = np.sqrt(mean_squared_error(y_true_rescaled, preds_rescaled))
    r2 = r2_score(y_true_rescaled, preds_rescaled)

    print(f"\n📊 Evaluation for {model_name}:")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ------------------ Config ------------------
# feature engineering config
USE_TECHNICAL_INDICATORS = True  # set to true to use technical indicators as features
USE_PRICE_DERIVED = True         # price-derived features are often strong predictors
USE_TEMPORAL = False             # set to false to simplify model

# Specific features to use (evidence-based selection)
USE_CLOSE = True                 # always keep Close price
USE_RETURN = True                # return is the strongest predictor in many studies
USE_EMA20 = True                 # short-term trend
USE_VOLATILITY = True            # volatility is a strong predictor
USE_RSI = False                  # disable RSI which can be noisy
USE_VOLUME = False               # disable volume which sometimes adds noise

# model architecture config
HIDDEN_SIZES = 50
INPUT_SIZE = 1  # will be updated based on features used
OUTPUT_SIZE = 1
NUM_LAYERS = 1

# training config
SEQ_LEN = 60
BATCH_SIZE = 32
PATIENCE = 15
DROPOUT = 0.2

# change for testing; refer to folders in saved_models readme.txt for epochs and learning rate
EPOCHS = 100
LEARNING_RATE = 0.001

# system config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# change for training; refer to folders in saved_models for folder names
SAVE_DIR = "saved_models/test-newft/"
os.makedirs(SAVE_DIR, exist_ok=True)

# initialize config dictionary
config_dict = {
    "features": {
        "use_technical_indicators": USE_TECHNICAL_INDICATORS,
        "use_price_derived": USE_PRICE_DERIVED,
        "use_temporal": USE_TEMPORAL,
        "use_close": USE_CLOSE,
        "use_ema20": USE_EMA20,
        "use_return": USE_RETURN,
        "use_volatility": USE_VOLATILITY,
        "use_rsi": USE_RSI,
        "use_volume": USE_VOLUME
    },
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

# ------------------ Load and preprocess data ------------------
# you can use sample_data.csv for quick training/testing, NOT FOR PAPER
df = pd.read_csv("datasets/air_liquide.csv")

# feature engineering functions
def add_technical_indicators(df):
    """add technical indicators to the dataframe"""
    # Only calculate the indicators we're using
    if USE_EMA20:
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()  # 20-day exponential moving average
    
    if USE_RETURN:
        # Returns are often the most predictive feature
        df['Return_1D'] = df['Close'].pct_change(1).replace([np.inf, -np.inf], np.nan)
        df['Return_5D'] = df['Close'].pct_change(5).replace([np.inf, -np.inf], np.nan)
    
    if USE_VOLATILITY:
        # Historical volatility (20-day standard deviation of returns)
        df['Volatility_20D'] = df['Close'].pct_change().rolling(window=20).std().replace([np.inf, -np.inf], np.nan)
    
    if USE_RSI:
        # rsi (relative strength index) - momentum oscillator
        delta = df['Close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # handle division by zero - replace zeros with small value
        avg_loss = avg_loss.replace(0, 1e-10)
        
        # calculate RSI and clip to valid range [0, 100]
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].clip(0, 100)  # ensure values stay in valid range
    
    if USE_VOLUME:
        # Volume features
        df['Volume_Change'] = df['Volume'].replace(0, np.nan).pct_change(1).replace([np.inf, -np.inf], np.nan)
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    
    return df

# Add features based on configuration
df = add_technical_indicators(df)

# Drop NaN values that result from calculations
df = df.dropna()

# Determine input size based on features
feature_columns = []

# Always include close price if enabled
if USE_CLOSE:
    feature_columns.append('Close')

# Add technical indicators if enabled
if USE_TECHNICAL_INDICATORS:
    if USE_EMA20:
        feature_columns.append('EMA20')
    if USE_RETURN:
        feature_columns.append('Return_1D')
        feature_columns.append('Return_5D')
    if USE_VOLATILITY:
        feature_columns.append('Volatility_20D')
    if USE_RSI:
        feature_columns.append('RSI')

# Add volume features if enabled
if USE_VOLUME:
    feature_columns.append('Volume_Change')
    feature_columns.append('Volume_MA5')

# update input size based on number of features
INPUT_SIZE = len(feature_columns)
config_dict["model_architecture"]["input_size"] = INPUT_SIZE
config_dict["features"]["feature_columns"] = feature_columns

# Add feature explanations to config
feature_explanations = {
    "Close": {
        "explanation": "Target variable and primary feature - closing stock price for each day",
        "removable": False,
        "impact": "Essential core feature - cannot be removed as it's the target prediction variable",
        "evidence": "Used in nearly all financial time series prediction models as primary signal"
    },
    "EMA20": {
        "explanation": "20-day exponential moving average - captures short-term price trend with emphasis on recent prices",
        "removable": True,
        "impact": "Provides trend information as context for prediction",
        "evidence": "Commonly used in financial forecasting models as a signal smoother"
    },
    "Return_1D": {
        "explanation": "1-day price return - next-day price predictor",
        "removable": True,
        "impact": "The strongest individual predictor in many financial forecasting studies",
        "evidence": "Multiple academic studies show returns are more predictive than absolute prices"
    },
    "Return_5D": {
        "explanation": "5-day price return - short-term momentum",
        "removable": True,
        "impact": "Captures medium-term price momentum which is a strong predictor",
        "evidence": "Momentum effects are well-documented in financial markets"
    },
    "Volatility_20D": {
        "explanation": "20-day historical volatility - price fluctuation magnitude",
        "removable": True,
        "impact": "Critical for understanding market conditions; stock behavior differs in low/high volatility regimes",
        "evidence": "Strong evidence that volatility regimes affect price prediction accuracy"
    },
    "RSI": {
        "explanation": "14-day relative strength index - momentum oscillator measuring speed and change of price movements",
        "removable": True,
        "impact": "Identifies overbought/oversold conditions and potential reversal points",
        "evidence": "Most effective in range-bound markets; can produce false signals during strong trends"
    }
}

config_dict["features"]["feature_explanations"] = feature_explanations

# save updated config
with open(f"{SAVE_DIR}/model_config.json", "w") as f:
    json.dump(config_dict, f, indent=4)

# select features for model input
features = df[feature_columns].values

# Clean features - replace infinities and very large values
print("Checking for problematic values in features...")
original_shape = features.shape

# Convert to pandas for easier manipulation
feature_df = pd.DataFrame(features, columns=feature_columns)

# Check and report problematic features
for col in feature_columns:
    inf_count = np.isinf(feature_df[col]).sum()
    na_count = feature_df[col].isna().sum()
    if inf_count > 0 or na_count > 0:
        print(f"⚠️ Found {inf_count} infinities and {na_count} NaNs in {col}")

# Replace infinity values with NaNs
feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# For each column, fill NaN values with the median of that column
for col in feature_columns:
    if feature_df[col].isna().sum() > 0:
        median_val = feature_df[col].median()
        feature_df[col].fillna(median_val, inplace=True)
        print(f"  - Replaced NaNs in {col} with median: {median_val:.6f}")

# Convert back to numpy array
features = feature_df.values

print(f"Feature cleaning complete. Shape before: {original_shape}, after: {features.shape}")

# scale features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# prepare sequences for LSTM
X, y = [], []
for i in range(SEQ_LEN, len(scaled_features)):
    X.append(scaled_features[i - SEQ_LEN : i])
    y.append(df['Close'].iloc[i])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

# scale target
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y)

# split into train/test sets
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y_scaled[:train_size]
X_test, y_test = X[train_size:], y_scaled[train_size:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)


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
    best_epoch = 0

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
        # comment out this part when using test cases
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch} with validation loss {best_loss:.5f}")
                break

    # Only use the best model state from early stopping
    model.load_state_dict(best_model_state)
    filename = f"{SAVE_DIR}/model_{activation_fn.lower()}.pth"
    torch.save(model.state_dict(), filename)
    print(f"✅ Saved best model with {activation_fn.upper()} activation to: {filename}")

    return model, train_losses, val_losses


# ------------------ Train Models and Plot Loss Curves ------------------
model_tanh, tanh_train_losses, tanh_val_losses = train_and_save("tanh")
model_elu, elu_train_losses, elu_val_losses = train_and_save("elu")

# Plot training/validation loss curves
# TANH Loss Curves
plt.figure(figsize=(12, 6))
plt.semilogy(tanh_train_losses, label="Train Loss (TANH)", linewidth=2)
plt.semilogy(tanh_val_losses, label="Val Loss (TANH)", linewidth=2)
plt.title("Baseline LSTM Model: Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE) - Log Scale")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/tanh_loss_curves.png")
plt.show()

# ELU Loss Curves
plt.figure(figsize=(12, 6))
plt.semilogy(elu_train_losses, label="Train Loss (ELU)", linewidth=2)
plt.semilogy(elu_val_losses, label="Val Loss (ELU)", linewidth=2)
plt.title("Enhanced LSTM-ELU Model: Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE) - Log Scale")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/elu_loss_curves.png")
plt.show()

# Combined Loss Curves Comparison
plt.figure(figsize=(12, 6))
plt.semilogy(
    tanh_train_losses, label="TANH Train Loss", linewidth=2, color="#4169E1"
)  # royal blue
plt.semilogy(
    tanh_val_losses, label="TANH Val Loss", linewidth=2, color="#FF7F50"
)  # coral
plt.semilogy(
    elu_train_losses, label="ELU Train Loss", linewidth=2, color="#CD5C5C"
)  # indian red
plt.semilogy(
    elu_val_losses, label="ELU Val Loss", linewidth=2, color="#3CB371"
)  # medium sea green
plt.title("LSTM Models: Training and Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE) - Log Scale")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/combined_loss_curves.png")
plt.show()


# ------------------ Plot Predictions vs Actual ------------------
def plot_predictions(model, X_data, title, color):
    model.eval()
    with torch.no_grad():
        preds = model(X_data.to(DEVICE)).squeeze().cpu().numpy()
    preds = target_scaler.inverse_transform(preds.reshape(-1, 1))
    return preds


actual = target_scaler.inverse_transform(y_test.squeeze().numpy().reshape(-1, 1))
actual_train = target_scaler.inverse_transform(y_train.squeeze().numpy().reshape(-1, 1))

# Get dates from the dataset
dates = pd.to_datetime(df["Date"])
train_dates = dates[SEQ_LEN : SEQ_LEN + len(actual_train)]
test_dates = dates[
    SEQ_LEN + len(actual_train) : SEQ_LEN + len(actual_train) + len(actual)
]

# Plot 1: Actual vs TANH vs ELU
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
plt.title("Actual vs LSTM Predictions (TANH vs ELU)")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
# format x-axis to show years
plt.gca().xaxis.set_major_locator(YearLocator())
plt.gcf().autofmt_xdate()  # rotate and align the tick labels so they look better
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/actual-tanh-elu.png")
plt.show()

# Plot 2: Actual vs TANH
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
plt.title("Actual vs LSTM Predictions (TANH)")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
# format x-axis to show years
plt.gca().xaxis.set_major_locator(YearLocator())
plt.gcf().autofmt_xdate()  # rotate and align the tick labels so they look better
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/actual-tanh.png")
plt.show()

# Plot 3: Actual vs ELU
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
plt.title("Actual vs LSTM Predictions (ELU)")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
# format x-axis to show years
plt.gca().xaxis.set_major_locator(YearLocator())
plt.gcf().autofmt_xdate()  # rotate and align the tick labels so they look better
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/actual-elu.png")
plt.show()

# ------------------ Evaluation Metrics ------------------
# Get evaluation results
tanh_results = evaluate_model(model_tanh, X_test, y_test, model_name="TANH Model")
elu_results = evaluate_model(model_elu, X_test, y_test, model_name="ELU Model")

# Convert numpy float32 to native float for JSON serialization
results_dict = {
    "TANH Model": {k: float(v) for k, v in tanh_results.items()},
    "ELU Model": {k: float(v) for k, v in elu_results.items()},
}

# Save as JSON
with open(f"{SAVE_DIR}/evaluation_metrics.json", "w") as f:
    json.dump(results_dict, f, indent=4)

# Create and save table visualization
metrics_df = pd.DataFrame(
    {
        "Metric": ["MAE", "RMSE", "R²"],
        "TANH Model": [
            f"{tanh_results['MAE']:.4f}",
            f"{tanh_results['RMSE']:.4f}",
            f"{tanh_results['R2']:.4f}",
        ],
        "ELU Model": [
            f"{elu_results['MAE']:.4f}",
            f"{elu_results['RMSE']:.4f}",
            f"{elu_results['R2']:.4f}",
        ],
    }
)

# Plot table
plt.figure(figsize=(8, 4))
plt.axis("off")
table = plt.table(
    cellText=metrics_df.values,
    colLabels=metrics_df.columns,
    cellLoc="center",
    loc="center",
    colColours=["#f2f2f2"] * len(metrics_df.columns),
    cellColours=[["#ffffff"] * len(metrics_df.columns)] * len(metrics_df),
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

plt.title("Model Evaluation Metrics Comparison", pad=20)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/evaluation_metrics_table.png", bbox_inches="tight", dpi=300)
plt.show()
