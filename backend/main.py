import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from src.models.custom_lstm import CustomLSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_data, y_true_scaled, model_name="Model"):
    model.eval()
    with torch.no_grad():
        preds = model(X_data.to(DEVICE)).squeeze().cpu().numpy()

    preds_rescaled = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_true_rescaled = scaler.inverse_transform(y_true_scaled.cpu().numpy().reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true_rescaled, preds_rescaled)
    rmse = np.sqrt(mean_squared_error(y_true_rescaled, preds_rescaled))
    r2 = r2_score(y_true_rescaled, preds_rescaled)

    print(f"\n📊 Evaluation for {model_name}:")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ------------------ Config ------------------
SEQ_LEN = 60
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)


# ------------------ Load and preprocess data ------------------
df = pd.read_csv("datasets/air_liquide.csv")
close_prices = df['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(close_prices)

X, y = [], []
for i in range(SEQ_LEN, len(scaled_prices)):
    X.append(scaled_prices[i - SEQ_LEN:i])
    y.append(scaled_prices[i])

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


# ------------------ Define Model ------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, activation_fn="tanh"):
        super(LSTMModel, self).__init__()
        self.lstm1 = CustomLSTM(input_size, hidden_size, num_layers=1,
                                hidden_activation=activation_fn, cell_activation=activation_fn)
        self.dropout1 = nn.Dropout(DROPOUT)

        self.lstm2 = CustomLSTM(hidden_size, hidden_size, num_layers=1,
                                hidden_activation=activation_fn, cell_activation=activation_fn)
        self.dropout2 = nn.Dropout(DROPOUT)

        self.lstm3 = CustomLSTM(hidden_size, hidden_size, num_layers=1,
                                hidden_activation=activation_fn, cell_activation=activation_fn)
        self.dropout3 = nn.Dropout(DROPOUT)

        self.fc = nn.Linear(hidden_size, 1)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
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

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_state)
    filename = f"{SAVE_DIR}/model_{activation_fn.lower()}.pth"
    torch.save(model.state_dict(), filename)
    print(f"✅ Saved best model with {activation_fn.upper()} activation to: {filename}")

    return model, train_losses, val_losses


# ------------------ Train Models and Plot Loss Curves ------------------
model_tanh, tanh_train_losses, tanh_val_losses = train_and_save("tanh")
model_elu, elu_train_losses, elu_val_losses = train_and_save("elu")

# Plot training/validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(tanh_train_losses, label="Train Loss (TANH)")
plt.plot(tanh_val_losses, label="Val Loss (TANH)")
plt.plot(elu_train_losses, label="Train Loss (ELU)")
plt.plot(elu_val_losses, label="Val Loss (ELU)")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("loss_curves.png")


# ------------------ Plot Predictions vs Actual ------------------
def plot_predictions(model, X_data, title, color):
    model.eval()
    with torch.no_grad():
        preds = model(X_data.to(DEVICE)).squeeze().cpu().numpy()
    preds = scaler.inverse_transform(preds.reshape(-1, 1))
    return preds

actual = scaler.inverse_transform(y_test.squeeze().numpy().reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(actual, label="Actual", linestyle='--', alpha=0.7)
plt.plot(plot_predictions(model_tanh, X_test, "TANH", "orange"), label="TANH", color="orange")
plt.plot(plot_predictions(model_elu, X_test, "ELU", "green"), label="ELU", color="green")
plt.title("Model Predictions vs Actual Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------ Evaluation Metrics ------------------
evaluate_model(model_tanh, X_test, y_test, model_name="TANH Model")
evaluate_model(model_elu, X_test, y_test, model_name="ELU Model")
