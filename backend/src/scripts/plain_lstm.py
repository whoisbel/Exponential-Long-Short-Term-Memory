import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import copy
import os
import sys
from datetime import datetime

# === 1. Create Output Directory with Timestamp ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f"saved_models/plain-lstm-{timestamp}/"
os.makedirs(SAVE_DIR, exist_ok=True)

# === 2. Load Dataset ===
df = pd.read_csv("datasets/air_liquide.csv")  # Make sure this has a 'Close' column
closing_prices = df['Close'].values.reshape(-1, 1)

# === 3. Normalize ===
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(closing_prices)

# === 4. Sequence Creation ===
SEQ_LEN = 60
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

x_all, y_all = create_sequences(normalized_data, SEQ_LEN)

# === 5. Train-Test Split ===
train_size = int(0.8 * len(x_all))
x_train, y_train = x_all[:train_size], y_all[:train_size]
x_test, y_test = x_all[train_size:], y_all[train_size:]

# === 6. Dataset & Dataloader ===
class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.x = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_loader = DataLoader(StockDataset(x_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(StockDataset(x_test, y_test), batch_size=32, shuffle=False)

# === 7. LSTM Model ===
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(1, 50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(50, 50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(50, 50, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        return self.fc(x[:, -1, :])

# === 8. Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = [], []
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float("inf")
patience = 15
counter = 0

# === 9. Train Loop ===
EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x_batch.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in test_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            val_output = model(x_val)
            val_loss += criterion(val_output, y_val).item() * x_val.size(0)

    val_loss /= len(test_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping.")
            break

# === 10. Load Best Model ===
model.load_state_dict(best_model_wts)

# === 11. Prediction ===
model.eval()
with torch.no_grad():
    train_preds = model(torch.tensor(x_train).float().to(device)).cpu().numpy()
    test_preds = model(torch.tensor(x_test).float().to(device)).cpu().numpy()

train_preds_inv = scaler.inverse_transform(train_preds)
test_preds_inv = scaler.inverse_transform(test_preds)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# === 12. Save Loss Curve ===
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("MSE Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
plt.close()

# === 13. Evaluation Metrics ===
rmse = np.sqrt(mean_squared_error(y_test_inv, test_preds_inv))
mae = mean_absolute_error(y_test_inv, test_preds_inv)
r2 = r2_score(y_test_inv, test_preds_inv)

results = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R²'],
    'Value': [rmse, mae, r2]
})
print("\nEvaluation Results Table:")
print(results.to_string(index=False))
results.to_csv(os.path.join(SAVE_DIR, "evaluation_metrics.csv"), index=False)

# === 14. Continuous Prediction Plot ===
combined_preds = np.concatenate([train_preds_inv, test_preds_inv])
combined_actual = np.concatenate([y_train_inv, y_test_inv])

plt.figure(figsize=(14, 6))
plt.plot(combined_actual, label="Actual", color='black')
plt.plot(range(len(train_preds_inv)), train_preds_inv, label="Train Prediction", color='blue')
plt.plot(range(len(train_preds_inv), len(combined_preds)), test_preds_inv, label="Test Prediction", color='orange')
plt.axvline(x=len(train_preds_inv), color='gray', linestyle='--', label="Train/Test Split")
plt.title("Stock Price Prediction (Train + Test)")
plt.xlabel("Time Steps")
plt.ylabel("Price (EUR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "prediction_plot.png"))
plt.close()

# === 15. Save Model ===
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))

# === 16. Save Predictions ===
combined_df = pd.DataFrame({
    'Actual': combined_actual.flatten(),
    'Predicted': combined_preds.flatten(),
    'Split': ['Train'] * len(train_preds_inv) + ['Test'] * len(test_preds_inv)
})
combined_df.to_csv(os.path.join(SAVE_DIR, "predictions.csv"), index=False)
