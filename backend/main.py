import os
import torch
import pandas as pd
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from src.models.custom_lstm import CustomLSTM
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed()


def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length, :-1])  # Use Open and Volume as features
        y.append(data[i + sequence_length, -1])  # Close is the target
    return np.array(X), np.array(y).reshape(-1, 1)


def save_model(model, sequence_length, activation_function, model_name, config):
    model_dir = f"models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(
        model_dir, f"model_seq{sequence_length}_{activation_function}.pt"
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    # Save the config to a JSON file
    config_file = os.path.join(model_dir, "config.json")
    if os.path.exists(config_file):
        return
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config saved in {config_file}")


@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    sequence_length: int = Form(...),
    model_name: str = Form(...),
    epochs: int = Form(50),
    train_split: float = Form(0.8),
    batch_size: int = Form(128),
    hidden_size: int = Form(128),
    learning_rate: float = Form(0.001),
):
    try:
        # Read the CSV file
        df = pd.read_csv(file.file)

        # Ensure the required columns exist
        required_columns = ["Open", "Volume", "Close"]
        if not all(col in df.columns for col in required_columns):
            return JSONResponse(
                {"error": f"Dataset must contain columns: {required_columns}"}
            )

        data = df[["Open", "Volume", "Close"]].dropna()
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.values)

        train_data, test_data = train_test_split(
            data_scaled, test_size=(1 - train_split), shuffle=False
        )
        X_train, y_train = create_sequences(train_data, sequence_length)
        X_test, y_test = create_sequences(test_data, sequence_length)

        train_dataloader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        test_dataloader = DataLoader(
            TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = X_train.shape[2]
        output_size = y_train.shape[1]

        results = {}
        config = {
            "sequence_length": sequence_length,
            "model_name": model_name,
            "epochs": epochs,
            "train_split": train_split,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
        }
        for activation_function in ["ELU", "Tanh"]:
            set_seed()
            model = CustomLSTM(
                input_size,
                hidden_size,
                output_size,
                getattr(torch.nn, activation_function)(),
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.MSELoss()

            for epoch in range(epochs):
                model.train()
                for batch in train_dataloader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), targets)
                    loss.backward()
                    optimizer.step()

            model.eval()
            all_predictions, all_actuals = [], []
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    outputs = model(inputs.to(device)).cpu().numpy()
                    all_predictions.extend(outputs)
                    all_actuals.extend(targets.numpy())

            all_predictions = scaler.inverse_transform(
                np.hstack((np.zeros((len(all_predictions), 2)), all_predictions))
            )[:, -1]
            all_actuals = scaler.inverse_transform(
                np.hstack((np.zeros((len(all_actuals), 2)), all_actuals))
            )[:, -1]

            results[activation_function] = {
                "predictions": all_predictions.tolist(),
                "actuals": all_actuals.tolist(),
                "metrics": {
                    "rmse": mean_squared_error(
                        all_actuals, all_predictions, squared=False
                    ),
                    "mae": mean_absolute_error(all_actuals, all_predictions),
                    "mse": mean_squared_error(all_actuals, all_predictions),
                    "r2": r2_score(all_actuals, all_predictions),
                },
            }
            save_model(model, sequence_length, activation_function, model_name, config)

        return JSONResponse({"message": "Training complete.", "results": results})

    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/models")
async def list_models():
    model_dir = "models"  # Folder containing saved models
    try:
        # Check if the directory exists
        if not os.path.exists(model_dir):
            return JSONResponse(
                {"error": f"Directory '{model_dir}' not found."}, status_code=404
            )

        # Get the list of all subdirectories (model names)
        model_dirs = [
            f
            for f in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, f))
        ]

        if not model_dirs:
            return JSONResponse({"message": "No models found in the directory."})

        models_configs = []  # List to store all model configs

        for model_name in model_dirs:
            model_path = os.path.join(model_dir, model_name)
            config_file = os.path.join(model_path, "config.json")

            # Check if the config.json file exists in the model directory
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config = json.load(f)
                models_configs.append(config)  # Add the config to the list
            else:
                models_configs.append(
                    {"error": "config.json not found"}
                )  # Add an error message

        return JSONResponse({"models": models_configs})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
