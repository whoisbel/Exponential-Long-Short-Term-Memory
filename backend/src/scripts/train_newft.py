import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator
import json
import argparse

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.models.custom_lstm import LSTMModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------ Feature Profiles for Testing ------------------
# Define different sets of features to easily switch between for experiments.
# Each profile represents a specific combination of features based on research.
FEATURE_PROFILES = {
    'baseline': { # rationale: baseline model for benchmarking (Wen et al., 2023) showed simple features often provide strong baselines
        'USE_CLOSE': True, 'USE_RETURN': True, 'USE_EMA20': False, 'USE_EMA50': False,
        'USE_MACD': False, 'USE_VOLATILITY': False, 'USE_RSI': False, 'USE_VOLUME': False,
        'USE_TECHNICAL_INDICATORS': True, 'USE_PRICE_DERIVED': True
    },
    'mean_variance': { # rationale: Jiang et al. (2021) "Deep Learning in Asset Pricing" demonstrated returns + volatility were primary drivers
        'USE_CLOSE': True, 'USE_RETURN': True, 'USE_EMA20': False, 'USE_EMA50': False,
        'USE_MACD': False, 'USE_VOLATILITY': True, 'USE_RSI': False, 'USE_VOLUME': False,
        'USE_TECHNICAL_INDICATORS': True, 'USE_PRICE_DERIVED': True
    },
    'volume_momentum': { # rationale: Yang et al. (2022) "Enhancing Stock Price Prediction with Volume-Price Correlation Networks" found synergy between volume and momentum
        'USE_CLOSE': True, 'USE_RETURN': True, 'USE_EMA20': False, 'USE_EMA50': False,
        'USE_MACD': True, 'USE_VOLATILITY': False, 'USE_RSI': True, 'USE_VOLUME': True,
        'USE_TECHNICAL_INDICATORS': True, 'USE_PRICE_DERIVED': True
    },
    'focused_oscillators': { # rationale: Zhang & Chen (2023) "Feature Selection for Stock Prediction" found oscillators (MACD, RSI) particularly effective
        'USE_CLOSE': True, 'USE_RETURN': True, 'USE_EMA20': False, 'USE_EMA50': False,
        'USE_MACD': True, 'USE_VOLATILITY': False, 'USE_RSI': True, 'USE_VOLUME': False,
        'USE_TECHNICAL_INDICATORS': True, 'USE_PRICE_DERIVED': True
    },
    'comprehensive': { # rationale: Liu et al. (2023) "Interpretable multivariate time series transformer" showed benefit of diverse indicators for robust forecasting
        'USE_CLOSE': True, 'USE_RETURN': True, 'USE_EMA20': True, 'USE_EMA50': True,
        'USE_MACD': True, 'USE_VOLATILITY': True, 'USE_RSI': True, 'USE_VOLUME': True,
        'USE_TECHNICAL_INDICATORS': True, 'USE_PRICE_DERIVED': True
    }
}

# Array of profile names for iteration
PROFILE_NAMES = list(FEATURE_PROFILES.keys())

# ------------------ Main Execution ------------------
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LSTM models for stock price prediction')
    parser.add_argument('--all-profiles', action='store_true', help='Train on all feature profiles')
    parser.add_argument('--profile', type=str, default='baseline', 
                        choices=PROFILE_NAMES,
                        help='Specific feature profile to train on')
    
    args = parser.parse_args()
    
    # Create base save directory
    base_save_dir = "saved_models/"
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Load dataset once
    df = pd.read_csv("datasets/air_liquide.csv")
    
    # Set device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Define model hyperparameters
    HIDDEN_SIZES = 50       # Number of hidden units in LSTM
    NUM_LAYERS = 1          # Number of LSTM layers within each block
    DROPOUT = 0.3           # Dropout rate for regularization
    SEQ_LEN = 60            # Sequence length for LSTM input (days of history)
    BATCH_SIZE = 16         # Training batch size
    EPOCHS = 100            # Maximum number of training epochs
    LEARNING_RATE = 0.001  # Learning rate for optimizer
    PATIENCE = 15           # Early stopping patience
    
    # Feature explanation dictionary for config - documents each feature's purpose and evidence
    feature_explanations = {
        "Close": {
            "explanation": "Target variable and primary feature - closing stock price for each day",
            "removable": False,
            "evidence": "Used in nearly all financial time series prediction models as primary signal"
        },
        "EMA20": {
            "explanation": "20-day exponential moving average - captures short-term price trend",
            "removable": True,
            "evidence": "Commonly used in financial forecasting models as a signal smoother"
        },
        "Return_1D": {
            "explanation": "1-day price return - next-day price predictor",
            "removable": True,
            "evidence": "Multiple academic studies show returns are more predictive than absolute prices"
        },
        "Return_5D": {
            "explanation": "5-day price return - short-term momentum",
            "removable": True,
            "evidence": "Momentum effects are well-documented in financial markets"
        },
        "Volatility_20D": {
            "explanation": "20-day historical volatility - price fluctuation magnitude",
            "removable": True,
            "evidence": "Strong evidence that volatility regimes affect price prediction accuracy"
        },
        "RSI": {
            "explanation": "14-day relative strength index - momentum oscillator",
            "removable": True,
            "evidence": "Most effective in range-bound markets"
        },
        "EMA50": {
            "explanation": "50-day exponential moving average - captures medium-term price trend",
            "removable": True,
            "evidence": "Widely used to identify medium-term trend direction"
        },
        "MACD_line": {
            "explanation": "MACD line (12-ema minus 26-ema) - momentum and trend direction",
            "removable": True,
            "evidence": "Core technical indicator for trend following strategies"
        },
        "MACD_signal": {
            "explanation": "Signal line (9-day ema of macd line) - smoother version for trigger signals",
            "removable": True,
            "evidence": "Used for generating buy/sell signals"
        }
    }
    
    def add_technical_indicators(df, profile_config):
        """Add technical indicators to the dataframe based on profile configuration"""
        df_profile = df.copy()
        
        # Only calculate the indicators we're using for efficiency
        if profile_config['USE_EMA20']:
            df_profile['EMA20'] = df_profile['Close'].ewm(span=20, adjust=False).mean()  # 20-day exponential moving average
        
        if profile_config['USE_EMA50']:
            df_profile['EMA50'] = df_profile['Close'].ewm(span=50, adjust=False).mean()  # 50-day exponential moving average
        
        if profile_config['USE_MACD']:
            # Calculate the short-term (12-day) and long-term (26-day) EMAs
            ema12 = df_profile['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df_profile['Close'].ewm(span=26, adjust=False).mean()
            # Calculate the MACD line (difference between short and long EMAs)
            df_profile['MACD_line'] = ema12 - ema26
            # Calculate the signal line (9-day EMA of the MACD line)
            df_profile['MACD_signal'] = df_profile['MACD_line'].ewm(span=9, adjust=False).mean()
        
        if profile_config['USE_RETURN']:
            # Returns are often the most predictive feature
            df_profile['Return_1D'] = df_profile['Close'].pct_change(1).replace([np.inf, -np.inf], np.nan)
            df_profile['Return_5D'] = df_profile['Close'].pct_change(5).replace([np.inf, -np.inf], np.nan)
        
        if profile_config['USE_VOLATILITY']:
            # Historical volatility (20-day standard deviation of returns)
            df_profile['Volatility_20D'] = df_profile['Close'].pct_change().rolling(window=20).std().replace([np.inf, -np.inf], np.nan)
        
        if profile_config['USE_RSI']:
            # RSI (Relative Strength Index) - momentum oscillator
            delta = df_profile['Close'].diff()
            gain = delta.mask(delta < 0, 0)
            loss = -delta.mask(delta > 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            # Handle division by zero - replace zeros with small value
            avg_loss = avg_loss.replace(0, 1e-10)
            
            # Calculate RSI and clip to valid range [0, 100]
            rs = avg_gain / avg_loss
            df_profile['RSI'] = 100 - (100 / (1 + rs))
            df_profile['RSI'] = df_profile['RSI'].clip(0, 100)  # Ensure values stay in valid range
        
        if profile_config['USE_VOLUME']:
            # Volume features
            df_profile['Volume_Change'] = df_profile['Volume'].replace(0, np.nan).pct_change(1).replace([np.inf, -np.inf], np.nan)
            df_profile['Volume_MA5'] = df_profile['Volume'].rolling(window=5).mean()
        
        return df_profile.dropna()
    
    def get_feature_columns(profile_config):
        """Get the list of feature columns based on profile configuration"""
        feature_columns = []
        
        # Always include close price if enabled
        if profile_config['USE_CLOSE']:
            feature_columns.append('Close')
        
        # Add technical indicators if enabled
        if profile_config['USE_TECHNICAL_INDICATORS']:
            if profile_config['USE_EMA20']:
                feature_columns.append('EMA20')
            if profile_config['USE_RETURN']:
                feature_columns.append('Return_1D')
                feature_columns.append('Return_5D')
            if profile_config['USE_VOLATILITY']:
                feature_columns.append('Volatility_20D')
            if profile_config['USE_RSI']:
                feature_columns.append('RSI')
            if profile_config['USE_EMA50']:
                feature_columns.append('EMA50')
            if profile_config['USE_MACD']:
                feature_columns.append('MACD_line')
                feature_columns.append('MACD_signal')
        
        # Add volume features if enabled
        if profile_config['USE_VOLUME']:
            feature_columns.append('Volume_Change')
            feature_columns.append('Volume_MA5')
        
        return feature_columns
    
    def train_and_save(model, train_loader, test_loader, activation_fn, save_dir, target_scaler):
        """Train and save a model with the specified activation function"""
        print(f"\n=== Training with activation: {activation_fn.upper()} ===")
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        
        train_losses = []
        val_losses = []
        
        try:
            for epoch in range(EPOCHS):
                # Training phase
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
                
                # Validation phase
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
                
                # Early stopping check
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
            
            # Load best model
            model.load_state_dict(best_model_state)
            filename = f"{save_dir}/model_{activation_fn.lower()}.pth"
            torch.save(model.state_dict(), filename)
            print(f"✅ Saved best model with {activation_fn.upper()} activation to: {filename}")
            
            return model, train_losses, val_losses
            
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            raise
            
        finally:
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def plot_loss_curves(train_losses, val_losses, title, save_path, activation=None):
        """Plot and save loss curves (training and validation)"""
        plt.figure(figsize=(12, 6))
        
        if activation:
            # Single activation function plot
            plt.semilogy(train_losses, label=f"Train Loss ({activation})", linewidth=2)
            plt.semilogy(val_losses, label=f"Val Loss ({activation})", linewidth=2)
        else:
            # Combined plot with multiple activation functions
            plt.semilogy(train_losses[0], label="TANH Train Loss", linewidth=2, color="#4169E1")  # royal blue
            plt.semilogy(val_losses[0], label="TANH Val Loss", linewidth=2, color="#FF7F50")  # coral
            plt.semilogy(train_losses[1], label="ELU Train Loss", linewidth=2, color="#CD5C5C")  # indian red
            plt.semilogy(val_losses[1], label="ELU Val Loss", linewidth=2, color="#3CB371")  # medium sea green
        
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE) - Log Scale")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def create_evaluation_table(results_dict, save_path):
        """Create and save a table visualization of evaluation metrics"""
        metrics_df = pd.DataFrame(
            {
                "Metric": ["MAE", "RMSE", "R²"],
                "TANH Model": [
                    f"{results_dict['TANH Model']['MAE']:.4f}",
                    f"{results_dict['TANH Model']['RMSE']:.4f}",
                    f"{results_dict['TANH Model']['R2']:.4f}",
                ],
                "ELU Model": [
                    f"{results_dict['ELU Model']['MAE']:.4f}",
                    f"{results_dict['ELU Model']['RMSE']:.4f}",
                    f"{results_dict['ELU Model']['R2']:.4f}",
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
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        print(f"📊 Saved evaluation metrics table to: {save_path}")

    def evaluate_model(model, X_data, y_true_scaled, target_scaler, model_name="Model"):
        """Evaluate model performance using different metrics"""
        model.eval()
        with torch.no_grad():
            preds = model(X_data.to(DEVICE)).squeeze().cpu().numpy()
        
        # Convert scaled predictions back to original values
        preds_rescaled = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        y_true_rescaled = target_scaler.inverse_transform(
            y_true_scaled.cpu().numpy().reshape(-1, 1)
        ).flatten()
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_true_rescaled, preds_rescaled)
        rmse = np.sqrt(mean_squared_error(y_true_rescaled, preds_rescaled))
        r2 = r2_score(y_true_rescaled, preds_rescaled)
        
        mean_actual = np.mean(y_true_rescaled)
        mae_percent = (mae / mean_actual) * 100
        rmse_percent = (rmse / mean_actual) * 100
        
        print(f"\n📊 Evaluation for {model_name}:")
        print(f"MAE  : {mae:.4f} ({mae_percent:.2f}%)")
        print(f"RMSE : {rmse:.4f} ({rmse_percent:.2f}%)")
        print(f"R²   : {r2:.4f}")
        
        return {
        "MAE": mae,
        "MAE_percent": mae_percent,
        "RMSE": rmse,
        "RMSE_percent": rmse_percent,
        "R2": r2
        }
    
    def generate_prediction_plots(model, X_train, X_test, y_train, y_test, df_profile, target_scaler, seq_len, profile_name, model_name, save_path, colors=None):
        """Generate and save plots of model predictions vs actual values"""
        model.eval()
        
        # Get predictions
        with torch.no_grad():
            train_preds = model(X_train.to(DEVICE)).squeeze().cpu().numpy()
            test_preds = model(X_test.to(DEVICE)).squeeze().cpu().numpy()
        
        # Rescale predictions and actual values
        train_preds = target_scaler.inverse_transform(train_preds.reshape(-1, 1)).flatten()
        test_preds = target_scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
        
        train_actual = target_scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1)).flatten()
        test_actual = target_scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()
        
        # Get dates from the dataset
        if "Date" in df_profile.columns:
            dates = pd.to_datetime(df_profile["Date"])
            train_dates = dates[seq_len:seq_len + len(train_actual)]
            test_dates = dates[seq_len + len(train_actual):seq_len + len(train_actual) + len(test_actual)]
        else:
            # Create date range if no Date column exists
            end_date = pd.Timestamp.today()
            dates = pd.date_range(end=end_date, periods=len(df_profile), freq='D')
            train_dates = dates[seq_len:seq_len + len(train_actual)]
            test_dates = dates[seq_len + len(train_actual):seq_len + len(train_actual) + len(test_actual)]
        
        # Default colors if not provided
        if colors is None:
            colors = {
                'train_actual': '#404040',
                'test_actual': '#404040',
                'train_pred': '#4169E1',  # royal blue
                'test_pred': '#FF7F50'    # coral
            }
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot actual values
        plt.plot(train_dates, train_actual, alpha=0.7, color=colors['train_actual'])
        plt.plot(
            test_dates,
            test_actual,
            label="Actual Price",
            alpha=1.0,
            color=colors['test_actual'],
        )
        
        # Plot predictions
        plt.plot(
            train_dates,
            train_preds,
            label=f"Predicted Price ({model_name} Train Set)",
            alpha=0.9,
            color=colors['train_pred'],
        )
        plt.plot(
            test_dates,
            test_preds,
            label=f"Predicted Price ({model_name} Test Set)",
            alpha=0.9,
            color=colors['test_pred'],
        )
        
        # Format plot
        plt.title(f"{profile_name.capitalize()} - Actual vs {model_name} Predictions")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.gcf().autofmt_xdate()  # rotate and align the tick labels
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"📈 Saved {model_name} prediction plot to: {save_path}")
        
        return {
            'train_dates': train_dates,
            'test_dates': test_dates,
            'train_actual': train_actual,
            'test_actual': test_actual,
            'train_preds': train_preds,
            'test_preds': test_preds
        }
    
    # Determine which profiles to train - all profiles or just one
    profiles_to_train = PROFILE_NAMES if args.all_profiles else [args.profile]
    
    # Train each profile one by one
    for profile_name in profiles_to_train:
        print(f"\n\n{'='*50}")
        print(f"   TRAINING WITH FEATURE PROFILE: {profile_name.upper()}")
        print(f"{'='*50}")
        
        # Create profile-specific save directory
        profile_save_dir = f"{base_save_dir}/{profile_name}"
        os.makedirs(profile_save_dir, exist_ok=True)
        
        try:
            # Get profile configuration
            profile_config = FEATURE_PROFILES[profile_name]
            
            # Calculate features for this profile
            df_profile = add_technical_indicators(df, profile_config)
            
            # Get feature columns for this profile
            feature_columns = get_feature_columns(profile_config)
            
            # Make sure we have at least one feature
            if len(feature_columns) == 0:
                print(f"❌ No features enabled in profile {profile_name}. Skipping.")
                continue
                
            # Update input size based on number of features
            input_size = len(feature_columns)
            
            # Create config dictionary for this profile
            config_dict = {
                "features": {
                    "profile_name": profile_name,
                    "use_technical_indicators": profile_config['USE_TECHNICAL_INDICATORS'],
                    "use_price_derived": profile_config['USE_PRICE_DERIVED'],
                    "use_close": profile_config['USE_CLOSE'],
                    "use_ema20": profile_config['USE_EMA20'],
                    "use_return": profile_config['USE_RETURN'],
                    "use_volatility": profile_config['USE_VOLATILITY'],
                    "use_rsi": profile_config['USE_RSI'],
                    "use_volume": profile_config['USE_VOLUME'],
                    "use_ema50": profile_config['USE_EMA50'],
                    "use_macd": profile_config['USE_MACD'],
                    "feature_columns": feature_columns,
                    "feature_explanations": {key: feature_explanations[key] for key in feature_columns if key in feature_explanations}
                },
                "model_architecture": {
                    "hidden_sizes": HIDDEN_SIZES,
                    "input_size": input_size,
                    "output_size": 1,
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
                "system": {
                    "device": str(DEVICE), 
                    "save_dir": profile_save_dir
                },
            }
            
            # Save config for reproducibility
            with open(f"{profile_save_dir}/model_config.json", "w") as f:
                json.dump(config_dict, f, indent=4)
            
            # Prepare data
            features = df_profile[feature_columns].values
            
            # Clean features - replace infinities and very large values
            print("Checking for problematic values in features...")
            feature_df = pd.DataFrame(features, columns=feature_columns)
            
            # Replace infinity values with NaNs and fill with median
            feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            for col in feature_columns:
                if feature_df[col].isna().sum() > 0:
                    median_val = feature_df[col].median()
                    feature_df[col].fillna(median_val, inplace=True)
                    print(f"  - Replaced NaNs in {col} with median: {median_val:.6f}")
            
            # Convert back to numpy array
            features = feature_df.values
            
            # Scale features
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Prepare sequences for LSTM
            X, y = [], []
            for i in range(SEQ_LEN, len(scaled_features)):
                X.append(scaled_features[i - SEQ_LEN : i])
                y.append(df_profile['Close'].iloc[i])
            
            X = np.array(X)
            y = np.array(y).reshape(-1, 1)
            
            # Scale target
            target_scaler = MinMaxScaler()
            y_scaled = target_scaler.fit_transform(y)
            
            # Split into train/test sets (80/20)
            train_size = int(0.8 * len(X))
            X_train, y_train = X[:train_size], y_scaled[:train_size]
            X_test, y_test = X[train_size:], y_scaled[train_size:]
            
            # Convert to PyTorch tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            
            # Create data loaders
            train_ds = TensorDataset(X_train, y_train)
            test_ds = TensorDataset(X_test, y_test)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
            
            # Train models
            print(f"Training models for profile: {profile_name}")
            
            # TANH model
            model_tanh = LSTMModel(
                input_size=input_size, 
                hidden_size=HIDDEN_SIZES, 
                dropout=DROPOUT,
                activation_fn="tanh"
            ).to(DEVICE)
            
            model_tanh, tanh_train_losses, tanh_val_losses = train_and_save(
                model_tanh, train_loader, test_loader, "tanh", profile_save_dir, target_scaler
            )
            
            # ELU model
            model_elu = LSTMModel(
                input_size=input_size, 
                hidden_size=HIDDEN_SIZES, 
                dropout=DROPOUT,
                activation_fn="elu"
            ).to(DEVICE)
            
            model_elu, elu_train_losses, elu_val_losses = train_and_save(
                model_elu, train_loader, test_loader, "elu", profile_save_dir, target_scaler
            )
            
            # Generate plots
            # TANH Loss Curves
            plot_loss_curves(
                tanh_train_losses, 
                tanh_val_losses, 
                f"{profile_name.capitalize()} - TANH: Training and Validation Loss",
                f"{profile_save_dir}/tanh_loss_curves.png",
                "TANH"
            )
            
            # ELU Loss Curves
            plot_loss_curves(
                elu_train_losses, 
                elu_val_losses, 
                f"{profile_name.capitalize()} - ELU: Training and Validation Loss",
                f"{profile_save_dir}/elu_loss_curves.png",
                "ELU"
            )
            
            # Combined Loss Curves
            plot_loss_curves(
                [tanh_train_losses, elu_train_losses],
                [tanh_val_losses, elu_val_losses],
                f"{profile_name.capitalize()} - LSTM Models: Loss Comparison",
                f"{profile_save_dir}/combined_loss_curves.png"
            )
            
            # Evaluate models
            tanh_results = evaluate_model(
                model_tanh, X_test, y_test, target_scaler, f"TANH Model ({profile_name})"
            )
            
            elu_results = evaluate_model(
                model_elu, X_test, y_test, target_scaler, f"ELU Model ({profile_name})"
            )
            
            # Save results
            results_dict = {
                "TANH Model": {k: float(v) for k, v in tanh_results.items()},
                "ELU Model": {k: float(v) for k, v in elu_results.items()},
            }
            
            with open(f"{profile_save_dir}/evaluation_metrics.json", "w") as f:
                json.dump(results_dict, f, indent=4)
            
            # Create evaluation metrics table
            create_evaluation_table(
                results_dict, 
                f"{profile_save_dir}/evaluation_metrics_table.png"
            )
            
            # Generate prediction plots for TANH model
            tanh_plot_data = generate_prediction_plots(
                model_tanh, X_train, X_test, y_train, y_test, 
                df_profile, target_scaler, SEQ_LEN, profile_name, 
                "TANH", f"{profile_save_dir}/actual-tanh.png",
                colors={
                    'train_actual': '#404040',
                    'test_actual': '#404040',
                    'train_pred': '#4169E1',  # royal blue
                    'test_pred': '#FF7F50'    # coral
                }
            )
            
            # Generate prediction plots for ELU model
            elu_plot_data = generate_prediction_plots(
                model_elu, X_train, X_test, y_train, y_test, 
                df_profile, target_scaler, SEQ_LEN, profile_name, 
                "ELU", f"{profile_save_dir}/actual-elu.png",
                colors={
                    'train_actual': '#404040',
                    'test_actual': '#404040',
                    'train_pred': '#CD5C5C',  # indian red
                    'test_pred': '#3CB371'    # medium sea green
                }
            )
            
            # Generate combined prediction plot
            plt.figure(figsize=(12, 6))
            
            # Plot actual values
            plt.plot(tanh_plot_data['train_dates'], tanh_plot_data['train_actual'], alpha=0.7, color='#404040')
            plt.plot(
                tanh_plot_data['test_dates'],
                tanh_plot_data['test_actual'],
                label="Actual Price",
                alpha=1.0,
                color='#404040',
            )
            
            # Plot TANH predictions
            plt.plot(
                tanh_plot_data['train_dates'],
                tanh_plot_data['train_preds'],
                label="Predicted Price (TANH Train Set)",
                alpha=0.9,
                color='#4169E1',  # royal blue
            )
            plt.plot(
                tanh_plot_data['test_dates'],
                tanh_plot_data['test_preds'],
                label="Predicted Price (TANH Test Set)",
                alpha=0.9,
                color='#FF7F50',  # coral
            )
            
            # Plot ELU predictions
            plt.plot(
                elu_plot_data['train_dates'],
                elu_plot_data['train_preds'],
                label="Predicted Price (ELU Train Set)",
                alpha=0.9,
                color='#CD5C5C',  # indian red
            )
            plt.plot(
                elu_plot_data['test_dates'],
                elu_plot_data['test_preds'],
                label="Predicted Price (ELU Test Set)",
                alpha=0.9,
                color='#3CB371',  # medium sea green
            )
            
            # Format plot
            plt.title(f"{profile_name.capitalize()} - Actual vs LSTM Predictions (TANH vs ELU)")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
            plt.gcf().autofmt_xdate()  # rotate and align the tick labels
            plt.tight_layout()
            plt.savefig(f"{profile_save_dir}/actual-tanh-elu.png")
            plt.close()
            
            print(f"📈 Saved combined prediction plot to: {profile_save_dir}/actual-tanh-elu.png")
            
            # Clean up
            del model_tanh, model_elu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"✅ Completed training and evaluation for profile: {profile_name}")
            
        except Exception as e:
            print(f"❌ Error in profile {profile_name}: {str(e)}")
    
    # After all profiles are trained, create comparative results if training multiple profiles
    if len(profiles_to_train) > 1:
        try:
            print("\n\n==== Creating Comparative Results Summary ====")
            
            # Collect results from all profiles
            comparative_results = {}
            
            for profile_name in profiles_to_train:
                profile_save_dir = f"{base_save_dir}/{profile_name}"
                results_file = f"{profile_save_dir}/evaluation_metrics.json"
                
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        profile_results = json.load(f)
                    
                    comparative_results[profile_name] = profile_results
            
            # Save comparative results
            with open(f"{base_save_dir}/comparative_results.json", "w") as f:
                json.dump(comparative_results, f, indent=4)
            
            # Create comparative results table
            results_df = pd.DataFrame(columns=['Profile', 'Model', 'MAE', 'RMSE', 'R2'])
            
            row = 0
            for profile, models in comparative_results.items():
                for model, metrics in models.items():
                    results_df.loc[row] = [
                        profile, model, 
                        metrics['MAE'], metrics['RMSE'], metrics['R2']
                    ]
                    row += 1
            
            # Sort by RMSE (lower is better)
            results_df = results_df.sort_values('RMSE')
            
            # Format the table for display
            plt.figure(figsize=(10, 6))
            plt.axis('off')
            table = plt.table(
                cellText=results_df.values,
                colLabels=results_df.columns,
                cellLoc='center',
                loc='center',
                colColours=["#f2f2f2"] * len(results_df.columns),
                cellColours=[
                    ["#ffffff"] * len(results_df.columns) 
                    for _ in range(len(results_df))
                ]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            plt.title("Comparative Results Across All Profiles", pad=20)
            plt.tight_layout()
            plt.savefig(f"{base_save_dir}/comparative_results_table.png", bbox_inches="tight", dpi=300)
            plt.close()
            
            print(f"✅ Saved comparative results to {base_save_dir}/comparative_results.json")
            print(f"📊 Saved comparative results table to {base_save_dir}/comparative_results_table.png")
            
        except Exception as e:
            print(f"❌ Error creating comparative results: {str(e)}")
    
    print("\nTraining completed for all profiles!")
