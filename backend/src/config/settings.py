"""
Configuration settings for the FastAPI application
"""

from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "saved_models" / "BEST_PERFORMANCE"
DATASETS_DIR = BASE_DIR / "datasets"

# Model paths
CONFIG_PATH = MODELS_DIR / "configs" / "final_config.json"
ELU_MODEL_PATH = MODELS_DIR / "models" / "model_elu.pth"
TANH_MODEL_PATH = MODELS_DIR / "models" / "model_tanh.pth"

# API settings
API_TITLE = "Stock Price Prediction API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "LSTM-based stock price prediction service"

# CORS settings
CORS_ORIGINS = ["*"]  # Adjust this to restrict origins if needed
CORS_CREDENTIALS = True
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# Model settings
DEVICE_TYPE = "cuda_if_available"  # "cuda", "cpu", or "cuda_if_available"
DEFAULT_TICKER = "AI.PA"
DEFAULT_TIMEZONE = "Asia/Manila"

# Prediction settings
DEFAULT_PRED_DAYS = 3
BACKTEST_DAYS = 38  # Number of days to backtest for validation
PREDICTION_CACHE_TTL = 300  # 5 minutes in seconds
