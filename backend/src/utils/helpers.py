"""
Utility functions for the stock prediction application
"""

import pandas as pd
import numpy as np
import torch
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


def setup_device(device_type: str = "cuda_if_available") -> torch.device:
    """
    Setup PyTorch device based on availability and preference

    Args:
        device_type: "cuda", "cpu", or "cuda_if_available"

    Returns:
        PyTorch device object
    """
    if device_type == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    elif device_type == "cpu":
        return torch.device("cpu")
    else:  # cuda_if_available
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that DataFrame has required columns

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        return False

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False

    return True


def handle_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle MultiIndex columns from Yahoo Finance data

    Args:
        df: DataFrame with potentially MultiIndex columns

    Returns:
        DataFrame with single-level columns
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)  # Remove ticker level, keep OHLCV level
    return df


def compute_technical_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute technical features for the model

    Args:
        df: DataFrame with OHLCV data

    Returns:
        NumPy array with computed features
    """
    df_copy = df.copy()
    df_copy["return_1d"] = (
        df_copy["Close"].pct_change().replace([np.inf, -np.inf], np.nan)
    )
    df_copy["return_5d"] = (
        df_copy["Close"].pct_change(5).replace([np.inf, -np.inf], np.nan)
    )
    df_copy.dropna(inplace=True)
    return df_copy[["Close", "return_1d", "return_5d"]].values


def prepare_sequences(
    data: np.ndarray, seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM input

    Args:
        data: Scaled data array
        seq_length: Length of input sequences

    Returns:
        Tuple of (X, y) arrays for training/prediction
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i])
        y.append(data[i, 0])  # Predict Close price
    return np.array(X), np.array(y)


def inverse_transform_predictions(
    predictions: np.ndarray, scaler: MinMaxScaler, feature_idx: int = 0
) -> np.ndarray:
    """
    Inverse transform predictions back to original scale

    Args:
        predictions: Scaled predictions
        scaler: Fitted MinMaxScaler
        feature_idx: Index of the feature to inverse transform (0 for Close price)

    Returns:
        Inverse transformed predictions
    """
    # Create dummy array with same number of features as scaler was fitted on
    dummy_array = np.zeros((len(predictions), scaler.n_features_in_))
    dummy_array[:, feature_idx] = predictions

    # Inverse transform and return only the desired feature
    return scaler.inverse_transform(dummy_array)[:, feature_idx]
