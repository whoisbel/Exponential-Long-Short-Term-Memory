import ta
import yfinance as yf
import os
import pandas as pd
import numpy as np
def preprocess_data(file_names):
    data = {}
    scalers = {}
    for file in file_names:
        ticker = file.split('/')[-1].replace('.csv', '')
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        
        df.dropna(inplace=True)

        feature_columns = ['Open', 'Close', 'Volume']
        target_column = 'Close'
        
        scaler_features = MinMaxScaler()
        scaler_target = MinMaxScaler()

        scaled_features = scaler_features.fit_transform(df[feature_columns])
        scaled_target = scaler_target.fit_transform(df[[target_column]])

        data[ticker] = {
            'scaled_features': scaled_features,
            'scaled_target': scaled_target
        }
        scalers[ticker] = {
            'feature_scaler': scaler_features,
            'target_scaler': scaler_target
        }

    return data, scalers


def scale_back_predictions(predictions, scalers, ticker):
    scaler_target = scalers[ticker]['target_scaler']
    return scaler_target.inverse_transform(predictions)


def create_sequences(features, target, seq_length):
    """
    Creates sequences for training/testing data.
    
    :param features: Scaled feature data.
    :param target: Scaled target data.
    :param seq_length: Length of each sequence.
    :return: Arrays of sequences for features and targets.
    """
    X, y = [], []
    for i in range(len(features) - seq_length):
        end_ix = i + seq_length
        if end_ix > len(features) - 1:
            break
        seq_x, seq_y = features[i:end_ix], target[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)