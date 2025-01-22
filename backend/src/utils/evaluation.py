# evaluation.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(predictions, targets):
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def print_evaluation_metrics(evaluation_metrics):
    print(f"{'Activation':<12} {'Seq Length':<12} {'Ticker':<8} {'MAE':<10} {'MSE':<10} {'RMSE':<10} {'R2':<10}")
    print("="*60)
    for (activation_name, seq_length, ticker), metrics in evaluation_metrics.items():
        mae = metrics.get('MAE', 'N/A')
        mse = metrics.get('MSE', 'N/A')
        rmse = metrics.get('RMSE', 'N/A')
        r2 = metrics.get('R2', 'N/A')
        print(f"{activation_name.capitalize():<12} {seq_length:<12} {ticker:<8} {mae:<10} {mse:<10} {rmse:<10} {r2:<10}")

