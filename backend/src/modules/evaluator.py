import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Any, Optional, Union
import json

class ModelEvaluator:
    """
    Class to handle model evaluation and results reporting.
    """
    
    def __init__(self, device: torch.device):
        """
        Initialize the model evaluator.
        
        Args:
            device: PyTorch device to use for inference
        """
        self.device = device
        
    def evaluate_model(
        self, 
        model: nn.Module, 
        X_data: torch.Tensor, 
        y_true_scaled: torch.Tensor, 
        target_scaler: MinMaxScaler, 
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Evaluate model performance using different metrics.
        
        Args:
            model: Trained model to evaluate
            X_data: Input features tensor
            y_true_scaled: True target values (scaled)
            target_scaler: Scaler used for the target variable
            model_name: Name to identify the model in output messages
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        with torch.no_grad():
            preds = model(X_data.to(self.device)).squeeze().cpu().numpy()
        
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
    
    def create_evaluation_table(
        self, 
        results_dict: Dict[str, Dict[str, float]], 
        save_path: str
    ) -> None:
        """
        Create and save a table visualization of evaluation metrics.
        
        Args:
            results_dict: Dictionary with model evaluation results
            save_path: Path to save the table image
        """
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
    
    def save_evaluation_results(
        self, 
        results_dict: Dict[str, Dict[str, float]], 
        save_dir: str
    ) -> None:
        """
        Save evaluation results to JSON and create metrics table.
        
        Args:
            results_dict: Dictionary with model evaluation results
            save_dir: Directory to save results
        """
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join(save_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save metrics as JSON
        json_path = os.path.join(metrics_dir, "evaluation_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f"📊 Saved evaluation metrics to: {json_path}")
        
        # Create and save metrics table
        table_path = os.path.join(metrics_dir, "evaluation_metrics_table.png")
        self.create_evaluation_table(results_dict, table_path)
    
    def _convert_to_serializable(self, obj):
        """
        Convert numpy/torch types to Python native types for JSON serialization.
        """
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        return obj

    def create_results_json(
        self,
        config_dict: Dict[str, Any],
        tanh_results: Dict[str, float],
        elu_results: Dict[str, float],
        tanh_train_losses: List[float],
        tanh_val_losses: List[float],
        tanh_test_losses: List[float],
        elu_train_losses: List[float],
        elu_val_losses: List[float],
        elu_test_losses: List[float],
        best_tanh_epoch: int,
        best_elu_epoch: int,
        save_path: str
    ) -> Dict[str, Any]:
        """
        Create a comprehensive results JSON with all model evaluation details.
        
        Args:
            config_dict: Dictionary with configuration parameters
            tanh_results: Evaluation metrics for TANH model
            elu_results: Evaluation metrics for ELU model
            tanh_train_losses: Training losses for TANH model
            tanh_val_losses: Validation losses for TANH model
            tanh_test_losses: Test losses for TANH model
            elu_train_losses: Training losses for ELU model
            elu_val_losses: Validation losses for ELU model
            elu_test_losses: Test losses for ELU model
            best_tanh_epoch: Best epoch for TANH model
            best_elu_epoch: Best epoch for ELU model
            save_path: Path to save the JSON file
            
        Returns:
            The complete results dictionary
        """
        # Find the best epochs
        min_tanh_val_loss_idx = tanh_val_losses.index(min(tanh_val_losses))
        min_elu_val_loss_idx = elu_val_losses.index(min(elu_val_losses))
        
        # Get test loss at best epoch, if available
        tanh_test_loss_at_best = None
        if tanh_test_losses is not None and min_tanh_val_loss_idx < len(tanh_test_losses):
            tanh_test_loss_at_best = tanh_test_losses[min_tanh_val_loss_idx]
        
        elu_test_loss_at_best = None
        if elu_test_losses is not None and min_elu_val_loss_idx < len(elu_test_losses):
            elu_test_loss_at_best = elu_test_losses[min_elu_val_loss_idx]
        
        # Create results dictionary
        results = {
            "metadata": config_dict.get("metadata", {}),
            "dataset": config_dict.get("dataset", {}),
            "models": {
                "TANH": {
                    "architecture": {k: v for k, v in config_dict.get("model_architecture", {}).items() 
                                   if k != "total_params_elu" and k != "activation_functions"},
                    "total_params": config_dict.get("model_architecture", {}).get("total_params_tanh"),
                    "activation": "tanh",
                    "best_epoch": best_tanh_epoch,
                    "losses_at_best_epoch": {
                        "train_loss": tanh_train_losses[min_tanh_val_loss_idx],
                        "val_loss": tanh_val_losses[min_tanh_val_loss_idx],
                        "test_loss": tanh_test_loss_at_best
                    },
                    "final_metrics": tanh_results
                },
                "ELU": {
                    "architecture": {k: v for k, v in config_dict.get("model_architecture", {}).items() 
                                   if k != "total_params_tanh" and k != "activation_functions"},
                    "total_params": config_dict.get("model_architecture", {}).get("total_params_elu"),
                    "activation": "elu",
                    "best_epoch": best_elu_epoch,
                    "losses_at_best_epoch": {
                        "train_loss": elu_train_losses[min_elu_val_loss_idx],
                        "val_loss": elu_val_losses[min_elu_val_loss_idx],
                        "test_loss": elu_test_loss_at_best
                    },
                    "final_metrics": elu_results
                }
            },
            "training_parameters": config_dict.get("training_parameters", {}),
            "system": config_dict.get("system", {})
        }
        
        # Convert all values to JSON serializable format
        results = self._convert_to_serializable(results)
        
        # Save results to JSON file
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"📝 Saved comprehensive results to: {save_path}")
        
        return results 