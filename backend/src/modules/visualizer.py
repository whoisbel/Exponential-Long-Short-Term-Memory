import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator
from typing import Dict, List, Tuple, Any, Optional, Union
import csv
from sklearn.preprocessing import MinMaxScaler
import torch

class Visualizer:
    """
    Class for creating and saving plots and visualizations.
    """
    
    def __init__(self):
        """
        Initialize the visualizer.
        """
        pass
    
    def plot_loss_curves(
        self,
        train_losses: Union[List[float], List[List[float]]],
        val_losses: Union[List[float], List[List[float]]],
        test_losses: Optional[Union[List[float], List[List[float]]]] = None,
        title: str = "Training and Validation Loss",
        save_path: Optional[str] = None,
        activation: Optional[str] = None
    ) -> None:
        """
        Plot and save loss curves for a single model or compare multiple models.
        
        Args:
            train_losses: Training loss values or list of training loss lists
            val_losses: Validation loss values or list of validation loss lists
            test_losses: Test loss values or list of test loss lists (optional)
            title: Plot title
            save_path: Path to save the plot image
            activation: Activation function name for single model plot
        """
        plt.figure(figsize=(12, 6))
        
        if activation:
            # Single activation function plot
            plt.semilogy(train_losses, label=f"Train Loss ({activation})", linewidth=2)
            plt.semilogy(val_losses, label=f"Val Loss ({activation})", linewidth=2)
            if test_losses is not None:
                plt.semilogy(test_losses, label=f"Test Loss ({activation})", linewidth=2, linestyle='--')
        else:
            # Combined plot with multiple activation functions
            # Make sure we have lists of losses
            if not isinstance(train_losses, list) or not isinstance(val_losses, list):
                raise ValueError("For combined plots, train_losses and val_losses must be lists of loss arrays")
            
            if len(train_losses) >= 1 and len(val_losses) >= 1:
                plt.semilogy(train_losses[0], label="TANH Train Loss", linewidth=2, color="#4169E1")  # royal blue
                plt.semilogy(val_losses[0], label="TANH Val Loss", linewidth=2, color="#FF7F50")  # coral
                
            if test_losses is not None and len(test_losses) > 0 and test_losses[0] is not None:
                plt.semilogy(test_losses[0], label="TANH Test Loss", linewidth=2, color="#FF7F50", linestyle='--')  # coral dashed
            
            if len(train_losses) >= 2 and len(val_losses) >= 2:
                plt.semilogy(train_losses[1], label="ELU Train Loss", linewidth=2, color="#CD5C5C")  # indian red
                plt.semilogy(val_losses[1], label="ELU Val Loss", linewidth=2, color="#3CB371")  # medium sea green
                
            if test_losses is not None and len(test_losses) > 1 and test_losses[1] is not None:
                plt.semilogy(test_losses[1], label="ELU Test Loss", linewidth=2, color="#3CB371", linestyle='--')  # medium sea green dashed
        
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE) - Log Scale")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"📊 Saved loss plot to: {save_path}")
            
        plt.close()
    
    def save_loss_curves_to_csv(
        self,
        train_losses: List[float],
        val_losses: List[float],
        test_losses: Optional[List[float]] = None,
        save_path: str = None,
        activation: Optional[str] = None
    ) -> None:
        """
        Save loss curves data to CSV file.
        
        Args:
            train_losses: Training loss values
            val_losses: Validation loss values
            test_losses: Test loss values (optional)
            save_path: Path to save the CSV file
            activation: Activation function name (for logging)
        """
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Determine header based on available data
            if test_losses is not None:
                writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Test_Loss'])
                for i in range(len(train_losses)):
                    test_val = test_losses[i] if test_losses is not None and i < len(test_losses) else None
                    writer.writerow([i+1, train_losses[i], val_losses[i], test_val])
            else:
                writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss'])
                for i in range(len(train_losses)):
                    writer.writerow([i+1, train_losses[i], val_losses[i]])
        
        print(f"📊 Saved {activation.upper() if activation else ''} loss data to: {save_path}")
    
    def generate_prediction_plots(
        self,
        model: Any,
        X_train: torch.Tensor,
        X_test: torch.Tensor,
        X_val: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor,
        y_val: torch.Tensor,
        df_profile: pd.DataFrame,
        target_scaler: MinMaxScaler,
        seq_len: int,
        profile_name: str,
        model_name: str,
        save_path: str,
        device: torch.device,
        colors: Optional[Dict[str, str]] = None,
        train_dates=None,
        val_dates=None,
        test_dates=None
    ) -> Dict[str, Any]:
        """
        Generate and save plots of model predictions vs actual values.
        """
        model.eval()
        
        # Get predictions
        with torch.no_grad():
            train_preds = model(X_train.to(device)).squeeze().cpu().numpy()
            test_preds = model(X_test.to(device)).squeeze().cpu().numpy()
            val_preds = model(X_val.to(device)).squeeze().cpu().numpy()
        
        # Get actual values
        train_actual = y_train.cpu().numpy()
        test_actual = y_test.cpu().numpy()
        val_actual = y_val.cpu().numpy()
        
        # Ensure shapes are correct for inverse transform
        train_preds = train_preds.reshape(-1, 1)
        test_preds = test_preds.reshape(-1, 1)
        val_preds = val_preds.reshape(-1, 1)
        train_actual = train_actual.reshape(-1, 1)
        test_actual = test_actual.reshape(-1, 1)
        val_actual = val_actual.reshape(-1, 1)
        
        # Inverse transform predictions and actual values
        train_preds = target_scaler.inverse_transform(train_preds)
        test_preds = target_scaler.inverse_transform(test_preds)
        val_preds = target_scaler.inverse_transform(val_preds)
        train_actual = target_scaler.inverse_transform(train_actual)
        test_actual = target_scaler.inverse_transform(test_actual)
        val_actual = target_scaler.inverse_transform(val_actual)
        
        # Flatten arrays
        train_preds = train_preds.flatten()
        test_preds = test_preds.flatten()
        val_preds = val_preds.flatten()
        train_actual = train_actual.flatten()
        test_actual = test_actual.flatten()
        val_actual = val_actual.flatten()
        
        # Use dates if provided, otherwise extract from dataframe
        if train_dates is not None and val_dates is not None and test_dates is not None:
            # Convert dates to pandas datetime if they aren't already
            if not isinstance(train_dates[0], pd.Timestamp):
                train_dates = pd.to_datetime(train_dates)
                val_dates = pd.to_datetime(val_dates)
                test_dates = pd.to_datetime(test_dates)
        else:
            # Get dates from the dataset
            if "Date" in df_profile.columns:
                dates = pd.to_datetime(df_profile["Date"])
                train_dates = dates[seq_len:seq_len + len(train_actual)]
                val_dates = dates[seq_len + len(train_actual):seq_len + len(train_actual) + len(val_actual)]
                test_dates = dates[seq_len + len(train_actual) + len(val_actual):seq_len + len(train_actual) + len(val_actual) + len(test_actual)]
            else:
                # Create date range if no Date column exists
                end_date = pd.Timestamp.today()
                dates = pd.date_range(end=end_date, periods=len(df_profile), freq='D')
                train_dates = dates[seq_len:seq_len + len(train_actual)]
                val_dates = dates[seq_len + len(train_actual):seq_len + len(train_actual) + len(val_actual)]
                test_dates = dates[seq_len + len(train_actual) + len(val_actual):seq_len + len(train_actual) + len(val_actual) + len(test_actual)]
        
        # Default colors if not provided
        if colors is None:
            colors = {
                'train_actual': '#404040',
                'val_actual': '#404040',
                'test_actual': '#404040',
                'train_pred': '#4169E1',  # royal blue
                'val_pred': '#d618c9',    # purple
                'test_pred': '#FF7F50'    # coral
            }
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot actual values as a single continuous line
        all_dates = pd.concat([pd.Series(train_dates), pd.Series(val_dates), pd.Series(test_dates)])
        all_actual = np.concatenate([train_actual, val_actual, test_actual])
        plt.plot(all_dates, all_actual, label="Actual Price", color=colors['train_actual'], alpha=0.7)
        
        # Plot predictions
        plt.plot(
            train_dates,
            train_preds,
            label=f"Predicted Price ({model_name} Train Set)",
            alpha=0.9,
            color=colors['train_pred'],
        )
        plt.plot(
            val_dates,
            val_preds,
            label=f"Predicted Price ({model_name} Validation Set)",
            alpha=0.9,
            color=colors['val_pred'],
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
            'val_dates': val_dates,
            'test_dates': test_dates,
            'train_actual': train_actual,
            'val_actual': val_actual,
            'test_actual': test_actual,
            'train_preds': train_preds,
            'val_preds': val_preds,
            'test_preds': test_preds
        }
    
    def generate_combined_prediction_plot(
        self,
        tanh_plot_data: Dict[str, Any],
        elu_plot_data: Dict[str, Any],
        save_path: str
    ) -> None:
        """
        Generate a combined plot comparing predictions from both models.
        
        Args:
            tanh_plot_data: Plot data from TANH model
            elu_plot_data: Plot data from ELU model
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Convert numpy arrays to pandas Series if needed
        train_dates = pd.Series(tanh_plot_data['train_dates'])
        val_dates = pd.Series(tanh_plot_data['val_dates'])
        test_dates = pd.Series(tanh_plot_data['test_dates'])
        
        # Plot actual values as a single continuous line
        all_dates = pd.concat([train_dates, val_dates, test_dates])
        all_actual = np.concatenate([
            tanh_plot_data['train_actual'],
            tanh_plot_data['val_actual'],
            tanh_plot_data['test_actual']
        ])
        plt.plot(all_dates, all_actual, label="Actual Price", color='#404040', alpha=0.7)
        
        # Plot TANH predictions
        plt.plot(
            tanh_plot_data['train_dates'],
            tanh_plot_data['train_preds'],
            label="Predicted Price (TANH Train Set)",
            alpha=0.9,
            color='#4169E1',  # royal blue
        )
        plt.plot(
            tanh_plot_data['val_dates'],
            tanh_plot_data['val_preds'],
            label="Predicted Price (TANH Validation Set)",
            alpha=0.9,
            color='#d618c9',  # purple
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
            elu_plot_data['val_dates'],
            elu_plot_data['val_preds'],
            label="Predicted Price (ELU Validation Set)",
            alpha=0.9,
            color='#98FB98',  # pale green
        )
        plt.plot(
            elu_plot_data['test_dates'],
            elu_plot_data['test_preds'],
            label="Predicted Price (ELU Test Set)",
            alpha=0.9,
            color='#3CB371',  # medium sea green
        )
        
        # Format plot
        plt.title("Baseline LSTM and Enhanced LSTM-ELU Predictions")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.gcf().autofmt_xdate()  # rotate and align the tick labels
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"📈 Saved combined prediction plot to: {save_path}")
    
    def save_all_plots(
        self,
        tanh_train_losses: List[float],
        tanh_val_losses: List[float],
        tanh_test_losses: List[float],
        elu_train_losses: List[float],
        elu_val_losses: List[float],
        elu_test_losses: List[float],
        tanh_plot_data: Dict[str, Any],
        elu_plot_data: Dict[str, Any],
        save_dir: str
    ) -> None:
        """
        Save all plots and CSV files for model training and evaluation.
        
        Args:
            tanh_train_losses: Training losses for TANH model
            tanh_val_losses: Validation losses for TANH model
            tanh_test_losses: Test losses for TANH model
            elu_train_losses: Training losses for ELU model
            elu_val_losses: Validation losses for ELU model
            elu_test_losses: Test losses for ELU model
            tanh_plot_data: Plot data for TANH model
            elu_plot_data: Plot data for ELU model
            save_dir: Base directory to save plots
        """
        # Create directory structure
        plots_loss_dir = os.path.join(save_dir, "plots/loss")
        plots_pred_dir = os.path.join(save_dir, "plots/predictions")
        loss_data_dir = os.path.join(save_dir, "loss_data")
        
        os.makedirs(plots_loss_dir, exist_ok=True)
        os.makedirs(plots_pred_dir, exist_ok=True)
        os.makedirs(loss_data_dir, exist_ok=True)
        
        # Save TANH loss plot and data
        self.plot_loss_curves(
            tanh_train_losses,
            tanh_val_losses,
            tanh_test_losses,
            "TANH Loss Curves",
            f"{plots_loss_dir}/tanh_loss_curves.png",
            "TANH"
        )
        self.save_loss_curves_to_csv(
            tanh_train_losses,
            tanh_val_losses,
            tanh_test_losses,
            f"{loss_data_dir}/tanh_loss_data.csv",
            "TANH"
        )
        
        # Save ELU loss plot and data
        self.plot_loss_curves(
            elu_train_losses,
            elu_val_losses,
            elu_test_losses,
            "ELU Loss Curves",
            f"{plots_loss_dir}/elu_loss_curves.png",
            "ELU"
        )
        self.save_loss_curves_to_csv(
            elu_train_losses,
            elu_val_losses,
            elu_test_losses,
            f"{loss_data_dir}/elu_loss_data.csv",
            "ELU"
        )
        
        # Save combined loss plot
        self.plot_loss_curves(
            [tanh_train_losses, elu_train_losses],
            [tanh_val_losses, elu_val_losses],
            [tanh_test_losses, elu_test_losses],
            "Loss Curves Comparison",
            f"{plots_loss_dir}/combined_loss_curves.png"
        )
        
        # Save combined prediction plot
        self.generate_combined_prediction_plot(
            tanh_plot_data,
            elu_plot_data,
            f"{plots_pred_dir}/combined_predictions.png"
        )
        
        # Save prediction data to CSV
        self.save_predictions_to_csv(
            tanh_plot_data,
            elu_plot_data,
            save_dir
        )
    
    def save_predictions_to_csv(
        self,
        tanh_plot_data: Dict[str, Any],
        elu_plot_data: Dict[str, Any],
        save_dir: str
    ) -> None:
        """
        Save prediction data and actual values to CSV files for further analysis.
        
        Args:
            tanh_plot_data: Plot data from TANH model
            elu_plot_data: Plot data from ELU model
            save_dir: Directory to save the CSV files
        """
        # Create data directory if it doesn't exist
        data_dir = os.path.join(save_dir, "predictions_data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Process train data
        train_data = pd.DataFrame({
            'Date': tanh_plot_data['train_dates'],
            'Set': 'Train',
            'Actual_Price': tanh_plot_data['train_actual'],
            'TANH_Prediction': tanh_plot_data['train_preds'],
            'ELU_Prediction': elu_plot_data['train_preds']
        })
        
        # Process validation data
        val_data = pd.DataFrame({
            'Date': tanh_plot_data['val_dates'],
            'Set': 'Validation',
            'Actual_Price': tanh_plot_data['val_actual'],
            'TANH_Prediction': tanh_plot_data['val_preds'],
            'ELU_Prediction': elu_plot_data['val_preds']
        })
        
        # Process test data
        test_data = pd.DataFrame({
            'Date': tanh_plot_data['test_dates'],
            'Set': 'Test',
            'Actual_Price': tanh_plot_data['test_actual'],
            'TANH_Prediction': tanh_plot_data['test_preds'],
            'ELU_Prediction': elu_plot_data['test_preds']
        })
        
        # Combine all data
        all_data = pd.concat([train_data, val_data, test_data]).reset_index(drop=True)
        
        # Save to CSV
        all_data.to_csv(f"{data_dir}/all_predictions.csv", index=False)
        train_data.to_csv(f"{data_dir}/train_predictions.csv", index=False)
        val_data.to_csv(f"{data_dir}/validation_predictions.csv", index=False)
        test_data.to_csv(f"{data_dir}/test_predictions.csv", index=False)
        
        print(f"📊 Saved prediction data to: {data_dir}/all_predictions.csv")
        
        # Calculate error metrics for each set
        def calculate_metrics(actual, tanh_pred, elu_pred):
            tanh_mse = ((actual - tanh_pred) ** 2).mean()
            tanh_mae = np.abs(actual - tanh_pred).mean()
            tanh_rmse = np.sqrt(tanh_mse)
            
            elu_mse = ((actual - elu_pred) ** 2).mean()
            elu_mae = np.abs(actual - elu_pred).mean()
            elu_rmse = np.sqrt(elu_mse)
            
            return {
                'TANH_MSE': tanh_mse,
                'TANH_MAE': tanh_mae,
                'TANH_RMSE': tanh_rmse,
                'ELU_MSE': elu_mse,
                'ELU_MAE': elu_mae,
                'ELU_RMSE': elu_rmse
            }
        
        # Calculate metrics for each set
        train_metrics = calculate_metrics(
            train_data['Actual_Price'], 
            train_data['TANH_Prediction'], 
            train_data['ELU_Prediction']
        )
        val_metrics = calculate_metrics(
            val_data['Actual_Price'], 
            val_data['TANH_Prediction'], 
            val_data['ELU_Prediction']
        )
        test_metrics = calculate_metrics(
            test_data['Actual_Price'], 
            test_data['TANH_Prediction'], 
            test_data['ELU_Prediction']
        )
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            'Set': ['Train', 'Validation', 'Test'],
            'TANH_MSE': [train_metrics['TANH_MSE'], val_metrics['TANH_MSE'], test_metrics['TANH_MSE']],
            'TANH_MAE': [train_metrics['TANH_MAE'], val_metrics['TANH_MAE'], test_metrics['TANH_MAE']],
            'TANH_RMSE': [train_metrics['TANH_RMSE'], val_metrics['TANH_RMSE'], test_metrics['TANH_RMSE']],
            'ELU_MSE': [train_metrics['ELU_MSE'], val_metrics['ELU_MSE'], test_metrics['ELU_MSE']],
            'ELU_MAE': [train_metrics['ELU_MAE'], val_metrics['ELU_MAE'], test_metrics['ELU_MAE']],
            'ELU_RMSE': [train_metrics['ELU_RMSE'], val_metrics['ELU_RMSE'], test_metrics['ELU_RMSE']]
        })
        
        # Save metrics to CSV
        metrics_df.to_csv(f"{data_dir}/prediction_metrics.csv", index=False)
        print(f"📊 Saved prediction metrics to: {data_dir}/prediction_metrics.csv") 