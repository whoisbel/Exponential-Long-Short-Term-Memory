import os
import json
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
import datetime
import shutil

class ConfigManager:
    """
    Class to handle configuration loading, validation, and directory setup.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.configs = None
        self.config_names = None
        self.selected_configs = None
        
    def load_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load configurations from the config file.
        
        Returns:
            Dictionary with configuration parameters
        """
        try:
            with open(self.config_path, 'r') as f:
                self.configs = json.load(f)
                self.config_names = list(self.configs.keys())
                return self.configs
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Config file not found at {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"❌ Invalid JSON format in config file: {self.config_path}")
    
    def validate_config_names(self, requested_configs: Optional[List[str]] = None) -> List[str]:
        """
        Validate requested config names and determine which to use.
        
        Args:
            requested_configs: List of configuration names to use
            
        Returns:
            List of validated configuration names
        """
        if self.configs is None:
            self.load_configs()
            
        if requested_configs:
            # Filter valid configs
            valid_configs = [cfg for cfg in requested_configs if cfg in self.config_names]
            if not valid_configs:
                print(f"⚠️ None of the requested configs were found. Using default: {self.config_names[0]}")
                valid_configs = [self.config_names[0]]
        else:
            # Use first config as default
            valid_configs = [self.config_names[0]]
            print(f"ℹ️ No configuration specified. Using default: {valid_configs[0]}")
        
        self.selected_configs = valid_configs
        return valid_configs
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get a specific configuration by name.
        
        Args:
            config_name: Name of the configuration to retrieve
            
        Returns:
            Configuration dictionary
        """
        if self.configs is None:
            self.load_configs()
            
        if config_name not in self.configs:
            raise ValueError(f"❌ Configuration '{config_name}' not found in config file")
            
        return self.configs[config_name]
    
    def create_config_dict(self, config_name: str, device: torch.device, model_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a comprehensive configuration dictionary with metadata.
        
        Args:
            config_name: Name of the configuration
            device: PyTorch device
            model_params: Additional model parameters to include
            
        Returns:
            Complete configuration dictionary
        """
        config = self.get_config(config_name)
        
        # Current timestamp for directory naming
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build comprehensive config dict
        config_dict = {
            "metadata": {
                "config_name": config_name,
                "timestamp": timestamp,
                "description": "LSTM model for stock price prediction"
            },
            "dataset": {
                "name": "air_liquide.csv",
                "sequence_length": config.get('seq_len', 60)
            },
            "model_architecture": {
                "model_type": "LSTM",
                "input_size": 3,  # Close, Return_1D, Return_5D
                "hidden_sizes": config.get('hidden_sizes', [50]),
                "dropout": config.get('dropout', 0.3),
                "activation_functions": ["tanh", "elu"]
            },
            "training_parameters": {
                "batch_size": config.get('batch_size', 32),
                "epochs": config.get('epochs', 100),
                "learning_rate": config.get('learning_rate', 0.001),
                "train_size": config.get('train_size', 0.7),
                "test_size": config.get('test_size', 0.15),
                "validation_size": config.get('validation_size', 0.15),
                "patience": config.get('patience', 10)
            },
            "system": {
                "device": str(device),
                "torch_version": torch.__version__,
                "python_version": torch.__version__  # Should be replaced with actual Python version
            }
        }
        
        # Add model params if provided
        if model_params:
            for key, value in model_params.items():
                config_dict["model_architecture"][key] = value
                
        return config_dict


def setup_directories(base_dir: str, config_name: str, timestamp: str = None) -> Dict[str, str]:
    """
    Create directory structure for model training outputs.
    
    Args:
        base_dir: Base directory path
        config_name: Name of the configuration
        timestamp: Optional timestamp to use in directory name
        
    Returns:
        Dictionary with paths to all created directories
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory structure
    config_save_dir = os.path.join(base_dir, f"{timestamp}_{config_name}")
    dirs = {
        "base": config_save_dir,
        "models": os.path.join(config_save_dir, "models"),
        "configs": os.path.join(config_save_dir, "configs"),
        "metrics": os.path.join(config_save_dir, "metrics"),
        "loss_data": os.path.join(config_save_dir, "loss_data"),
        "plots_loss": os.path.join(config_save_dir, "plots/loss"),
        "plots_pred": os.path.join(config_save_dir, "plots/predictions"),
        "data": os.path.join(config_save_dir, "data")
    }
    
    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"📁 Created directory structure at: {config_save_dir}")
    
    return dirs


def save_config(config_dict: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration dictionary to a JSON file.
    
    Args:
        config_dict: Configuration dictionary to save
        save_path: Path to save the configuration file
    """
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✅ Saved configuration to: {save_path}")


def print_summary(save_dirs: Dict[str, str]) -> None:
    """
    Print a summary of all saved files for reference.
    
    Args:
        save_dirs: Dictionary with directories for the saved files
    """
    print("\n📁 Files saved for this configuration:")
    print(f"  - Main directory: {save_dirs['base']}")
    print(f"  - Model files: {save_dirs['models']}/model_tanh.pth, {save_dirs['models']}/model_elu.pth")
    print(f"  - Config files: {save_dirs['configs']}/initial_config.json, {save_dirs['configs']}/final_config.json")
    print(f"  - Results file: {save_dirs['metrics']}/results.json")
    print(f"  - Evaluation metrics: {save_dirs['metrics']}/evaluation_metrics.json, {save_dirs['metrics']}/evaluation_metrics_table.png")
    print(f"  - Loss data: {save_dirs['loss_data']}/tanh_loss_data.csv, {save_dirs['loss_data']}/elu_loss_data.csv")
    print(f"  - Loss plots: {save_dirs['plots_loss']}/tanh_loss_curves.png, {save_dirs['plots_loss']}/elu_loss_curves.png, {save_dirs['plots_loss']}/combined_loss_curves.png")
    print(f"  - Prediction plots: {save_dirs['plots_pred']}/tanh_predictions.png, {save_dirs['plots_pred']}/elu_predictions.png, {save_dirs['plots_pred']}/combined_predictions.png")
    print(f"  - Dataset files: {save_dirs['data']}/processed_data.csv, {save_dirs['data']}/data_comparison.csv, {save_dirs['data']}/sequence_samples.csv")
    print(f"  - Prediction data: {save_dirs['base']}/predictions_data/all_predictions.csv, {save_dirs['base']}/predictions_data/prediction_metrics.csv") 