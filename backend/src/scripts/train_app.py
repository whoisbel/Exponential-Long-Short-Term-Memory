import os
import sys
import argparse
import torch
import pandas as pd
from typing import Dict, List, Any, Optional

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from modules.custom_lstm import LSTMModel
from modules.data_processor import DataProcessor
from modules.trainer import ModelTrainer
from modules.evaluator import ModelEvaluator 
from modules.visualizer import Visualizer
from modules.config_manager import ConfigManager, setup_directories, save_config, print_summary

def main():
    """Main function to orchestrate the LSTM model training and evaluation process."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LSTM models for stock price prediction')
    parser.add_argument('--all-configs', action='store_true', help='Train using all configurations from config.json')
    parser.add_argument('--config', type=str, help='Specific configuration to use from config.json')
    parser.add_argument('--dataset', type=str, default='datasets/air_liquide.csv', help='Path to the dataset CSV file')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize config manager
    config_manager = ConfigManager("src/scripts/config.json")
    config_manager.load_configs()
    
    # Determine which configs to use
    if args.all_configs:
        configs_to_train = config_manager.config_names
    elif args.config:
        requested_configs = [args.config]
        configs_to_train = config_manager.validate_config_names(requested_configs)
    else:
        configs_to_train = config_manager.validate_config_names()
    
    # Create base save directory
    base_save_dir = "saved_models/"
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Loop through each configuration
    for config_name in configs_to_train:
        print(f"\n{'='*50}")
        print(f"Training configuration: {config_name}")
        print(f"{'='*50}\n")
        
        # Get configuration
        config = config_manager.get_config(config_name)
        
        # Setup directories
        save_dirs = setup_directories(base_save_dir, config_name)
        
        # Create configuration dictionary
        config_dict = config_manager.create_config_dict(config_name, device)
        
        # Save initial config
        save_config(config_dict, f"{save_dirs['configs']}/initial_config.json")
        
        # Initialize data processor
        data_processor = DataProcessor(config)
        
        # Load and process data
        data_processor.load_data(args.dataset)
        data_processor.add_technical_indicators()
        
        # Save processed data
        data_processor.save_processed_data(save_dirs["base"])
        
        # Prepare sequences
        X, y_scaled, dataset_info = data_processor.prepare_sequences()
        
        # Update config with dataset info
        config_dict["dataset"].update(dataset_info)
        
        # Create data loaders
        train_loader, test_loader, val_loader = data_processor.create_data_loaders(X, y_scaled)
        
        # Get feature and target scalers
        _, target_scaler = data_processor.get_scalers()
        
        # Initialize models
        input_size = dataset_info['feature_columns'].__len__()
        hidden_sizes = config.get('hidden_sizes', [50])
        dropout = config.get('dropout', 0.3)
        
        # Create models
        model_tanh = LSTMModel(
            input_size=input_size, 
            hidden_size=hidden_sizes, 
            dropout=dropout,
            activation_fn="tanh"
        ).to(device)
        
        model_elu = LSTMModel(
            input_size=input_size, 
            hidden_size=hidden_sizes, 
            dropout=dropout,
            activation_fn="elu"
        ).to(device)
        
        # Update config with model parameters
        total_params_tanh = sum(p.numel() for p in model_tanh.parameters())
        total_params_elu = sum(p.numel() for p in model_elu.parameters())
        config_dict["model_architecture"]["total_params_tanh"] = total_params_tanh
        config_dict["model_architecture"]["total_params_elu"] = total_params_elu
        
        # Initialize trainer
        trainer = ModelTrainer(config, device)
        
        # Train TANH model
        print("\nTraining TANH model...")
        model_tanh, tanh_train_losses, tanh_val_losses, tanh_test_losses, best_tanh_epoch = trainer.train_model(
            model_tanh, train_loader, test_loader, "tanh", val_loader
        )
        trainer.save_model(model_tanh, "tanh", save_dirs["base"])
        
        # Train ELU model
        print("\nTraining ELU model...")
        model_elu, elu_train_losses, elu_val_losses, elu_test_losses, best_elu_epoch = trainer.train_model(
            model_elu, train_loader, test_loader, "elu", val_loader
        )
        trainer.save_model(model_elu, "elu", save_dirs["base"])
        
        # Extract X_train, X_test, y_train, y_test tensors from loaders for evaluation
        # Instead of using potentially shuffled data from loaders, use the original data in correct order
        data_processor.df_processed['Date'] = pd.to_datetime(data_processor.df_processed['Date'])
        sequence_data = data_processor.get_sequence_data_for_visualization()
        X_all = sequence_data['X']
        y_all = sequence_data['y']
        dates = sequence_data['dates']
        
        # Get original split sizes and points
        train_size = int(config.get('train_size', 0.7) * len(X_all))
        val_size = int(config.get('validation_size', 0.15) * len(X_all))
        
        # Split the data in the correct time order
        X_train = torch.tensor(X_all[:train_size], dtype=torch.float32)
        y_train = torch.tensor(y_all[:train_size], dtype=torch.float32).reshape(-1, 1)
        
        X_val = torch.tensor(X_all[train_size:train_size+val_size], dtype=torch.float32)
        y_val = torch.tensor(y_all[train_size:train_size+val_size], dtype=torch.float32).reshape(-1, 1)
        
        X_test = torch.tensor(X_all[train_size+val_size:], dtype=torch.float32)
        y_test = torch.tensor(y_all[train_size+val_size:], dtype=torch.float32).reshape(-1, 1)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(device)
        
        # Evaluate models
        print("\nEvaluating models...")
        tanh_results = evaluator.evaluate_model(
            model_tanh, X_test, y_test, target_scaler, f"TANH Model ({config_name})"
        )
        
        elu_results = evaluator.evaluate_model(
            model_elu, X_test, y_test, target_scaler, f"ELU Model ({config_name})"
        )
        
        # Create results dictionary
        results_dict = {
            "TANH Model": {k: float(v) for k, v in tanh_results.items()},
            "ELU Model": {k: float(v) for k, v in elu_results.items()}
        }
        
        # Save evaluation results
        evaluator.save_evaluation_results(results_dict, save_dirs["base"])
        
        # Create comprehensive results JSON
        results_json = evaluator.create_results_json(
            config_dict,
            tanh_results,
            elu_results,
            tanh_train_losses,
            tanh_val_losses,
            tanh_test_losses,
            elu_train_losses,
            elu_val_losses,
            elu_test_losses,
            best_tanh_epoch,
            best_elu_epoch,
            f"{save_dirs['metrics']}/results.json"
        )
        
        # Initialize visualizer
        visualizer = Visualizer()
        
        # Get the dates for train, validation, and test sets
        train_dates = dates[:train_size]
        val_dates = dates[train_size:train_size+val_size]
        test_dates = dates[train_size+val_size:]
        
        # Generate prediction plots
        print("\nGenerating prediction plots...")
        tanh_plot_data = visualizer.generate_prediction_plots(
            model_tanh, X_train, X_test, X_val, y_train, y_test, y_val,
            data_processor.df_processed, target_scaler, dataset_info['sequence_length'], "Model", 
            "TANH", f"{save_dirs['plots_pred']}/tanh_predictions.png",
            device,
            colors={
                'train_actual': '#404040',
                'val_actual': '#404040',
                'test_actual': '#404040',
                'train_pred': '#4169E1',  # royal blue
                'val_pred': '#32CD32',    # lime green
                'test_pred': '#FF7F50'    # coral
            },
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=test_dates
        )
        
        elu_plot_data = visualizer.generate_prediction_plots(
            model_elu, X_train, X_test, X_val, y_train, y_test, y_val,
            data_processor.df_processed, target_scaler, dataset_info['sequence_length'], "Model", 
            "ELU", f"{save_dirs['plots_pred']}/elu_predictions.png",
            device,
            colors={
                'train_actual': '#404040',
                'val_actual': '#404040',
                'test_actual': '#404040',
                'train_pred': '#CD5C5C',  # indian red
                'val_pred': '#32CD32',    # lime green
                'test_pred': '#3CB371'    # medium sea green
            },
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=test_dates
        )
        
        # Save all visualizations
        visualizer.save_all_plots(
            tanh_train_losses,
            tanh_val_losses,
            tanh_test_losses,
            elu_train_losses,
            elu_val_losses,
            elu_test_losses,
            tanh_plot_data,
            elu_plot_data,
            save_dirs["base"]
        )
        
        # Update config with training details
        config_dict["training_details"] = {
            "tanh_model": {
                "best_epoch": best_tanh_epoch,
                "total_epochs_trained": len(tanh_train_losses),
                "best_val_loss": min(tanh_val_losses),
                "final_train_loss": tanh_train_losses[-1],
                "final_val_loss": tanh_val_losses[-1],
                "final_test_loss": tanh_test_losses[-1] if tanh_test_losses and len(tanh_test_losses) > 0 else None
            },
            "elu_model": {
                "best_epoch": best_elu_epoch,
                "total_epochs_trained": len(elu_train_losses),
                "best_val_loss": min(elu_val_losses),
                "final_train_loss": elu_train_losses[-1],
                "final_val_loss": elu_val_losses[-1],
                "final_test_loss": elu_test_losses[-1] if elu_test_losses and len(elu_test_losses) > 0 else None
            }
        }
        
        # Save final config with results
        config_dict["evaluation_results"] = results_dict
        save_config(config_dict, f"{save_dirs['configs']}/final_config.json")
        
        # Clean up
        del model_tanh, model_elu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print summary
        print_summary(save_dirs)
        
        print(f"\n✅ Completed training and evaluation for {config_name}")
    
    print("\n🎉 All configurations completed successfully!")


if __name__ == "__main__":
    main() 