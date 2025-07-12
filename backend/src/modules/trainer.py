import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.preprocessing import MinMaxScaler

class ModelTrainer:
    """
    Class to handle training of PyTorch models with early stopping and validation.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize the model trainer.
        
        Args:
            config: Dictionary with training configuration
            device: PyTorch device to use for training
        """
        self.config = config
        self.device = device
        self.criterion = nn.MSELoss()
        self.learning_rate = config.get('learning_rate', 0.001)
        self.patience = config.get('patience', 10)
        self.epochs = config.get('epochs', 100)
        
    def train_model(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        activation_fn: str,
        validation_loader: Optional[DataLoader] = None
    ) -> Tuple[nn.Module, List[float], List[float], List[float], int]:
        """
        Train a model with early stopping.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            activation_fn: Name of activation function used
            validation_loader: Optional DataLoader for validation data
            
        Returns:
            Tuple of (trained_model, train_losses, val_losses, test_losses, best_epoch)
        """
        print(f"\n=== Training with activation: {activation_fn.upper()} ===")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        
        train_losses = []
        val_losses = []
        test_losses = []
        
        try:
            for epoch in range(self.epochs):
                # Training phase
                model.train()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    output = model(X_batch).squeeze()
                    loss = self.criterion(output, y_batch.squeeze())
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)
                
                # Validation phase
                model.eval()
                
                # If we have a dedicated validation set, use it
                if validation_loader is not None:
                    val_loss = 0
                    with torch.no_grad():
                        for X_val, y_val in validation_loader:
                            X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                            output = model(X_val).squeeze()
                            loss = self.criterion(output, y_val.squeeze())
                            val_loss += loss.item()
                    val_loss /= len(validation_loader)
                    
                    # Also calculate test loss for monitoring
                    test_loss = 0
                    with torch.no_grad():
                        for X_test, y_test in test_loader:
                            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                            output = model(X_test).squeeze()
                            loss = self.criterion(output, y_test.squeeze())
                            test_loss += loss.item()
                    test_loss /= len(test_loader)
                    test_losses.append(test_loss)
                    
                    # Use validation loss for early stopping
                    monitoring_loss = val_loss
                    print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Test Loss: {test_loss:.5f}")
                else:
                    # If no validation set, use test set for validation
                    val_loss = 0
                    with torch.no_grad():
                        for X_val, y_val in test_loader:
                            X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                            output = model(X_val).squeeze()
                            loss = self.criterion(output, y_val.squeeze())
                            val_loss += loss.item()
                    val_loss /= len(test_loader)
                    
                    # No separate test loss in this case
                    test_loss = val_loss
                    # Still add to test_losses for consistency
                    test_losses.append(test_loss)
                    
                    # Use test loss (as validation) for early stopping
                    monitoring_loss = val_loss
                    print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.5f} | Test Loss: {val_loss:.5f}")
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # Early stopping check
                if monitoring_loss < best_loss:
                    best_loss = monitoring_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch} with validation loss {best_loss:.5f}")
                        break
            
            # Load best model
            model.load_state_dict(best_model_state)
            return model, train_losses, val_losses, test_losses, best_epoch
            
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            raise
            
        finally:
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def save_model(self, model: nn.Module, activation_fn: str, save_dir: str) -> str:
        """
        Save the trained model to disk.
        
        Args:
            model: Trained PyTorch model
            activation_fn: Name of activation function used
            save_dir: Base directory to save the model
            
        Returns:
            Path to the saved model file
        """
        # Create models directory if it doesn't exist
        models_dir = os.path.join(save_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Save the model
        filename = f"{models_dir}/model_{activation_fn.lower()}.pth"
        torch.save(model.state_dict(), filename)
        print(f"✅ Saved model with {activation_fn.upper()} activation to: {filename}")
        
        return filename 