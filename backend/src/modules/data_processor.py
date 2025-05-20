import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

class DataProcessor:
    """
    Class to handle loading, preprocessing, and feature engineering for stock data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor with configuration.
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config
        self.feature_columns = ['Close', 'Return_1D', 'Return_5D']
        self.df = None
        self.df_processed = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load the dataset from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with the loaded data
        """
        print(f"Loading data from: {filepath}")
        self.df = pd.read_csv(filepath)
        return self.df
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe.
        
        Returns:
            DataFrame with added indicators
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        self.df_processed = self.df.copy()
        
        # Calculate 1-day and 5-day returns
        self.df_processed['Return_1D'] = self.df_processed['Close'].pct_change(1).replace([np.inf, -np.inf], np.nan)
        self.df_processed['Return_5D'] = self.df_processed['Close'].pct_change(5).replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN values
        self.df_processed = self.df_processed.dropna()
        
        return self.df_processed
    
    def prepare_sequences(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prepare sequences for LSTM training.
        
        Returns:
            Tuple containing X sequences, y values, and dataset info dictionary
        """
        if self.df_processed is None:
            raise ValueError("No processed data available. Call add_technical_indicators() first.")
            
        seq_len = self.config.get('seq_len', 60)
        
        # Extract features
        features = self.df_processed[self.feature_columns].values
        
        # Clean features - replace infinities and very large values
        feature_df = pd.DataFrame(features, columns=self.feature_columns)
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN values with median
        for col in self.feature_columns:
            if feature_df[col].isna().sum() > 0:
                median_val = feature_df[col].median()
                feature_df[col].fillna(median_val, inplace=True)
                print(f"Replaced NaNs in {col} with median: {median_val:.6f}")
        
        # Scale features
        features = feature_df.values
        scaled_features = self.feature_scaler.fit_transform(features)
        
        # Prepare sequences for LSTM
        X, y = [], []
        for i in range(seq_len, len(scaled_features)):
            X.append(scaled_features[i - seq_len : i])
            y.append(self.df_processed['Close'].iloc[i])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Scale target
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Dataset info for reporting
        dataset_info = {
            "original_size": len(self.df) if self.df is not None else 0,
            "processed_size": len(self.df_processed) if self.df_processed is not None else 0,
            "sequence_dataset_size": len(X),
            "feature_columns": self.feature_columns,
            "sequence_length": seq_len
        }
        
        return X, y_scaled, dataset_info
    
    def create_data_loaders(self, X: np.ndarray, y: np.ndarray) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Split data and create PyTorch DataLoaders.
        
        Args:
            X: Feature sequences
            y: Target values
            
        Returns:
            Tuple containing train_loader, test_loader, and val_loader (if applicable)
        """
        train_ratio = self.config.get('train_size', 0.7)
        test_ratio = self.config.get('test_size', 0.15)
        validation_ratio = self.config.get('validation_size', 0.15)
        batch_size = self.config.get('batch_size', 32)
        
        has_validation = validation_ratio > 0.0
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        if has_validation:
            # Calculate split sizes
            train_size = int(train_ratio * len(X))
            validation_size = int(validation_ratio * len(X))
            
            # Split data
            X_train = X_tensor[:train_size]
            y_train = y_tensor[:train_size]
            
            X_val = X_tensor[train_size:train_size+validation_size]
            y_val = y_tensor[train_size:train_size+validation_size]
            
            X_test = X_tensor[train_size+validation_size:]
            y_test = y_tensor[train_size+validation_size:]
            
            # Create datasets
            train_ds = TensorDataset(X_train, y_train)
            val_ds = TensorDataset(X_val, y_val)
            test_ds = TensorDataset(X_test, y_test)
            
            # Create data loaders
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size)
            test_loader = DataLoader(test_ds, batch_size=batch_size)
            
            print(f"Dataset split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
            
            return train_loader, test_loader, val_loader
        else:
            # Standard train/test split
            train_size = int(train_ratio * len(X))
            
            # Split data
            X_train = X_tensor[:train_size]
            y_train = y_tensor[:train_size]
            
            X_test = X_tensor[train_size:]
            y_test = y_tensor[train_size:]
            
            # Create datasets
            train_ds = TensorDataset(X_train, y_train)
            test_ds = TensorDataset(X_test, y_test)
            
            # Create data loaders
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size)
            
            print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")
            
            return train_loader, test_loader, None
    
    def save_processed_data(self, save_dir: str) -> None:
        """
        Save processed dataframes to CSV.
        
        Args:
            save_dir: Directory to save the data
        """
        if self.df_processed is None:
            raise ValueError("No processed data available. Call add_technical_indicators() first.")
            
        # Create data directory if it doesn't exist
        data_dir = os.path.join(save_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save the processed dataframe to CSV
        self.df_processed.to_csv(f"{data_dir}/processed_data.csv", index=False)
        print(f"📊 Saved processed dataframe to: {data_dir}/processed_data.csv")
        
        # If we have the original dataframe, create a comparison
        if self.df is not None and 'Date' in self.df.columns:
            # Assuming 'Date' is a common column between original and processed data
            original_subset = self.df[['Date', 'Close']].copy()
            original_subset.rename(columns={'Close': 'Original_Close'}, inplace=True)
            
            # Create processed subset with common indices
            processed_subset = self.df_processed[['Date', 'Close', 'Return_1D', 'Return_5D']].copy()
            processed_subset.rename(columns={'Close': 'Processed_Close'}, inplace=True)
            
            # Merge on Date to see differences
            comparison_df = pd.merge(original_subset, processed_subset, on='Date', how='outer')
            
            # Compute differences
            comparison_df['Difference'] = comparison_df['Original_Close'] - comparison_df['Processed_Close']
            
            # Save comparison
            comparison_df.to_csv(f"{data_dir}/data_comparison.csv", index=False)
            print(f"📊 Saved original vs processed data comparison to: {data_dir}/data_comparison.csv")
        
        # Save a few sample sequences
        X, y, _ = self.prepare_sequences()
        if len(X) > 0:
            sample_size = min(5, len(X))
            seq_data = []
            seq_len = self.config.get('seq_len', 60)
            
            for i in range(sample_size):
                seq = X[i]
                target = y[i][0]
                
                # For each timestep in the sequence
                for t in range(seq_len):
                    row = {'sample_id': i, 'timestep': t, 'target': target}
                    
                    # Add each feature
                    for f_idx, feature_name in enumerate(self.feature_columns):
                        row[f'feature_{feature_name}'] = seq[t, f_idx]
                    
                    seq_data.append(row)
            
            # Convert to dataframe and save
            seq_df = pd.DataFrame(seq_data)
            seq_df.to_csv(f"{data_dir}/sequence_samples.csv", index=False)
            print(f"📊 Saved {sample_size} sample sequences to: {data_dir}/sequence_samples.csv")
    
    def get_scalers(self) -> Tuple[MinMaxScaler, MinMaxScaler]:
        """
        Get the fitted scalers.
        
        Returns:
            Tuple containing (feature_scaler, target_scaler)
        """
        return self.feature_scaler, self.target_scaler
    
    def get_sequence_data_for_visualization(self) -> Dict[str, np.ndarray]:
        """
        Prepare sequence data specifically for visualization in the correct time order.
        
        Returns:
            Dictionary with X features, y targets, and dates
        """
        if self.df_processed is None:
            raise ValueError("No processed data available. Call add_technical_indicators() first.")
            
        seq_len = self.config.get('seq_len', 60)
        
        # Extract features
        features = self.df_processed[self.feature_columns].values
        
        # Clean features - replace infinities and very large values
        feature_df = pd.DataFrame(features, columns=self.feature_columns)
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN values with median
        for col in self.feature_columns:
            if feature_df[col].isna().sum() > 0:
                median_val = feature_df[col].median()
                feature_df[col].fillna(median_val, inplace=True)
        
        # Scale features
        features = feature_df.values
        scaled_features = self.feature_scaler.transform(features)  # Use transform instead of fit_transform
        
        # Prepare sequences for LSTM
        X, y, dates = [], [], []
        for i in range(seq_len, len(scaled_features)):
            X.append(scaled_features[i - seq_len : i])
            y.append(self.df_processed['Close'].iloc[i])
            dates.append(self.df_processed['Date'].iloc[i])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Scale target
        y_scaled = self.target_scaler.transform(y)  # Use transform instead of fit_transform
        
        return {
            "X": X,
            "y": y_scaled,
            "dates": np.array(dates)
        } 