# LSTM Model Training Documentation

## Data Processing

### Frameworks and Libraries Used
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations and array operations
- **Scikit-learn**: Data preprocessing and metrics
- **PyTorch**: Deep learning framework
- **Matplotlib**: Data visualization

### Data Loading and Preprocessing
- **Source**: `air_liquide.csv` containing historical stock price data
- **Key Components**:
  ```python
  df = pd.read_csv("datasets/air_liquide.csv")
  close_prices = df["Close"].values.reshape(-1, 1)
  ```

### Data Cleaning
- **Input Data**: Historical stock prices from L'Air Liquide
- **Features Used**: Closing prices only
- **Data Shape**: Reshaped to 2D array (n_samples, 1) for single feature
- **Missing Values**: Handled by pandas during loading

### Data Normalization
- **Method**: Min-Max Scaling (MinMaxScaler)
- **Range**: Values scaled between 0 and 1
- **Purpose**: 
  - Prevents numerical instability
  - Makes training more efficient
  - Helps with gradient descent
- **Implementation**:
  ```python
  scaler = MinMaxScaler()
  scaled_prices = scaler.fit_transform(close_prices)
  ```

### Train-Test Split
- **Split Ratio**: 80% training, 20% testing
- **Implementation**:
  ```python
  train_size = int(0.8 * len(X))
  X_train, y_train = X[:train_size], y[:train_size]
  X_test, y_test = X[train_size:], y[train_size:]
  ```
- **Purpose**:
  - Training set: Model learning
  - Test set: Independent evaluation
  - Prevents data leakage

### Sequence Creation
- **Window Size**: 60 days (SEQ_LEN)
- **Process**:
  - Creates sequences of 60 days to predict the next day
  - Sliding window approach
- **Implementation**:
  ```python
  X, y = [], []
  for i in range(SEQ_LEN, len(scaled_prices)):
      X.append(scaled_prices[i - SEQ_LEN : i])
      y.append(scaled_prices[i])
  ```
- **Data Structure**:
  - X: 3D array (n_samples, sequence_length, n_features)
  - y: 2D array (n_samples, n_features)
- **Purpose**:
  - Creates temporal dependencies
  - Enables LSTM to learn patterns over time
  - Maintains order of historical data

### Data Loading for Training
- **PyTorch DataLoader**:
  ```python
  train_ds = TensorDataset(X_train, y_train)
  test_ds = TensorDataset(X_test, y_test)
  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
  ```
- **Features**:
  - Batch processing
  - Shuffling for training
  - Memory efficiency
  - Parallel data loading

## Model Development

### Model Architecture
- **Type**: Deep LSTM with custom implementation
- **Layers**:
  1. Three LSTM layers with dropout
  2. Final fully connected layer
- **Key Parameters**:
  - Hidden Size: 50
  - Input Size: 1 (Close price)
  - Output Size: 1 (Predicted price)
  - Dropout Rate: 0.2
  - Number of Layers: 1 per LSTM block

### Model Structure
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZES, activation_fn="tanh"):
        # Three LSTM layers with dropout
        self.lstm1 = CustomLSTM(input_size, hidden_size, ...)
        self.lstm2 = CustomLSTM(hidden_size, hidden_size, ...)
        self.lstm3 = CustomLSTM(hidden_size, hidden_size, ...)
        # Final prediction layer
        self.fc = nn.Linear(hidden_size, OUTPUT_SIZE)
```

## Model Training and Testing

### Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 64
- **Early Stopping**: Patience of 15 epochs
- **Loss Function**: Mean Squared Error (MSE)

### Training Process
1. **Data Preparation**:
   ```python
   train_ds = TensorDataset(X_train, y_train)
   train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
   ```

2. **Training Loop**:
   - Forward pass
   - Loss calculation
   - Backward pass
   - Parameter updates
   - Early stopping check

3. **Model Saving**:
   - Saves best model based on validation loss
   - Stores model configuration in JSON
   - Saves model weights in .pth format

### Testing Process
- Uses 20% of data as test set
- Evaluates model performance on unseen data
- Generates predictions for visualization

## Evaluation Metrics

### Performance Metrics
1. **Mean Absolute Error (MAE)**:
   - Average absolute difference between predicted and actual prices
   - Less sensitive to outliers than RMSE

2. **Root Mean Square Error (RMSE)**:
   - Square root of average squared differences
   - More sensitive to large errors
   - In same units as original data

3. **R-squared (R²)**:
   - Measures how well model fits the data
   - Range: 0 to 1 (higher is better)
   - Indicates proportion of variance explained

### Visualization
1. **Loss Curves**:
   - Training vs Validation loss
   - Helps identify overfitting/underfitting
   - Early stopping visualization

2. **Prediction Plots**:
   - Actual vs Predicted prices
   - Separate plots for TANH and ELU activations
   - Shows model performance on both training and test sets

3. **Metrics Table**:
   - Comparative view of all metrics
   - Easy comparison between TANH and ELU models

### Results Storage
- Metrics saved in JSON format
- Visualizations saved as PNG files
- Model configurations preserved for reproducibility

## File Structure
```
saved_models/
├── model_config.json        # Training configuration
├── model_tanh.pth          # TANH model weights
├── model_elu.pth           # ELU model weights
├── loss_curves.png         # Training/validation loss
├── actual-tanh-elu.png     # Combined predictions
├── actual-tanh.png         # TANH predictions
├── actual-elu.png          # ELU predictions
├── evaluation_metrics.json # Performance metrics
└── evaluation_metrics_table.png # Metrics visualization
``` 