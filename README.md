# Enhanced LSTM-ELU for Stock Market Forecasting

This project implements an enhanced LSTM model with ELU activation for stock market forecasting, as described in the research paper. The full-stack application includes a Python backend for model training/prediction and a Next.js frontend for visualization.

## Features

- LSTM neural network with ELU activation function
- Stock data collection from Yahoo Finance
- Model training and evaluation pipeline
- Interactive visualization of predictions
- Comparison between baseline LSTM and enhanced LSTM-ELU

## Tech Stack

**Backend:**
- Python 3.13+
- PyTorch (Deep Learning)
- yfinance (Data Collection)
- pandas (Data Processing)
- numpy (Numerical Operations)
- matplotlib (Visualization)
- FastAPI (API)

**Frontend:**
- Next.js 15
- React
- Apexchart/React-chart(Data Visualization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/whoisbel/Exponential-Long-Short-Term-Memory
cd Exponential-Long-Short-Term-Memory
```

2. Set up the backend:
```bash
cd backend
uv sync
uv run src/scripts/train_app.py # FOR TRAINING
uv run uvicorn main:app --reload # Running the FASTAPI backend
```

3. Set up the frontend:
```bash
cd ../frontend
npm install
```

## Configuration

Modify the model parameters in `backend/src/scripts/config.json`:

```json
{
  "60-20_split_baseline": {
    "train_size": 0.6,
    "test_size": 0.2,
    "validation_size": 0.2,
    "hidden_sizes": [124],
    "dropout": 0.3,
    "seq_len": 28,
    "batch_size": 64,
    "epochs": 25,
    "learning_rate": 0.0015,
    "patience": 15
  },
  "80-20_split_enhanced": {
    "train_size": 0.8,
    "test_size": 0.2,
    "hidden_sizes": [50, 50, 50],
    "dropout": 0.2,
    "seq_len": 60,
    "batch_size": 128,
    "epochs": 100,
    "learning_rate": 0.001,
    "patience": 15
  }
}
```

## Usage

1. Run the backend server:
```bash
cd backend
uv sync
uv run uvicorn main:app --reload
```

2. Run the frontend development server:
```bash
cd ../frontend
npm install
npm run dev
```

3. Access the application at `http://localhost:3000`

## Data Collection

The application automatically fetches stock data from Yahoo Finance for:
- Air Liquide (AI.PA)
- Date range: January 2010 - December 2024
- OHLCV (Open, High, Low, Close, Volume) data

## Model Training

The system provides two models:
1. **Baseline LSTM**: Traditional LSTM with Tanh activation
2. **Enhanced LSTM-ELU**: Modified LSTM with ELU activation

Training metrics tracked:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Results

Based on the research paper, the enhanced LSTM-ELU model achieved:
- 75.64% reduction in MAE (1.90 vs 7.78)
- 75.37% reduction in RMSE (2.49 vs 10.10)
- R² score of 0.9852 (vs 0.7566 for baseline)

## Acknowledgments

This implementation is based on the research paper:
"Enhanced Long Short-Term Memory with Exponential Linear Unit for Stock Market Forecasting" by Dutaro, Belciña, and Cañedo (University of Mindanao, 2023)