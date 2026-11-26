# Stock Price Prediction

Predicts the next trading day's closing price for any stock using technical indicators and machine learning.

## Features
- Downloads historical data with `yfinance`
- Engineered features: moving-average ratios, RSI, volatility, volume change, lagged returns
- Models: Ridge Regression and Random Forest
- Compared against a naive baseline (tomorrow = today)
- Metrics: RMSE, MAE, and directional accuracy
- Chronological train/test split (no data leakage)
- Saves prediction and feature-importance plots

## Setup
```bash
git clone https://github.com/<your-username>/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt
```

## Usage
```bash
python stock_prediction.py --ticker AAPL --start 2015-01-01
```

Options: `--ticker`, `--start`, `--end`, `--test_size`

## Methodology
1. Download OHLCV data and compute technical indicators.
2. Target is the **next-day return** (more stationary than raw price), converted back to price for evaluation.
3. Train on the first 80% of dates, test on the most recent 20%.
4. Compare each model to the naive baseline.

## Results
Add your own results and the plots from `outputs/` here after running the script.

## Limitations
Stock markets are noisy and close to efficient, so models like this barely beat the naive baseline. Directional accuracy near 50-55% is typical. This is an educational project, **not financial advice**.

## Future Work
- LSTM / GRU models
- Sentiment features from news
- Walk-forward cross-validation
- Streamlit web app
