"""
Stock Price Prediction
----------------------
Predicts the next trading day's closing price using technical indicators
and two ML models (Ridge Regression, Random Forest), compared against a
naive baseline ("tomorrow's price = today's price").

Usage:
    python stock_prediction.py --ticker AAPL --start 2015-01-01
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data(ticker, start, end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss)


def add_features(df):
    d = df.copy()
    d["return_1d"] = d["Close"].pct_change()
    for w in (5, 10, 20, 50):
        d[f"ma_{w}_ratio"] = d["Close"] / d["Close"].rolling(w).mean() - 1
    d["volatility_10"] = d["return_1d"].rolling(10).std()
    d["rsi_14"] = rsi(d["Close"])
    d["volume_change"] = d["Volume"].pct_change()
    d["hl_range"] = (d["High"] - d["Low"]) / d["Close"]
    for lag in (1, 2, 3, 5):
        d[f"ret_lag_{lag}"] = d["return_1d"].shift(lag)
    # Target: next day's return (more stationary than raw price)
    d["target_return"] = d["Close"].pct_change().shift(-1)
    return d.replace([np.inf, -np.inf], np.nan)


FEATURES = [
    "return_1d", "ma_5_ratio", "ma_10_ratio", "ma_20_ratio", "ma_50_ratio",
    "volatility_10", "rsi_14", "volume_change", "hl_range",
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
]


def evaluate(name, close, pred_return, actual_return):
    actual_price = close * (1 + actual_return)
    pred_price = close * (1 + pred_return)
    rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
    mae = mean_absolute_error(actual_price, pred_price)
    direction = (np.sign(pred_return) == np.sign(actual_return)).mean() * 100
    print(f"{name:<15} RMSE: {rmse:8.3f} | MAE: {mae:8.3f} | Direction acc: {direction:5.1f}%")
    return pred_price


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    print(f"Downloading {args.ticker} data...")
    df = add_features(load_data(args.ticker, args.start, args.end))
    data = df.dropna(subset=FEATURES)
    latest = data.iloc[[-1]]                      # newest row, used for tomorrow's forecast
    labeled = data.dropna(subset=["target_return"])

    # Chronological split (never shuffle time series!)
    split = int(len(labeled) * (1 - args.test_size))
    train, test = labeled.iloc[:split], labeled.iloc[split:]
    X_train, y_train = train[FEATURES], train["target_return"]
    X_test, y_test = test[FEATURES], test["target_return"]

    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
    }

    print(f"\nTrain: {len(train)} days | Test: {len(test)} days\n")
    results = {}
    results["Naive"] = evaluate("Naive baseline", test["Close"], np.zeros(len(test)), y_test)
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(name, test["Close"], model.predict(X_test), y_test)

    # Feature importance
    rf = models["RandomForest"]
    importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()

    # Plots
    os.makedirs("outputs", exist_ok=True)
    actual = test["Close"] * (1 + y_test)
    plt.figure(figsize=(12, 5))
    plt.plot(actual.index, actual, label="Actual", linewidth=1.5)
    plt.plot(actual.index, results["Ridge"], label="Ridge", alpha=0.8)
    plt.plot(actual.index, results["RandomForest"], label="Random Forest", alpha=0.8)
    plt.title(f"{args.ticker}: Next-Day Close, Actual vs Predicted (test set)")
    plt.xlabel("Date"); plt.ylabel("Price (USD)"); plt.legend(); plt.tight_layout()
    plt.savefig("outputs/predictions.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 5))
    importance.plot(kind="barh")
    plt.title("Random Forest Feature Importance"); plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150); plt.close()

    # Tomorrow's forecast
    last_close = float(latest["Close"].iloc[0])
    pred_ret = float(rf.predict(latest[FEATURES])[0])
    print(f"\nLast close ({latest.index[0].date()}): {last_close:.2f}")
    print(f"Random Forest forecast for next trading day: {last_close * (1 + pred_ret):.2f} "
          f"({pred_ret * 100:+.2f}%)")
    print("\nPlots saved to the 'outputs/' folder.")
    print("Disclaimer: educational project only, not financial advice.")


if __name__ == "__main__":
    main()
