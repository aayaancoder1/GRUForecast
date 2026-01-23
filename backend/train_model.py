"""
Train LSTM model using Twelve Data for S&P 500 tickers.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from dotenv import load_dotenv

from data_providers.twelvedata import TwelveDataProvider
from data_providers.ticker_universe import load_sp500_tickers


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "lstm_model.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
METRICS_PATH = BASE_DIR / "models" / "training_metrics.json"
CACHE_DIR = BASE_DIR / "data_cache"


def build_sequences(
    series: np.ndarray, sequence_length: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - sequence_length - horizon + 1):
        X.append(series[i : i + sequence_length])
        y.append(series[i + sequence_length : i + sequence_length + horizon])
    return np.array(X), np.array(y)


QUANTILES = [0.1, 0.5, 0.9]


def quantile_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # y_true: (batch, horizon), y_pred: (batch, horizon, q)
    losses = []
    for q_index, q in enumerate(QUANTILES):
        errors = y_true - y_pred[:, :, q_index]
        losses.append(tf.maximum(q * errors, (q - 1) * errors))
    return tf.reduce_mean(tf.add_n(losses))


def median_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    median = y_pred[:, :, 1]
    return tf.reduce_mean(tf.abs(y_true - median))


def median_mape(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    median = y_pred[:, :, 1]
    return tf.reduce_mean(tf.abs((y_true - median) / tf.maximum(y_true, 1e-6))) * 100


def build_model(sequence_length: int, horizon: int) -> keras.Model:
    inputs = keras.layers.Input(shape=(sequence_length, 1))
    x = keras.layers.GRU(64, return_sequences=True)(inputs)
    x = keras.layers.Dropout(0.25)(x)
    attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = keras.layers.Add()([x, attn])
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.GRU(64)(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(horizon * len(QUANTILES))(x)
    outputs = keras.layers.Reshape((horizon, len(QUANTILES)))(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=quantile_loss,
        metrics=[median_mae, median_mape],
    )
    return model


def fetch_ticker_series(
    provider: TwelveDataProvider, ticker: str, days: int
) -> np.ndarray:
    df = provider.fetch_time_series(ticker, days=days)
    if "Close" not in df.columns or df["Close"].isna().all():
        raise ValueError(f"No close data for {ticker}")
    return df["Close"].astype(float).values


def load_cached_series(ticker: str, days: int, ttl_days: int) -> np.ndarray | None:
    cache_path = CACHE_DIR / f"{ticker}.csv"
    if not cache_path.exists():
        return None
    mtime = cache_path.stat().st_mtime
    age_seconds = time.time() - mtime
    if age_seconds > ttl_days * 86400:
        return None
    data = np.loadtxt(cache_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        closes = data[1:]
    else:
        closes = data[:, 1]
    if len(closes) < days:
        return None
    return closes[-days:]


def save_cached_series(ticker: str, series: np.ndarray) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}.csv"
    timestamps = np.arange(len(series))
    data = np.column_stack([timestamps, series])
    header = "index,close"
    np.savetxt(cache_path, data, delimiter=",", header=header, comments="")


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-6)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100)


def main() -> None:
    load_dotenv()
    sequence_length = int(os.getenv("SEQUENCE_LENGTH", "100"))
    horizon = int(os.getenv("TRAIN_HORIZON", "30"))
    days = int(os.getenv("TRAIN_DAYS", "730"))
    max_tickers = int(os.getenv("TRAIN_MAX_TICKERS", "200"))
    min_price = float(os.getenv("TRAIN_MIN_PRICE", "5"))
    min_history = int(os.getenv("TRAIN_MIN_HISTORY", str(sequence_length + horizon + 10)))
    batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "64"))
    epochs = int(os.getenv("TRAIN_EPOCHS", "20"))
    seed = int(os.getenv("TRAIN_SEED", "42"))
    use_cache = os.getenv("TRAIN_USE_CACHE", "false").lower() == "true"
    cache_ttl_days = int(os.getenv("TRAIN_CACHE_TTL_DAYS", "7"))

    provider = TwelveDataProvider.from_env(days=days, interval="1day")
    tickers = load_sp500_tickers()[:max_tickers]

    X_all, y_all = [], []
    raw_all = []
    skipped = []
    skipped_low_price = []
    skipped_short_history = []
    failures = []
    scalers: Dict[str, MinMaxScaler] = {}
    sample_tickers: List[str] = []
    request_delay = float(os.getenv("TRAIN_REQUEST_DELAY", "0.2"))

    for ticker in tickers:
        try:
            series = None
            if use_cache:
                series = load_cached_series(ticker, days, cache_ttl_days)
            if series is None:
                series = fetch_ticker_series(provider, ticker, days)
                save_cached_series(ticker, series)
            if len(series) < min_history:
                skipped_short_history.append(ticker)
                continue
            if float(np.nanmin(series)) < min_price:
                skipped_low_price.append(ticker)
                continue
            raw_all.append(series)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
            X, y = build_sequences(scaled, sequence_length, horizon)
            if len(X) == 0:
                skipped.append(ticker)
                continue
            X_all.append(X)
            y_all.append(y)
            scalers[ticker] = scaler
            sample_tickers.extend([ticker] * len(X))
        except Exception as exc:
            failures.append({"ticker": ticker, "error": str(exc)})
            skipped.append(ticker)
        finally:
            if request_delay > 0:
                time.sleep(request_delay)

    if not X_all:
        failure_sample = failures[:5]
        raise RuntimeError(
            "No training data collected. Sample failures: "
            + json.dumps(failure_sample, indent=2)
        )

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    ticker_array = np.array(sample_tickers)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]
    ticker_array = ticker_array[indices]

    split_idx = int(len(X) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    ticker_val = ticker_array[split_idx:]

    model = build_model(sequence_length, horizon)
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Sanity check: predict a small batch and ensure outputs are finite
    sample_preds = model.predict(X_val[:64], verbose=0)
    if not np.isfinite(sample_preds).all():
        raise RuntimeError("Model produced non-finite predictions during sanity check.")

    # Sanity check: compare normalized scale ranges
    recent_slice = X_val[:64]
    pred_min = float(sample_preds.min())
    pred_max = float(sample_preds.max())
    input_min = float(recent_slice.min())
    input_max = float(recent_slice.max())
    if pred_min < -0.5 or pred_max > 1.5:
        raise RuntimeError(
            f"Predictions out of expected normalized range: {pred_min:.3f}..{pred_max:.3f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    if not raw_all:
        raise RuntimeError("Raw data cache is empty; cannot build scaler.")

    with open(SCALER_PATH, "wb") as f:
        import pickle

        global_scaler = MinMaxScaler(feature_range=(0, 1))
        all_values = np.concatenate(raw_all)
        global_scaler.fit(all_values.reshape(-1, 1))
        print(f"Saving scaler to {SCALER_PATH}")
        pickle.dump(global_scaler, f)

    # Validation metrics in original price scale using per-ticker scalers
    val_preds = model.predict(X_val, verbose=0)
    val_median = val_preds[:, :, 1]
    denorm_true = []
    denorm_pred = []
    for i, ticker in enumerate(ticker_val):
        scaler = scalers.get(ticker)
        if scaler is None:
            continue
        true_vals = scaler.inverse_transform(y_val[i].reshape(-1, 1)).flatten()
        pred_vals = scaler.inverse_transform(val_median[i].reshape(-1, 1)).flatten()
        denorm_true.append(true_vals)
        denorm_pred.append(pred_vals)

    if denorm_true and denorm_pred:
        denorm_true_arr = np.concatenate(denorm_true)
        denorm_pred_arr = np.concatenate(denorm_pred)
        val_mae_usd = float(np.mean(np.abs(denorm_true_arr - denorm_pred_arr)))
        val_mape_pct = mean_absolute_percentage_error(denorm_true_arr, denorm_pred_arr)
        val_smape_pct = symmetric_mape(denorm_true_arr, denorm_pred_arr)
    else:
        val_mae_usd = None
        val_mape_pct = None
        val_smape_pct = None

    metrics = {
        "train_loss": history.history["loss"][-1],
        "train_mae": history.history["median_mae"][-1],
        "train_mape": history.history["median_mape"][-1],
        "val_loss": history.history["val_loss"][-1],
        "val_mae": history.history["val_median_mae"][-1],
        "val_mape": history.history["val_median_mape"][-1],
        "val_mae_usd": val_mae_usd,
        "val_mape_percent": val_mape_pct,
        "val_smape_percent": val_smape_pct,
        "predicted_range": {"min": pred_min, "max": pred_max},
        "input_range": {"min": input_min, "max": input_max},
        "sequence_length": sequence_length,
        "horizon": horizon,
        "days": days,
        "num_samples": int(len(X)),
        "skipped_tickers": skipped,
        "skipped_low_price": skipped_low_price,
        "skipped_short_history": skipped_short_history,
        "failed_tickers": failures,
    }
    print(f"Saving metrics to {METRICS_PATH}")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
