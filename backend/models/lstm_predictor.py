"""
LSTM Model predictor module
"""
import numpy as np
import pandas as pd
from tensorflow import keras
import pickle
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Protocol, Dict, Tuple
from datetime import datetime
import os

from data_providers.twelvedata import TwelveDataProvider, TwelveDataError


QUANTILES = [0.1, 0.5, 0.9]


@keras.utils.register_keras_serializable()
def quantile_loss(y_true, y_pred):
    losses = []
    for q_index, q in enumerate(QUANTILES):
        errors = y_true - y_pred[:, :, q_index]
        losses.append(keras.ops.maximum(q * errors, (q - 1) * errors))
    return keras.ops.mean(keras.ops.add_n(losses))


@keras.utils.register_keras_serializable()
def median_mae(y_true, y_pred):
    median = y_pred[:, :, 1]
    return keras.ops.mean(keras.ops.abs(y_true - median))


@keras.utils.register_keras_serializable()
def median_mape(y_true, y_pred):
    median = y_pred[:, :, 1]
    denom = keras.ops.maximum(keras.ops.abs(y_true), 1e-6)
    return keras.ops.mean(keras.ops.abs((y_true - median) / denom)) * 100


class StockDataProvider(Protocol):
    def fetch_time_series(self, ticker: str, days: Optional[int] = None) -> pd.DataFrame:
        ...

class LSTMPredictor:
    def __init__(
        self,
        model_path: str,
        scaler_path: Optional[str] = None,
        sequence_length: int = 100,
        data_provider: Optional[StockDataProvider] = None,
    ):
        """
        Initialize LSTM predictor
        
        Args:
            model_path: Path to trained LSTM model (.keras)
            scaler_path: Path to saved scaler (.pkl)
            sequence_length: Number of days for input sequence
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.data_provider = data_provider or TwelveDataProvider.from_env(days=365, interval="1day")
        self.scaler_cache: Dict[str, Tuple[MinMaxScaler, int, float]] = {}
        
        # Resolve absolute paths
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, model_path.lstrip('./'))
        
        if scaler_path and not os.path.isabs(scaler_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            scaler_path = os.path.join(base_dir, scaler_path.lstrip('./'))
        
        # Load model
        if os.path.exists(model_path):
            self.model = keras.models.load_model(
                model_path,
                custom_objects={
                    "quantile_loss": quantile_loss,
                    "median_mae": median_mae,
                    "median_mape": median_mape,
                },
            )
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}")
        
        # Load or create scaler
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded from {scaler_path}")
        else:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            print("New scaler created (will be fitted on first prediction)")
    
    def fetch_stock_data(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """
        Fetch stock data from Twelve Data

        Args:
            ticker: Stock ticker symbol (e.g., 'TCS.NS')
            days: Number of days of historical data to fetch

        Returns:
            DataFrame with stock data
        """
        try:
            data = self.data_provider.fetch_time_series(ticker, days=days)
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            return data
        except TwelveDataError as e:
            raise Exception(f"Error fetching stock data: {str(e)}")
        except Exception as e:
            raise Exception(f"Error fetching stock data: {str(e)}")
    
    def prepare_data(self, data: np.ndarray, ticker: str) -> tuple[np.ndarray, MinMaxScaler]:
        """
        Prepare data for model input
        
        Args:
            data: Raw price data
            
        Returns:
            Normalized data and scaler
        """
        data = data.reshape(-1, 1)
        last_price = float(data[-1][0])
        cached = self.scaler_cache.get(ticker)
        if cached and cached[1] == len(data) and cached[2] == last_price:
            scaler = cached[0]
            normalized_data = scaler.transform(data)
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_data = scaler.fit_transform(data)
            self.scaler_cache[ticker] = (scaler, len(data), last_price)
        return normalized_data, scaler
    
    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Create sequences for LSTM input
        
        Args:
            data: Normalized data
            
        Returns:
            Sequences of shape (samples, sequence_length, 1)
        """
        sequences = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def predict_next_price(self, ticker: str, days_ahead: int = 1) -> dict:
        """
        Predict next price
        
        Args:
            ticker: Stock ticker symbol
            days_ahead: Number of days ahead to predict
            
        Returns:
            Dictionary with predictions
        """
        try:
            if not self.model:
                raise Exception("Model not loaded")
            
            # Fetch recent data
            data = self.fetch_stock_data(ticker, days=365)
            
            if 'Close' not in data.columns:
                raise Exception("No 'Close' price data available for this ticker")
            
            close_prices = data['Close'].values
            
            if len(close_prices) < self.sequence_length:
                raise Exception(f"Insufficient data: need at least {self.sequence_length} days of data, got {len(close_prices)}")
            
            # Prepare data (per-ticker normalization)
            normalized_data, local_scaler = self.prepare_data(close_prices, ticker)
            
            # Get last sequence
            last_sequence = normalized_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            # Make prediction
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days_ahead):
                pred = self.model.predict(current_sequence, verbose=0)
                # Extract median prediction if quantiles are present
                if pred.ndim == 3 and pred.shape[-1] >= 3:
                    pred_value = float(pred[0, 0, 1])
                else:
                    pred_value = float(pred[0, 0])
                predictions.append(pred_value)
                
                # Update sequence for next iteration
                current_sequence = np.append(current_sequence[:, 1:, :], 
                                           [[[pred_value]]], axis=1)
            
            # Inverse transform predictions
            predictions_array = np.array(predictions).reshape(-1, 1)
            denormalized_predictions = local_scaler.inverse_transform(predictions_array)
            
            # Get current and recent prices - ensure proper conversion to Python scalars
            current_price_val = close_prices[-1]
            if isinstance(current_price_val, np.ndarray):
                current_price = float(current_price_val.item())
            else:
                current_price = float(current_price_val)
            
            recent_prices = []
            for p in close_prices[-30:]:
                if isinstance(p, np.ndarray):
                    recent_prices.append(float(p.item()))
                else:
                    recent_prices.append(float(p))
            
            # Extract predicted price as scalar
            pred_price_val = denormalized_predictions[0, 0]
            if isinstance(pred_price_val, np.ndarray):
                predicted_price = float(pred_price_val.item())
            else:
                predicted_price = float(pred_price_val)

            if predicted_price <= 0 or not np.isfinite(predicted_price):
                raise Exception("Model returned invalid predicted price.")
            
            # Extract all predictions as Python floats
            predictions_list = []
            for p in denormalized_predictions:
                val = p[0]
                if isinstance(val, np.ndarray):
                    predictions_list.append(float(val.item()))
                else:
                    predictions_list.append(float(val))

            if any(not np.isfinite(v) for v in predictions_list):
                raise Exception("Model returned invalid prediction values.")
            
            price_change = float(predicted_price - current_price)
            price_change_percent = float((price_change / current_price) * 100)
            if not np.isfinite(price_change_percent) or abs(price_change_percent) > 200:
                raise Exception("Prediction change is unrealistically large.")

            return {
                "ticker": ticker,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "predictions": predictions_list,
                "days_ahead": days_ahead,
                "recent_prices": recent_prices,
                "price_change": price_change,
                "price_change_percent": price_change_percent,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")
    
    def get_model_info(self) -> dict:
        """
        Get model information
        
        Returns:
            Dictionary with model details
        """
        if not self.model:
            return {"error": "Model not loaded"}
        
        model_config = self.model.get_config()
        
        return {
            "name": model_config.get("name", "Unknown"),
            "layers": len(self.model.layers),
            "parameters": int(self.model.count_params()),
            "sequence_length": self.sequence_length,
            "layer_details": [
                {
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "config": layer.get_config() if hasattr(layer, 'get_config') else {}
                }
                for layer in self.model.layers
            ]
        }
