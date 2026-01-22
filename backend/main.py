"""
FastAPI backend for Stock Price Prediction using LSTM
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.prediction import router as prediction_router
from routes.model_info import router as model_info_router
import os
from dotenv import load_dotenv
from models.lstm_predictor import LSTMPredictor
from data_providers.twelvedata import TwelveDataProvider

load_dotenv()

app = FastAPI(
    title="Stock Price Prediction API",
    description="LSTM-based stock price prediction using historical data",
    version="1.0.0"
)

# Predictor initialization
def create_predictor() -> LSTMPredictor:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.getenv("MODEL_PATH", os.path.join(base_dir, "models", "lstm_model.keras"))
    scaler_path = os.getenv("SCALER_PATH", os.path.join(base_dir, "models", "scaler.pkl"))
    sequence_length = int(os.getenv("SEQUENCE_LENGTH", "100"))
    data_days = int(os.getenv("DATA_DAYS", "365"))
    data_interval = os.getenv("DATA_INTERVAL", "1day")

    provider = TwelveDataProvider.from_env(days=data_days, interval=data_interval)
    return LSTMPredictor(model_path, scaler_path, sequence_length, data_provider=provider)


@app.on_event("startup")
async def startup_event() -> None:
    app.state.predictor = create_predictor()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(prediction_router, prefix="/api", tags=["Predictions"])
app.include_router(model_info_router, prefix="/api", tags=["Model Info"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Stock Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/predict",
            "model_info": "/api/model-info",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
