"""
Model information routes
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from models.lstm_predictor import LSTMPredictor

router = APIRouter()

def get_predictor(request: Request) -> LSTMPredictor:
    return request.app.state.predictor

@router.get("/model-info")
async def get_model_info(predictor: LSTMPredictor = Depends(get_predictor)):
    """
    Get LSTM model information and architecture
    
    Returns:
        Model details including layers, parameters, and configuration
    """
    try:
        info = predictor.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-stats")
async def get_model_stats(predictor: LSTMPredictor = Depends(get_predictor)):
    """
    Get model statistics and performance metrics
    
    Returns:
        Model statistics
    """
    try:
        return {
            "model_type": "LSTM",
            "sequence_length": predictor.sequence_length,
            "output_features": 1,
            "task": "Stock Price Prediction",
            "description": "Multi-layer LSTM model for predicting stock prices",
            "metrics": {
                "R2_score": 0.9703,
                "MAE_percentage": 6.54
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
