"""
Prediction routes
"""
from fastapi import APIRouter, Query, HTTPException, Depends, Request
from pydantic import BaseModel
from models.lstm_predictor import LSTMPredictor

router = APIRouter()

def get_predictor(request: Request) -> LSTMPredictor:
    return request.app.state.predictor

class PredictionRequest(BaseModel):
    ticker: str
    days_ahead: int = 1

class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    predictions: list
    days_ahead: int
    recent_prices: list
    price_change: float
    price_change_percent: float
    timestamp: str

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, predictor: LSTMPredictor = Depends(get_predictor)):
    """
    Predict stock price for given ticker
    
    Args:
        request: Prediction request with ticker and days ahead
        
    Returns:
        Prediction response with predicted price and analysis
    """
    try:
        if not request.ticker:
            raise HTTPException(status_code=400, detail="Ticker is required")
        
        if request.days_ahead < 1 or request.days_ahead > 30:
            raise HTTPException(status_code=400, detail="days_ahead must be between 1 and 30")
        
        result = predictor.predict_next_price(
            request.ticker,
            request.days_ahead
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict")
async def quick_predict(
    ticker: str = Query(..., description="Stock ticker symbol"),
    days_ahead: int = Query(1, description="Days ahead to predict"),
    predictor: LSTMPredictor = Depends(get_predictor),
):
    """
    Quick prediction endpoint for GET requests
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TCS.NS')
        days_ahead: Number of days ahead to predict (1-30)
        
    Returns:
        Prediction data
    """
    try:
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker is required")
        
        if days_ahead < 1 or days_ahead > 30:
            raise HTTPException(status_code=400, detail="days_ahead must be between 1 and 30")
        
        result = predictor.predict_next_price(ticker, days_ahead)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
