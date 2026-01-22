# Backend Setup Instructions

## Quick Start

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Twelve Data API Key
Create a `.env` file with `TWELVE_API_SECRET_KEY` so the backend can fetch market data.

### 4. Add Your Model
Place your trained `lstm_model.keras` file in the `models/` folder.

### 5. Run the Server
```bash
python main.py
```

API runs at: http://localhost:8000

---

## API Endpoints

### 1. Predict Stock Price
**POST** `/api/predict`

Request:
```json
{
  "ticker": "TCS.NS",
  "days_ahead": 5
}
```

Response:
```json
{
  "ticker": "TCS.NS",
  "current_price": 3500.00,
  "predicted_price": 3550.25,
  "predictions": [3520.15, 3535.80, 3550.25],
  "price_change": 50.25,
  "price_change_percent": 1.43,
  "recent_prices": [...],
  "timestamp": "2024-01-22T10:30:00"
}
```

### 2. Get Model Info
**GET** `/api/model-info`

Returns model architecture and details.

### 3. Get Model Stats
**GET** `/api/model-stats`

Returns performance metrics (R² Score, MAE, etc.)

### 4. Health Check
**GET** `/health`

Returns: `{"status": "healthy"}`

---

## Environment Variables (.env)

```env
PORT=8000
MODEL_PATH=./models/lstm_model.keras
SCALER_PATH=./models/scaler.pkl
SEQUENCE_LENGTH=100
TWELVE_API_SECRET_KEY=your_twelvedata_api_key
DATA_DAYS=365
DATA_INTERVAL=1day
```

---

## Troubleshooting

**Module not found error?**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Port already in use?**
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
```

**Model loading issues?**
- Ensure model file is in `models/` folder
- Verify TensorFlow 2.14.0 is installed
- Check file permissions
