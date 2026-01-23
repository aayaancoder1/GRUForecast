# Stock Price Predictor

A full-stack web application for predicting stock prices using LSTM (Long Short-Term Memory) neural networks. Built with React frontend and FastAPI backend.

## Features

- Real-time stock price predictions using LSTM models
- Support for multiple stock exchanges (NYSE, NASDAQ, NSE)
- Interactive charts and visualizations
- Forecast up to 30 days ahead
- Beautiful, modern UI with Tailwind CSS

## Tech Stack

### Frontend
- React 18
- Recharts for data visualization
- Tailwind CSS for styling
- Axios for API calls

### Backend
- FastAPI
- TensorFlow/Keras for LSTM model
- Twelve Data API for stock data
- NumPy, Pandas for data processing

## Setup Instructions

### Prerequisites
- Python 3.12+
- Node.js 14+
- npm or yarn

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add Twelve Data API key:
   - Create `backend/.env` and set `TWELVE_API_SECRET_KEY`

5. Ensure model file exists:
   - Place your trained `lstm_model.keras` file in `backend/models/` directory

6. Run the server:
```bash
python main.py
```

Backend runs on: http://localhost:8000

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start development server:
```bash
npm start
```

Frontend runs on: http://localhost:3000

## API Endpoints

### POST `/api/predict`
Predict stock price for a given ticker.

**Request:**
```json
{
  "ticker": "AAPL",
  "days_ahead": 5
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "current_price": 247.65,
  "predicted_price": 257.22,
  "predictions": [257.22, 255.76, ...],
  "days_ahead": 5,
  "recent_prices": [...],
  "price_change": 9.57,
  "price_change_percent": 3.86,
  "timestamp": "2024-01-22T10:30:00"
}
```

### GET `/api/model-info`
Get LSTM model architecture and details.

### GET `/health`
Health check endpoint.

## Project Structure

```
stock-predictor/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   ├── lstm_model.keras # Trained LSTM model
│   │   └── lstm_predictor.py
│   ├── data_providers/
│   │   └── twelvedata.py    # Twelve Data API client
│   ├── routes/
│   │   ├── prediction.py   # Prediction endpoints
│   │   └── model_info.py    # Model info endpoints
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   └── index.js
│   └── package.json
└── README.md
```

## Usage

1. Start both backend and frontend servers
2. Open http://localhost:3000 in your browser
3. Enter a stock ticker symbol (e.g., AAPL, MSFT, TCS.NS)
4. Select the number of days to forecast (1-30)
5. Click "Forecast" to get predictions

## Supported Stock Exchanges

- US Stocks: AAPL, GOOGL, MSFT, TSLA, etc.
- Indian Stocks: TCS.NS, RELIANCE.NS, INFY.NS, etc.
- Cryptocurrencies: BTC-USD, ETH-USD

## Notes

- Predictions are for educational purposes only
- Model accuracy depends on training data and market conditions
- Not financial advice
- See `backend/ML_MODEL.md` for full model documentation

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

