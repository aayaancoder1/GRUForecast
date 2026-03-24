---

# GRUForecast

[![Backend](https://img.shields.io/badge/Backend-Hugging%20Face%20Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces/YOUR_HF_USERNAME/gruforecast-backend)
[![Frontend](https://img.shields.io/badge/Frontend-Vercel-black?logo=vercel)](https://your-vercel-url.vercel.app)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fyzanshaik/GRUForecast/blob/main/Stock_Predictor_Colab.ipynb)

A full-stack web application for predicting stock prices using GRU + Multi-Head Attention neural networks with quantile regression for uncertainty estimation.

## Model Architecture

**Key Features:**
- **GRU + Attention**: Captures temporal patterns and focuses on relevant time steps
- **Quantile Outputs**: Predicts 10th, 50th, and 90th percentiles for uncertainty estimation
- **Per-Ticker Normalization**: Handles stocks with vastly different price ranges
- **30-Day Horizon**: Forecasts up to 30 days ahead with confidence bands

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed model documentation.

## Try it Now

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fyzanshaik/GRUForecast/blob/main/Stock_Predictor_Colab.ipynb)

## Features

- Real-time stock price predictions with uncertainty bands
- Support for multiple stock exchanges (NYSE, NASDAQ, NSE)
- Interactive charts and visualizations
- Forecast up to 30 days ahead
- Beautiful, modern UI with Tailwind CSS

## Tech Stack

**Frontend** — React 18, Recharts, Tailwind CSS, Axios

**Backend** — FastAPI, TensorFlow/Keras, Twelve Data API, NumPy, Pandas

**Model** — GRU layers, Multi-Head Self-Attention, Quantile Loss, Per-ticker MinMax normalization

**Hosting** — Vercel (frontend), Hugging Face Spaces (backend)

---

## Setup Instructions

### Prerequisites
- Python 3.12+
- Node.js 14+
- npm or yarn
- [Twelve Data API key](https://twelvedata.com) (free tier works)

### Backend — Local Development

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Copy the example env file and fill in your API key:

```bash
cp .env.example .env
# Edit .env and add your TWELVE_API_SECRET_KEY
```

Then run:

```bash
python main.py
```

Backend runs on: http://localhost:8000

### Frontend — Local Development

```bash
cd frontend
npm install
npm start
```

Frontend runs on: http://localhost:3000

---

## Deployment

### Backend — Hugging Face Spaces

The backend is hosted as a Docker Space on Hugging Face. This handles the full TensorFlow stack for free with no cold-start spin-down.

**Steps to deploy your own instance:**

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Docker**
   - Visibility: **Public**

2. Clone the Space repo:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/gruforecast-backend
   ```

3. Copy backend files into the Space:
   ```bash
   cp -r backend/* gruforecast-backend/
   ```

4. Upload `lstm_model.keras` via the Space's **Files** tab under `models/lstm_model.keras`

5. Add your API key under **Settings → Variables and secrets**:
   - `TWELVE_API_SECRET_KEY` = your Twelve Data key

6. Push to deploy:
   ```bash
   cd gruforecast-backend
   git add .
   git commit -m "deploy backend"
   git push
   ```

The Space URL will be: `https://YOUR_HF_USERNAME-gruforecast-backend.hf.space`

### Frontend — Vercel

Set the backend URL in your Vercel project's **Environment Variables**:

```
REACT_APP_API_URL=https://YOUR_HF_USERNAME-gruforecast-backend.hf.space
```

Or if the frontend uses Vite:

```
VITE_API_URL=https://YOUR_HF_USERNAME-gruforecast-backend.hf.space
```

---

## API Reference

### `POST /api/predict`

```json
// Request
{ "ticker": "AAPL", "days_ahead": 5 }

// Response
{
  "ticker": "AAPL",
  "current_price": 247.65,
  "predicted_price": 257.22,
  "predictions": [257.22, 255.76, "..."],
  "days_ahead": 5,
  "recent_prices": ["..."],
  "price_change": 9.57,
  "price_change_percent": 3.86,
  "timestamp": "2024-01-22T10:30:00"
}
```

### `GET /api/predict?ticker=AAPL&days_ahead=5`

Same response, quick GET version.

### `GET /api/model-info`

Returns model architecture and layer details.

### `GET /health`

Returns `{ "status": "healthy" }`.

---

## Project Structure

```
GRUForecast/
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── train_model.py             # Training script
│   ├── Dockerfile                 # Docker config for HF Spaces
│   ├── .env.example               # Environment variable template
│   ├── models/
│   │   ├── lstm_model.keras       # Trained model
│   │   ├── lstm_predictor.py      # Prediction logic
│   │   └── training_metrics.json  # Training results
│   ├── data_providers/
│   │   ├── twelvedata.py          # Twelve Data API client
│   │   └── ticker_universe.py     # S&P 500 tickers
│   ├── routes/
│   │   ├── prediction.py          # Prediction endpoints
│   │   └── model_info.py          # Model info endpoints
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Main React component
│   │   └── components/            # UI components
│   └── package.json
├── Stock_Predictor_Colab.ipynb    # Google Colab notebook
├── ARCHITECTURE.md                # Model documentation
└── README.md
```

## Model Performance

| Metric | Value |
|--------|-------|
| Validation MAE (USD) | ~$7.91 |
| Validation MAPE | ~4.57% |
| Validation SMAPE | ~4.53% |
| Training Samples | 15,000+ |

## Notes

- Predictions are for educational purposes only
- Not financial advice
- Uncertainty bands represent 80% confidence interval

## License

MIT
