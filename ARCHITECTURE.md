# GRUForecast - Model Architecture & Documentation

A complete guide to understanding how the stock price prediction model works.

---

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Training Process](#training-process)
5. [Inference Process](#inference-process)
6. [Worked Example](#worked-example)
7. [Quantile Predictions](#quantile-predictions)

---

## Overview

GRUForecast is a deep learning model that predicts stock prices up to 30 days ahead. It uses:

- **GRU (Gated Recurrent Unit)** layers to learn temporal patterns
- **Multi-Head Attention** to focus on important time steps
- **Quantile Regression** to provide uncertainty estimates

```mermaid
flowchart LR
    A[Historical Prices] --> B[Normalization]
    B --> C[GRU + Attention Model]
    C --> D[30-Day Forecast]
    D --> E[Uncertainty Bands]
```

### Key Features

| Feature | Description |
|---------|-------------|
| Input | 100 days of closing prices |
| Output | 30-day forecast with confidence bands |
| Architecture | GRU + Multi-Head Attention |
| Loss Function | Quantile Loss (Pinball) |
| Outputs | 3 quantiles: 10%, 50%, 90% |

---

## Model Architecture

### High-Level Architecture

```mermaid
flowchart TB
    subgraph Input
        A[Price Sequence<br/>Shape: 100 x 1]
    end
    
    subgraph "GRU Block 1"
        B[GRU Layer<br/>64 units, return sequences]
        C[Dropout 25%]
    end
    
    subgraph "Attention Block"
        D[Multi-Head Attention<br/>4 heads, key_dim=16]
        E[Residual Connection]
        F[Layer Normalization]
    end
    
    subgraph "GRU Block 2"
        G[GRU Layer<br/>64 units]
        H[Dropout 25%]
    end
    
    subgraph Output
        I[Dense Layer<br/>90 units]
        J[Reshape<br/>30 x 3]
    end
    
    A --> B --> C --> D
    C --> E
    D --> E --> F --> G --> H --> I --> J
```

### Layer-by-Layer Breakdown

#### 1. Input Layer
```
Shape: (batch_size, 100, 1)
```
- Takes 100 days of normalized closing prices
- Each price is a single feature (univariate time series)

#### 2. First GRU Layer
```
GRU(64, return_sequences=True)
```
- 64 hidden units
- Returns output for each time step (100 outputs)
- Learns short-term and medium-term patterns

```mermaid
flowchart LR
    subgraph "GRU Cell"
        direction TB
        R[Reset Gate] 
        U[Update Gate]
        H[Hidden State]
    end
    
    X1[Day 1] --> GRU1[GRU] --> H1[h₁]
    X2[Day 2] --> GRU2[GRU] --> H2[h₂]
    X3[...] --> GRU3[GRU] --> H3[...]
    X100[Day 100] --> GRU100[GRU] --> H100[h₁₀₀]
    
    H1 --> GRU2
    H2 --> GRU3
    H3 --> GRU100
```

#### 3. Dropout Layer
```
Dropout(0.25)
```
- Randomly drops 25% of connections during training
- Prevents overfitting

#### 4. Multi-Head Self-Attention
```
MultiHeadAttention(num_heads=4, key_dim=16)
```
- 4 parallel attention heads
- Each head learns different relationships
- Allows model to focus on relevant past days

```mermaid
flowchart TB
    subgraph "Multi-Head Attention"
        direction LR
        Q[Query] 
        K[Key]
        V[Value]
        
        Q --> H1[Head 1]
        K --> H1
        V --> H1
        
        Q --> H2[Head 2]
        K --> H2
        V --> H2
        
        Q --> H3[Head 3]
        K --> H3
        V --> H3
        
        Q --> H4[Head 4]
        K --> H4
        V --> H4
        
        H1 --> Concat[Concatenate]
        H2 --> Concat
        H3 --> Concat
        H4 --> Concat
        
        Concat --> Linear[Linear]
    end
```

**Why Attention?**
- Can identify which past days are most relevant for prediction
- Example: A price spike 30 days ago might be more relevant than yesterday's small movement

#### 5. Residual Connection + Layer Normalization
```python
x = Add()([x, attention_output])  # Residual
x = LayerNormalization()(x)
```
- Residual connection helps gradients flow during training
- Layer normalization stabilizes training

#### 6. Second GRU Layer
```
GRU(64)
```
- Compresses the sequence into a single vector
- Output shape: (batch_size, 64)

#### 7. Output Layers
```python
Dense(90)           # 30 days × 3 quantiles = 90
Reshape((30, 3))    # Reshape to (30, 3)
```
- Produces 30 predictions, each with 3 quantile values

---

## Data Pipeline

### Training Data Flow

```mermaid
flowchart TB
    subgraph "Data Collection"
        A[S&P 500 Tickers] --> B[yfinance API]
        B --> C[Raw Price Data<br/>730 days per stock]
    end
    
    subgraph "Preprocessing"
        C --> D[Per-Ticker Normalization<br/>MinMax 0-1]
        D --> E[Sliding Window<br/>Sequence Builder]
    end
    
    subgraph "Sequence Building"
        E --> F[Input: Days 1-100<br/>Target: Days 101-130]
        F --> G[Input: Days 2-101<br/>Target: Days 102-131]
        G --> H[... more sequences]
    end
    
    subgraph "Training"
        H --> I[Shuffle & Split<br/>85% Train / 15% Val]
        I --> J[Model Training]
    end
```

### Normalization

Each stock is normalized independently using MinMax scaling:

```
normalized_price = (price - min_price) / (max_price - min_price)
```

**Example:**
```
AAPL prices: [150, 155, 160, 145, 170]
Min: 145, Max: 170

Normalized: [0.20, 0.40, 0.60, 0.00, 1.00]
```

### Sequence Building

```mermaid
flowchart LR
    subgraph "730 Days of Data"
        D1[Day 1]
        D2[Day 2]
        D100[Day 100]
        D101[Day 101]
        D130[Day 130]
        D131[Day 131]
        D730[Day 730]
    end
    
    subgraph "Sequence 1"
        S1I[Input: Days 1-100]
        S1T[Target: Days 101-130]
    end
    
    subgraph "Sequence 2"
        S2I[Input: Days 2-101]
        S2T[Target: Days 102-131]
    end
```

From 730 days, we generate approximately **600 sequences** per stock.

---

## Training Process

### Loss Function: Quantile Loss

The model uses **Pinball Loss** (Quantile Loss) to learn different percentiles:

```mermaid
flowchart LR
    subgraph "Quantile Loss"
        P[Prediction] --> E[Error = Actual - Predicted]
        E --> C{Error > 0?}
        C -->|Yes| U[q × Error]
        C -->|No| O["(q-1) × Error"]
    end
```

**Formula:**
```
L(y, ŷ, q) = max(q × (y - ŷ), (q - 1) × (y - ŷ))
```

For q = 0.5 (median):
- Penalizes over-predictions and under-predictions equally

For q = 0.9 (upper bound):
- Penalizes under-predictions more heavily
- Model learns to predict higher values

For q = 0.1 (lower bound):
- Penalizes over-predictions more heavily
- Model learns to predict lower values

### Training Loop

```mermaid
flowchart TB
    A[Initialize Model] --> B[Load Batch of 64 Sequences]
    B --> C[Forward Pass]
    C --> D[Compute Quantile Loss]
    D --> E[Backpropagation]
    E --> F[Update Weights]
    F --> G{More Batches?}
    G -->|Yes| B
    G -->|No| H[Validation]
    H --> I{Early Stopping?}
    I -->|No| B
    I -->|Yes| J[Save Best Model]
```

### Callbacks

1. **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus
2. **EarlyStopping**: Stops training if no improvement for 5 epochs

---

## Inference Process

### Prediction Flow

```mermaid
flowchart TB
    subgraph "Input"
        A[Ticker Symbol<br/>e.g., AAPL]
    end
    
    subgraph "Data Fetch"
        A --> B[Fetch 365 Days<br/>from yfinance]
        B --> C[Extract Last 100 Days]
    end
    
    subgraph "Preprocessing"
        C --> D[Fit MinMax Scaler<br/>on 365 days]
        D --> E[Normalize Last 100 Days]
        E --> F[Reshape to<br/>1 × 100 × 1]
    end
    
    subgraph "Model Inference"
        F --> G[Convert to TF Tensor]
        G --> H[Model Forward Pass]
        H --> I[Output: 1 × 30 × 3]
    end
    
    subgraph "Post-processing"
        I --> J[Inverse Transform<br/>to Real Prices]
        J --> K[Extract Quantiles]
    end
    
    subgraph "Output"
        K --> L[Lower Band q0.1]
        K --> M[Median q0.5]
        K --> N[Upper Band q0.9]
    end
```

### Why Per-Ticker Scaling at Inference?

- Different stocks have vastly different price ranges (AAPL ~$150 vs BRK-A ~$600,000)
- Training used per-ticker scaling, so inference must match
- Scaler is fitted on recent 365 days to capture current price range

---

## Worked Example

Let's walk through a complete prediction for **AAPL**.

### Step 1: Fetch Data

```
Fetched 365 days of AAPL closing prices
Latest prices: [..., 245.12, 246.50, 247.53]
Current price: $247.53
```

### Step 2: Normalize

```
Min price (365 days): $165.00
Max price (365 days): $260.00
Range: $95.00

Normalized current price: (247.53 - 165) / 95 = 0.869
```

### Step 3: Create Input Sequence

```
Last 100 normalized prices → Shape: (1, 100, 1)
[0.521, 0.534, 0.548, ..., 0.856, 0.869]
```

### Step 4: Model Prediction

```mermaid
flowchart LR
    subgraph "Input"
        I[100 Normalized Prices]
    end
    
    subgraph "Model"
        I --> GRU1[GRU 64]
        GRU1 --> ATT[Attention]
        ATT --> GRU2[GRU 64]
        GRU2 --> OUT[Dense 90]
    end
    
    subgraph "Raw Output"
        OUT --> R["[0.85, 0.87, 0.89,<br/>0.84, 0.86, 0.88,<br/>...]<br/>Shape: 30 × 3"]
    end
```

### Step 5: Inverse Transform

```
Raw prediction (Day 1): [0.82, 0.87, 0.92]

Inverse transform:
- Lower (q0.1): 0.82 × 95 + 165 = $242.90
- Median (q0.5): 0.87 × 95 + 165 = $247.65
- Upper (q0.9): 0.92 × 95 + 165 = $252.40
```

### Step 6: Final Output

```
{
  "ticker": "AAPL",
  "current_price": 247.53,
  "predicted_price": 247.65,      # Day 1 median
  "price_change": +0.12,
  "price_change_percent": +0.05%,
  "predictions": {
    "lower":  [242.90, 241.50, ...],  # 30 values
    "median": [247.65, 248.20, ...],  # 30 values
    "upper":  [252.40, 254.90, ...]   # 30 values
  }
}
```

---

## Quantile Predictions

### What Are Quantiles?

```mermaid
flowchart TB
    subgraph "Probability Distribution"
        A[All Possible<br/>Future Prices]
    end
    
    A --> Q1["q0.1 (10th percentile)<br/>90% chance price is ABOVE this"]
    A --> Q2["q0.5 (50th percentile)<br/>Median - best single estimate"]
    A --> Q3["q0.9 (90th percentile)<br/>90% chance price is BELOW this"]
```

### Interpretation

| Quantile | Meaning | Use Case |
|----------|---------|----------|
| q0.1 | Lower bound | Worst-case scenario |
| q0.5 | Median | Best point estimate |
| q0.9 | Upper bound | Best-case scenario |

### Visual Representation

```
Price ($)
   |
260|                    ╱───── Upper (q0.9)
   |                 ╱─╱
250|              ╱─╱    ───── Median (q0.5)
   |           ╱─╱
240|        ╱─╱──────────────── Lower (q0.1)
   |     ╱─╱
230|  ╱─╱
   |─╱
   └──────────────────────────────
   Today    Day 10    Day 20    Day 30
```

The shaded area between q0.1 and q0.9 represents the **80% confidence interval**.

---

## Model Limitations

1. **Only uses closing prices** - No volume, fundamentals, or news
2. **Assumes patterns repeat** - May fail during unprecedented events
3. **Global model** - Same model for all stocks; may underperform on unusual stocks
4. **30-day horizon** - Accuracy decreases for longer forecasts

---

## Summary

```mermaid
flowchart LR
    subgraph "GRUForecast"
        direction TB
        A[100 Days History] --> B[GRU + Attention]
        B --> C[30 Day Forecast]
        C --> D[With Uncertainty]
    end
    
    style A fill:#e0f2fe
    style B fill:#fef3c7
    style C fill:#d1fae5
    style D fill:#fce7f3
```

**Key Takeaways:**

1. GRU learns sequential patterns in price movements
2. Attention helps focus on relevant past events
3. Quantile loss provides uncertainty estimates
4. Per-ticker normalization handles different price scales
5. 80% confidence bands give a range of likely outcomes

---

*For implementation details, see the [Colab Notebook](Stock_Predictor_Colab.ipynb).*
