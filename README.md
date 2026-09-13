````markdown
# AAPL Finance AI

### AI-Powered Stock Direction Prediction

An end-to-end financial machine learning application that combines technical indicators and financial news sentiment using LSTM + FinBERT to predict the next-day direction of Apple (AAPL) stock.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-FinBERT-red?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![yfinance](https://img.shields.io/badge/Data-yfinance-green)](https://github.com/ranaroussi/yfinance)
[![Alpha Vantage](https://img.shields.io/badge/News-Alpha%20Vantage-purple)](https://www.alphavantage.co/)

---

## Live Demo

[Open AAPL Finance AI](https://aapl-financeai-erbvgxbwhbnky5sepasruq.streamlit.app/)

---

## Overview

AAPL Finance AI is an experimental financial machine learning application that combines market data, technical indicators, and financial news sentiment to predict the next-day direction of Apple (AAPL) stock.

The project demonstrates how time-series deep learning and NLP can be combined in a financial application.

The system predicts:

- `0` → DOWN
- `1` → UP

---

## Key Features

### Market Analysis

- Current AAPL price
- Daily return
- 5-day return
- 10-day moving average
- 20-day moving average
- 20-day volatility
- Technical trend classification

### Financial News Analysis

- Financial news retrieved using Alpha Vantage
- AAPL-associated news filtering
- Financial sentiment analysis using FinBERT
- Numerical sentiment scoring

Sentiment mapping:

- Positive → `+1`
- Neutral → `0`
- Negative → `-1`

### Machine Learning Prediction

- LSTM-based sequence model
- 60 trading-day input window
- 8 model features
- Next-day direction prediction
- Streamlit dashboard for visualization

### Dashboard

- Market overview
- News sentiment
- AAPL price chart
- Latest financial news
- Technical trend
- Next-day prediction
- Prediction methodology
- Disclaimer

---

## Machine Learning Pipeline

```text
AAPL Market Data
        ↓
Technical Indicators
        ↓
Financial News
        ↓
FinBERT Sentiment Analysis
        ↓
Combine Market + Sentiment Features
        ↓
Latest 60 Trading Days
        ↓
StandardScaler
        ↓
LSTM Model
        ↓
Next-Day Direction
        ↓
UP / DOWN
````

---

## Features Used by the Model

The LSTM model uses 8 features:

1. Close Price
2. Volume
3. Daily Return
4. 5-Day Return
5. 10-Day Moving Average
6. 20-Day Moving Average
7. 20-Day Volatility
8. News Sentiment

The model input shape is:

```text
60 trading days × 8 features
```

Input tensor:

```text
(1, 60, 8)
```

Where:

* `1` = one prediction sample
* `60` = 60 trading days
* `8` = input features

---

## Technical Indicators

### Close Price

Daily closing price of AAPL.

### Volume

Daily trading volume.

### Daily Return

```python
Daily_Return = Close.pct_change()
```

### 5-Day Return

```python
Return_5D = Close.pct_change(5)
```

### 10-Day Moving Average

```python
MA_10 = Close.rolling(10).mean()
```

### 20-Day Moving Average

```python
MA_20 = Close.rolling(20).mean()
```

### 20-Day Volatility

```python
Volatility_20D = Daily_Return.rolling(20).std()
```

---

## Financial News Sentiment

The project uses Alpha Vantage to retrieve financial news associated with AAPL.

Each article is processed using:

**ProsusAI/FinBERT**

The sentiment classification produces:

```text
Positive
Neutral
Negative
```

These are converted into numerical scores:

```text
Positive → +1
Neutral  →  0
Negative → -1
```

The scores are then averaged to create an overall news sentiment signal.

---

## LSTM Model

The project uses a Long Short-Term Memory (LSTM) neural network to process sequential market data.

Each prediction uses the latest 60 trading days.

```text
60 Days
   ×
8 Features
   ↓
StandardScaler
   ↓
LSTM
   ↓
Binary Classification
   ↓
UP / DOWN
```

The LSTM receives both market and sentiment information, allowing the model to incorporate multiple types of financial signals.

---

## Dashboard

### Market Overview

The dashboard displays:

```text
Current Price
Daily Return
5-Day Return
MA-10
MA-20
20-Day Volatility
```

### News Sentiment

Displays:

```text
Overall Sentiment
Positive / Neutral / Negative
```

### Price Chart

Displays recent AAPL closing-price movement.

### Latest News

Displays available AAPL-associated financial news with clickable article links.

### Next-Day Prediction

Displays:

```text
UP
```

or:

```text
DOWN
```

The dashboard also displays:

```text
Technical Trend
News Sentiment
Model Signal
```

---

## System Architecture

```text
                    ┌────────────────────┐
                    │     AAPL Data      │
                    │      yfinance      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Technical Features │
                    └─────────┬──────────┘
                              │
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼
┌────────────────────┐                 ┌────────────────────┐
│  Alpha Vantage     │                 │ Historical Market  │
│   Financial News   │                 │     Features       │
└─────────┬──────────┘                 └─────────┬──────────┘
          │                                      │
          ▼                                      │
┌────────────────────┐                           │
│      FinBERT       │                           │
│ Sentiment Analysis │                           │
└─────────┬──────────┘                           │
          │                                      │
          ▼                                      │
┌────────────────────┐                           │
│ News Sentiment     │                           │
│     -1 / 0 / +1    │                           │
└─────────┬──────────┘                           │
          │                                      │
          └─────────────────┬────────────────────┘
                            ▼
                   ┌────────────────────┐
                   │ 8 Combined         │
                   │     Features       │
                   └─────────┬──────────┘
                             ▼
                   ┌────────────────────┐
                   │ Latest 60 Days     │
                   └─────────┬──────────┘
                             ▼
                   ┌────────────────────┐
                   │  StandardScaler    │
                   └─────────┬──────────┘
                             ▼
                   ┌────────────────────┐
                   │    LSTM Model      │
                   └─────────┬──────────┘
                             ▼
                   ┌────────────────────┐
                   │ Next-Day Direction │
                   │      UP / DOWN     │
                   └────────────────────┘
```

---

## Technologies Used

### Programming

* Python

### Data Processing

* Pandas
* NumPy
* Joblib

### Machine Learning

* TensorFlow
* Keras
* LSTM
* Scikit-learn
* StandardScaler

### NLP

* PyTorch
* Hugging Face Transformers
* FinBERT

### Data Sources

* yfinance
* Alpha Vantage API

### Deployment

* Streamlit
* Streamlit Community Cloud

---

## Project Structure

```text
AAPL-FinanceAI/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── aapl_lstm_news_model.keras
├── aapl_scaler.pkl
├── aapl_features.pkl
└── aapl_daily_sentiment.csv
```

### File Description

| File                         | Description                                   |
| ---------------------------- | --------------------------------------------- |
| `app.py`                     | Streamlit application and prediction pipeline |
| `requirements.txt`           | Python dependencies                           |
| `aapl_lstm_news_model.keras` | Trained LSTM model                            |
| `aapl_scaler.pkl`            | Saved feature scaler                          |
| `aapl_features.pkl`          | Saved model feature order                     |
| `aapl_daily_sentiment.csv`   | Historical daily sentiment data               |
| `README.md`                  | Project documentation                         |
| `.gitignore`                 | Git ignored files                             |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/samarth3868-coder/AAPL-FinanceAI.git
```

Move into the project directory:

```bash
cd AAPL-FinanceAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Requirements

The project uses:

```text
streamlit
pandas
numpy
requests
yfinance
joblib
scikit-learn
tensorflow
torch
transformers
torchvision
```

The deployed application uses Python 3.12.

---

## Data Sources

### Market Data

Yahoo Finance through `yfinance`.

### Financial News

Alpha Vantage News Sentiment API.

### Sentiment Model

ProsusAI/FinBERT.

---

## Model Workflow

```text
1. Download recent AAPL market data
2. Calculate technical indicators
3. Retrieve financial news
4. Identify AAPL-associated articles
5. Analyze news using FinBERT
6. Convert sentiment to numerical values
7. Combine market and sentiment features
8. Select the latest 60 trading days
9. Scale the features using the saved scaler
10. Pass the sequence to the LSTM
11. Generate next-day UP/DOWN prediction
12. Display results in Streamlit
```

---

## Current Limitations

This project is primarily an educational and experimental financial machine learning application.

### Model Generalization

Financial markets are highly noisy and difficult to predict. Historical relationships may not remain stable in future market conditions.

### Directional Prediction

The model predicts only:

```text
UP / DOWN
```

It does not predict the exact future price.

### News Availability

News availability depends on external API availability and provider coverage.

### Sentiment Representation

Complex financial language is simplified into a numerical sentiment score:

```text
-1 / 0 / +1
```

### API Limitations

External APIs can have usage limits, availability restrictions, and changing data coverage.

### Experimental Nature

The model output should not be interpreted as a guaranteed market forecast or trading recommendation.

---

## Future Improvements

* Expand the project to multiple stocks
* Improve historical news coverage
* Improve news relevance filtering
* Add fundamental financial data
* Add richer financial features
* Improve model evaluation
* Perform walk-forward validation
* Add systematic backtesting
* Compare LSTM with alternative models
* Experiment with transformer-based time-series models
* Add portfolio-level analysis
* Improve API security
* Add automated model retraining

---

## Learning Outcomes

This project covers an end-to-end machine learning workflow:

```text
Data Collection
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Time-Series Preparation
        ↓
NLP
        ↓
Sentiment Analysis
        ↓
Deep Learning
        ↓
Model Inference
        ↓
Dashboard Development
        ↓
Cloud Deployment
```

Key concepts implemented:

* Machine Learning
* Deep Learning
* NLP
* Sentiment Analysis
* Time-Series Modeling
* Financial Data Analysis
* Feature Engineering
* API Integration
* Data Visualization
* Model Deployment


```
```
