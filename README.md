# AAPL Finance AI

### AI-powered market intelligence for Apple stock

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-FinBERT-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Educational-lightgrey)](#limitations)

AAPL Finance AI is an interactive Streamlit dashboard that combines **technical market indicators**, **financial-news sentiment**, and a trained **LSTM model** to estimate the next-day direction of Apple (AAPL) stock.

> **Educational project:** Model outputs are experimental signals, not investment advice or a guarantee of future returns.

## Live demo

**[Open the AAPL Finance AI dashboard](https://aapl-financeai-gzgjdzvbvm3segt8ksw6qq.streamlit.app/)**

## What the dashboard shows

- Current AAPL price, daily return, five-day return, trend, and 20-day volatility.
- Interactive price chart with 10-day and 20-day moving averages.
- Next-day UP/DOWN model signal with estimated confidence.
- Recent AAPL headlines scored with ProsusAI/FinBERT.
- A methodology panel explaining the feature engineering and model workflow.
- Defensive handling for missing market data, unavailable news, and incomplete model windows.

## How it works

```text
Yahoo Finance ──► Price history ──► Technical indicators ──┐
                                                           ├─► Scaled 60-day feature window ──► LSTM signal
Alpha Vantage ─► AAPL news ───────► FinBERT sentiment ─────┘
```

The live feature set includes daily returns, five-day returns, 10-day and 20-day moving averages, 20-day volatility, and a daily news-sentiment value. The most recent 60 trading days are scaled with the saved preprocessing object and passed to the trained Keras model.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard, data loading, feature engineering, sentiment scoring, and inference |
| `aapl_lstm_news_model.keras` | Trained LSTM model artifact |
| `aapl_scaler.pkl` | Saved feature scaler used during model training |
| `aapl_features.pkl` | Feature order expected by the model |
| `aapl_daily_sentiment.csv` | Historical daily sentiment feature used during inference |
| `requirements.txt` | Python dependencies |

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/samarth3868-coder/AAPL-FinanceAI.git
cd AAPL-FinanceAI
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 3. Configure Alpha Vantage

Create `.streamlit/secrets.toml`:

```toml
ALPHA_VANTAGE_API_KEY = "your_api_key_here"
```

The dashboard still loads market data when the news key is absent, but live news sentiment will be unavailable.

### 4. Start the app

```bash
streamlit run app.py
```

## Technology stack

- **Streamlit** for the interactive application layer.
- **yfinance** for historical AAPL market data.
- **Alpha Vantage** for recent financial-news headlines.
- **ProsusAI/FinBERT** for finance-specific sentiment classification.
- **TensorFlow/Keras** for the saved LSTM direction model.
- **PyTorch** and **Hugging Face Transformers** for sentiment inference.

## Limitations

This project is designed for learning and experimentation. It does not perform backtesting in the dashboard, provide a calibrated probability of returns, account for transaction costs, or guarantee accuracy. Market data and news APIs can be delayed, rate-limited, or temporarily unavailable. The model is trained for AAPL and should not be assumed to generalize to other securities without retraining and validation.

## Responsible use

Use the dashboard to explore how alternative data and machine learning can be combined in a financial workflow. Validate any hypothesis with out-of-sample testing, proper baselines, risk controls, and independent research before making decisions.

## Author

Built by **Samarth** as an applied machine-learning project connecting time-series modeling, NLP, financial data, and product design.
