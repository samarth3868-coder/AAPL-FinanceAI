# AAPL Market Intelligence

### AI-Powered Stock Direction Prediction

</p>

An end-to-end financial machine learning application that combines **technical indicators** and **financial news sentiment** using **LSTM + FinBERT** to estimate the next-day direction of Apple (AAPL) stock.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow)](https://www.tensorflow.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-FinBERT-red?logo=pytorch)](https://pytorch.org/)[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface)](https://huggingface.co/)[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)[![yfinance](https://img.shields.io/badge/Data-yfinance-green)](https://github.com/ranaroussi/yfinance)[![Alpha Vantage](https://img.shields.io/badge/News-Alpha%20Vantage-purple)](https://www.alphavantage.co/)


## Live Demo

[**Open AAPL Finance AI**](https://aapl-financeai-gzgjdzvbvm3segt8ksw6qq.streamlit.app/)

## Overview

AAPL Finance AI combines market price information with financial news sentiment in an interactive Streamlit application. It brings together historical AAPL market data, technical indicators, financial news, FinBERT sentiment analysis, and LSTM sequence modeling to estimate the next-day direction of AAPL stock.

## Dashboard

The dashboard displays:

- Current AAPL price and daily return

- Five-day return and moving averages

- Twenty-day volatility

- Financial-news sentiment

- AAPL price history chart

- Latest news headlines

- Next-day UP/DOWN prediction
<p align="center">
<img src="P1.png" alt="AAPL Market Intelligence dashboard overview" width="100%">
<p align="center">
<img src="P2.png" alt="Latest AAPL news and next-day prediction" width="100%">
</p>

## Model Architecture

<p align="center">
  <img src="model-architecture.png" alt="AAPL Finance AI model architecture" width="100%">
</p>

## Technology stack

| Technology | Role in the project |
| --- | --- |
| **Python** | Application logic and data processing |
| **Streamlit** | Interactive dashboard and user interface |
| **yfinance** | Historical AAPL market data |
| **Alpha Vantage** | Financial-news data |
| **FinBERT** | Finance-specific sentiment analysis |
| **TensorFlow / Keras** | LSTM direction model |
| **PyTorch + Transformers** | Sentiment-model inference |

## Run locally

Follow these steps to run the dashboard locally.

### 1. Clone the repository

```bash
git clone https://github.com/samarth3868-coder/AAPL-FinanceAI.git
cd AAPL-FinanceAI
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 3. Configure Alpha Vantage

Create `.streamlit/secrets.toml`:

```
ALPHA_VANTAGE_API_KEY = "your_api_key_here"
```

### 4. Run the application

```bash
streamlit run app.py
```

## Repository files

| File | Description |
| --- | --- |
| `app.py` | Streamlit dashboard and prediction pipeline |
| `aapl_lstm_news_model.keras` | Trained LSTM model |
| `aapl_scaler.pkl` | Saved feature scaler |
| `aapl_features.pkl` | Saved model feature list |
| `aapl_daily_sentiment.csv` | Daily sentiment data |
| `P1.png` | Dashboard overview screenshot |
| `P2.png` | News and prediction screenshot |
| `model-architecture.png` | Model workflow and architecture diagram |
| `requirements.txt` | Python dependencies |

## Limitations

This project is designed for education and experimentation. It does not guarantee prediction accuracy, provide financial advice, or replace independent research. Market and news data may be delayed or temporarily unavailable. The model is trained for AAPL and should not automatically be applied to other securities.
