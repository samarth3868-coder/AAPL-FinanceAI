# AAPL Finance AI

An AI-powered financial dashboard that predicts the next-day direction of Apple (AAPL) stock using historical market data, technical indicators, and financial news sentiment.

## Project Overview

This project combines two types of financial signals:

- Historical stock market data
- Financial news sentiment

The system uses an LSTM neural network to predict whether AAPL is likely to move **UP or DOWN** on the next trading day.

## Features

- Live AAPL market data using yfinance
- Daily return calculation
- 5-day return
- 10-day moving average
- 20-day moving average
- 20-day volatility
- Financial news from Alpha Vantage
- Financial sentiment analysis using FinBERT
- LSTM-based next-day direction prediction
- Interactive Streamlit dashboard
- Latest financial news with clickable links
- AAPL price chart

## Machine Learning Pipeline

```text
AAPL Market Data
       ↓
Technical Indicators
       ↓
Financial News
       ↓
FinBERT Sentiment
       ↓
Feature Combination
       ↓
60-Day Sequence
       ↓
StandardScaler
       ↓
LSTM Model
       ↓
Next-Day Direction
       ↓
UP / DOWN


Features Used by the Model
Close Price
Volume
Daily Return
5-Day Return
10-Day Moving Average
20-Day Moving Average
20-Day Volatility
News Sentiment


Technologies Used
Python
Pandas
NumPy
Scikit-learn
TensorFlow / Keras
LSTM
PyTorch
Hugging Face Transformers
FinBERT
yfinance
Alpha Vantage API
Streamlit

Model

The project uses an LSTM neural network to process the last 60 trading days of market and sentiment features.

The model receives:
60 days × 8 features

and produces a binary next-day direction prediction:

0 → DOWN
1 → UP

Dashboard

The Streamlit dashboard displays:

Current AAPL price
Daily return
5-day return
MA-10
MA-20
20-day volatility
Overall news sentiment
Technical trend
AAPL price chart
Latest financial news
Next-day UP/DOWN prediction
Prediction methodology

How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run the Streamlit application
streamlit run app.py
Project Structure
Finance_AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── aapl_lstm_news_model.keras
├── aapl_scaler.pkl
├── aapl_features.pkl
└── aapl_daily_sentiment.csv
Data Sources
Market Data: Yahoo Finance through yfinance
Financial News: Alpha Vantage News Sentiment API
News Sentiment: FinBERT (ProsusAI/finbert)
Limitations

The model predicts stock direction as an experimental machine-learning exercise.

Financial markets are highly unpredictable, and the prediction should not be considered financial advice or a guaranteed forecast.

The current model may have limited generalization performance and should not be used for real-world trading decisions.

Future Improvements
Improve historical news coverage
Experiment with additional financial features
Improve model generalization
Add support for multiple stocks
Add portfolio-level analysis
Improve API security
Improve model evaluation and backtesting