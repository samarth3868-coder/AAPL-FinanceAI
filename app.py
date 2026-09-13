import streamlit as st
import joblib
import pandas as pd
import requests
import yfinance as yf
import torch
from datetime import datetime

from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = load_model("aapl_lstm_news_model.keras")

scaler = joblib.load("aapl_scaler.pkl")

features = joblib.load("aapl_features.pkl")

st.title("AAPL Market Intelligence")

st.button("Refresh Data")

data = yf.download(tickers="AAPL",period="6mo")

data.columns = data.columns.get_level_values(0)

data["Daily_Return"] = data["Close"].pct_change()

data["Return_5D"] = data["Close"].pct_change(5)

data["MA_10"] = data["Close"].rolling(10).mean()

data["MA_20"] = data["Close"].rolling(20).mean()

data["Volatility_20D"] = data["Daily_Return"].rolling(20).std()

API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"

@st.cache_data(ttl=3600)
def get_news():

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "AAPL",
        "sort": "LATEST",
        "limit": 10,
        "apikey": API_KEY
    }

    response = requests.get(
        "https://www.alphavantage.co/query",
        params=params
    )

    return response.json()

news_data = get_news()


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment(text):
    inputs = tokenizer(text,return_tensors="pt", truncation=True)

    outputs = finbert_model(**inputs)

    probabilities = torch.softmax(outputs.logits,dim=1)

    predicted_class = probabilities.argmax(dim=1)

    sentiment = finbert_model.config.id2label[predicted_class.item()]

    confidence = probabilities[0, predicted_class.item()]

    return sentiment, confidence

news_data = get_news()
articles = news_data["feed"]

sentiments = []

for article in articles:

    text = article["title"] + ". " + article["summary"]

    sentiment, confidence = get_sentiment(text)

    sentiments.append(sentiment)

sentiment_score = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

scores = [
    sentiment_score[sentiment]
    for sentiment in sentiments
]

overall_sentiment = sum(scores) / len(scores)

# ===== MARKET OVERVIEW =====

st.subheader("Market Overview")

latest = data.iloc[-1]

current_price = latest["Close"]
daily_return = latest["Daily_Return"]
return_5d = latest["Return_5D"]
ma_10 = latest["MA_10"]
ma_20 = latest["MA_20"]
volatility = latest["Volatility_20D"]

col1, col2 = st.columns(2)

with col1:
    st.metric("Current Price", f"${current_price:.2f}")

with col2:
    st.metric("Daily Return", f"{daily_return * 100:.2f}%")

col3, col4 = st.columns(2)

with col3:
    st.metric("5-Day Return", f"{return_5d * 100:.2f}%")

with col4:
    st.metric("MA-10", f"${ma_10:.2f}")

col5, col6 = st.columns(2)

with col5:
    st.metric("MA-20", f"${ma_20:.2f}")

with col6:
    st.metric("20-Day Volatility", f"{volatility:.4f}")


st.subheader("News Sentiment")

if overall_sentiment > 0.1:
    sentiment_label = "Positive"
elif overall_sentiment < -0.1:
    sentiment_label = "Negative"
else:
    sentiment_label = "Neutral"

st.write(f"Market sentiment: **{sentiment_label}**")


st.subheader("AAPL Price Chart")

chart_data = data[["Close"]].copy()

st.line_chart(chart_data)


st.subheader("Latest News")

for article in articles[:5]:
    st.markdown(
        f"- [{article['title']}]({article['url']})"
    )
    st.caption(article["source"])

daily_sentiment = pd.read_csv("aapl_daily_sentiment.csv")

daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"]).dt.date

data = data.reset_index()

data["date"] = pd.to_datetime(data["Date"]).dt.date

data = data.merge(
    daily_sentiment,
    on="date",
    how="left")

data["sentiment"] = data["sentiment"].fillna(0)

data.loc[data.index[-1], "sentiment"] = overall_sentiment

X_live = data[features].copy()

X_live = X_live.dropna()

last_60 = X_live.tail(60)

scaled_60 = scaler.transform(last_60)

import numpy as np

input_data = np.expand_dims(scaled_60,axis=0)

prediction_probability = model.predict(input_data)[0][0]

if prediction_probability >= 0.5:
    prediction = "UP"
else:
    prediction = "DOWN"

st.subheader("Next-Day Prediction")

if prediction == "UP":
    st.success("UP")
else:
    st.error("DOWN")


if current_price > ma_10 and current_price > ma_20:
    trend = "Bullish"
elif current_price < ma_10 and current_price < ma_20:
    trend = "Bearish"
else:
    trend = "Mixed"





with st.expander("How does this prediction work?"):
    st.write("""
    The model combines technical indicators with financial
    news sentiment and uses the last 60 trading days to
    predict the next-day direction.
    """)