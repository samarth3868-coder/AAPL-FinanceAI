from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from tensorflow.keras.models import load_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


# -----------------------------------------------------------------------------
# App configuration and styling
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TICKER = "AAPL"
NEWS_LIMIT = 20

st.set_page_config(
    page_title="AAPL Finance AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1250px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero { padding: 1.5rem 1.75rem; border-radius: 18px; background: linear-gradient(135deg, #0f172a 0%, #172554 55%, #1d4ed8 100%); color: white; margin-bottom: 1.25rem; }
    .hero h1 { margin: 0; font-size: 2.35rem; letter-spacing: -0.04em; }
    .hero p { color: #cbd5e1; margin: .45rem 0 0; font-size: 1.02rem; }
    .eyebrow { color: #93c5fd; text-transform: uppercase; font-size: .75rem; font-weight: 700; letter-spacing: .12em; }
    .pill { display: inline-block; padding: .3rem .7rem; border-radius: 999px; background: #dbeafe; color: #1d4ed8; font-weight: 700; font-size: .82rem; }
    .muted { color: #64748b; font-size: .88rem; }
    div[data-testid="stMetric"] { background: #f8fafc; border: 1px solid #e2e8f0; padding: .9rem 1rem; border-radius: 14px; }
    .disclaimer { padding: .9rem 1rem; border-left: 4px solid #f59e0b; background: #fffbeb; color: #713f12; border-radius: 8px; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached resources and data access
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(BASE_DIR / "aapl_lstm_news_model.keras")
    scaler = joblib.load(BASE_DIR / "aapl_scaler.pkl")
    features = joblib.load(BASE_DIR / "aapl_features.pkl")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    sentiment_model.eval()
    return model, scaler, features, tokenizer, sentiment_model


@st.cache_data(ttl=900, show_spinner=False)
def get_price_data(period="6mo"):
    data = yf.download(TICKER, period=period, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError("No market data was returned by Yahoo Finance.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data["Daily_Return"] = data["Close"].pct_change()
    data["Return_5D"] = data["Close"].pct_change(5)
    data["MA_10"] = data["Close"].rolling(10).mean()
    data["MA_20"] = data["Close"].rolling(20).mean()
    data["Volatility_20D"] = data["Daily_Return"].rolling(20).std()
    return data.dropna(subset=["Close"]).copy()


@st.cache_data(ttl=3600, show_spinner=False)
def get_news(api_key):
    if not api_key:
        return []
    response = requests.get(
        "https://www.alphavantage.co/query",
        params={"function": "NEWS_SENTIMENT", "tickers": TICKER, "sort": "LATEST", "limit": NEWS_LIMIT, "apikey": api_key},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("feed", [])


@st.cache_data(ttl=3600, show_spinner=False)
def score_news(articles):
    if not articles:
        return [], 0.0
    _, _, _, tokenizer, sentiment_model = load_artifacts()
    labels, scores = [], []
    label_score = {"positive": 1, "neutral": 0, "negative": -1}
    for article in articles:
        text = f"{article.get('title', '')}. {article.get('summary', '')}".strip()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            probabilities = torch.softmax(sentiment_model(**inputs).logits, dim=1)[0]
        index = int(probabilities.argmax())
        label = sentiment_model.config.id2label[index].lower()
        confidence = float(probabilities[index])
        labels.append({"label": label, "confidence": confidence})
        scores.append(label_score.get(label, 0))
    return labels, float(np.mean(scores)) if scores else 0.0


def sentiment_label(score):
    if score > 0.1:
        return "Positive", "🟢"
    if score < -0.1:
        return "Negative", "🔴"
    return "Neutral", "🟡"


def fmt_pct(value):
    return "—" if pd.isna(value) else f"{value * 100:+.2f}%"


# -----------------------------------------------------------------------------
# Load data and calculate the live signal
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Dashboard controls")
    st.caption("Data refreshes automatically. Cached results reduce API calls and startup time.")
    period = st.selectbox("Price history", ["3mo", "6mo", "1y"], index=1)
    st.divider()
    st.markdown("**Model stack**")
    st.caption("LSTM + FinBERT\nTechnical indicators + news sentiment")
    st.divider()
    st.caption("Educational research tool — not investment advice.")

st.markdown(
    '<div class="hero"><div class="eyebrow">AI-powered market intelligence</div><h1>AAPL Finance AI</h1><p>Combining technical signals and financial news sentiment to estimate next-day direction.</p></div>',
    unsafe_allow_html=True,
)

try:
    with st.spinner("Loading market data and model signal..."):
        model, scaler, features, _, _ = load_artifacts()
        data = get_price_data(period)
        api_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
        articles = get_news(api_key)
        article_scores, overall_sentiment = score_news(tuple(articles))
except Exception as exc:
    st.error("The dashboard could not load its live data.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

# Merge historical sentiment for the LSTM feature set.
sentiment_path = BASE_DIR / "aapl_daily_sentiment.csv"
daily_sentiment = pd.read_csv(sentiment_path)
daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"]).dt.date
data["date"] = data["Date"].dt.date
data = data.merge(daily_sentiment, on="date", how="left")
data["sentiment"] = data["sentiment"].fillna(0.0)
data.loc[data.index[-1], "sentiment"] = overall_sentiment

prediction_probability = None
prediction = "Unavailable"
model_input = data[list(features)].dropna()
if len(model_input) >= 60:
    scaled = scaler.transform(model_input.tail(60))
    prediction_probability = float(model.predict(np.expand_dims(scaled, axis=0), verbose=0)[0][0])
    prediction = "UP" if prediction_probability >= 0.5 else "DOWN"

latest = data.iloc[-1]
current_price = float(latest["Close"])
daily_return = float(latest["Daily_Return"])
return_5d = float(latest["Return_5D"])
ma_10 = float(latest["MA_10"])
ma_20 = float(latest["MA_20"])
volatility = float(latest["Volatility_20D"])
trend = "Bullish" if current_price > ma_10 and current_price > ma_20 else "Bearish" if current_price < ma_10 and current_price < ma_20 else "Mixed"
news_label, news_icon = sentiment_label(overall_sentiment)
last_updated = latest["Date"].strftime("%d %b %Y")

# -----------------------------------------------------------------------------
# Dashboard sections
# -----------------------------------------------------------------------------
st.markdown(f'<span class="pill">{TICKER} · Last market data: {last_updated}</span>', unsafe_allow_html=True)
st.markdown("### Market snapshot")
metrics = st.columns(4)
metrics[0].metric("Current price", f"${current_price:,.2f}", fmt_pct(daily_return))
metrics[1].metric("5-day return", fmt_pct(return_5d))
metrics[2].metric("Trend", trend)
metrics[3].metric("20-day volatility", f"{volatility * 100:.2f}%")

left, right = st.columns([1.4, 1])
with left:
    st.markdown("### Price momentum")
    chart = data.set_index("Date")[["Close", "MA_10", "MA_20"]].rename(columns={"Close": "Price", "MA_10": "10-day MA", "MA_20": "20-day MA"})
    st.line_chart(chart, height=340)
with right:
    st.markdown("### Next-day model signal")
    if prediction == "UP":
        st.success("▲ UP", icon="📈")
    elif prediction == "DOWN":
        st.error("▼ DOWN", icon="📉")
    else:
        st.warning("Signal unavailable", icon="⚠️")
    if prediction_probability is not None:
        confidence = prediction_probability if prediction == "UP" else 1 - prediction_probability
        st.metric("Model confidence", f"{confidence * 100:.1f}%")
        st.progress(confidence)
    st.markdown(f"**News sentiment:** {news_icon} {news_label}")
    st.caption(f"Based on {len(articles)} recent AAPL articles. Sentiment index: {overall_sentiment:+.2f}")
    st.markdown("<div class='disclaimer'>This is an experimental model output, not a recommendation to buy or sell securities.</div>", unsafe_allow_html=True)

news_tab, methodology_tab = st.tabs(["Latest news", "Methodology"])
with news_tab:
    if not articles:
        if not api_key:
            st.info("Add ALPHA_VANTAGE_API_KEY to Streamlit secrets to enable live news sentiment.")
        else:
            st.info("No recent AAPL news was returned.")
    for article, scored in zip(articles[:8], article_scores[:8]):
        title = article.get("title", "Untitled article")
        url = article.get("url", "#")
        source = article.get("source", "Unknown source")
        label = scored["label"].title()
        st.markdown(f"**[{title}]({url})**")
        st.caption(f"{source} · {label} sentiment · {scored['confidence'] * 100:.0f}% confidence")
        st.divider()

with methodology_tab:
    st.markdown("""
    The application combines **10-day and 20-day moving averages**, daily and 5-day returns, 20-day volatility, and a daily news-sentiment feature. Financial headlines are scored with **ProsusAI/FinBERT**, then the most recent 60 trading days are scaled and passed to the trained **LSTM** model.

    The displayed UP/DOWN signal represents the model's estimated direction for the next trading day. It is intended for experimentation and portfolio-research education; it has not been presented as a guarantee of future performance.
    """)
    st.info("Data sources: Yahoo Finance for market data and Alpha Vantage for news sentiment.")

st.caption(f"Refreshed {datetime.now(timezone.utc).strftime('%d %b %Y at %H:%M UTC')} · Built with Streamlit, TensorFlow, PyTorch, and Hugging Face Transformers")
