# AAPL Finance AI

### AI-Powered Stock Direction Prediction

An end-to-end financial machine learning application that combines **technical indicators** and **financial news sentiment** using **LSTM + FinBERT** to predict the next-day direction of Apple (AAPL) stock.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-FinBERT-red?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![yfinance](https://img.shields.io/badge/Data-yfinance-green)](https://github.com/ranaroussi/yfinance)
[![Alpha Vantage](https://img.shields.io/badge/News-Alpha%20Vantage-purple)](https://www.alphavantage.co/)

---

## Live Demo

🚀 **[Open AAPL Finance AI](https://aapl-financeai-gzgjdzvbvm3segt8ksw6qq.streamlit.app/)**

---

## Overview

**AAPL Finance AI** is an experimental financial machine learning application designed to combine **market price information** with **financial news sentiment**.

The project uses:

- Historical AAPL market data
- Technical indicators
- Financial news
- FinBERT sentiment analysis
- LSTM sequence modeling
- Streamlit for interactive visualization

The objective is to predict the **next-day direction** of AAPL stock:

```text
0 → DOWN
1 → UP
