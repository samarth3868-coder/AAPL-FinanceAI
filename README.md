# AAPL Finance AI

### AI-Powered Stock Direction Prediction

An end-to-end financial machine learning application that combines **technical indicators** and **financial news sentiment** using **LSTM + FinBERT** to predict the next-day direction of Apple (AAPL) stock.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-FinBERT-red?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational-lightgrey)](#disclaimer)

## Live Demo

🚀 **[Open AAPL Finance AI](https://aapl-financeai-gzgjdzvbvm3segt8ksw6qq.streamlit.app/)**

## What It Does

The application:

- Fetches AAPL market data using `yfinance`
- Calculates technical indicators
- Retrieves financial news using Alpha Vantage
- Analyzes financial sentiment with FinBERT
- Combines 8 features over a 60-day window
- Uses an LSTM model to predict next-day direction
- Presents the results through an interactive Streamlit dashboard
