import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.title("NSE Pro Stock Scanner")
st.write("Find stocks likely to rise")

# List of major NSE stocks
stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "ITC.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "BHARTIARTL.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS"
]

results = []

if st.button("Scan Stocks"):
    for stock in stocks:
        try:
            data = yf.download(stock, period="2y", progress=False)

            data['Return'] = data['Close'].pct_change()
            data['MA10'] = data['Close'].rolling(10).mean()
            data['MA50'] = data['Close'].rolling(50).mean()
            data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

            data = data.dropna()

            X = data[['Return', 'MA10', 'MA50']]
            y = data['Target']

            model = RandomForestClassifier(n_estimators=100)
            model.fit(X, y)

            latest = X.iloc[-1].values.reshape(1, -1)
            prob = model.predict_proba(latest)[0][1]

            results.append((stock, round(prob * 100, 2)))

        except:
            pass

    results = sorted(results, key=lambda x: x[1], reverse=True)

    st.subheader("Top Stocks Likely to Rise")

    for stock, prob in results[:5]:
        st.write(f"{stock} — {prob}% probability")
