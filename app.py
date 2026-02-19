import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="India Stock News & Prediction", layout="wide")

st.title("India Stock News & Intelligence")

# -------- NEWS SECTION --------
st.header("Latest Market News")

API_KEY = "PASTE_YOUR_NEWSAPI_KEY"

url = f"https://newsapi.org/v2/everything?q=indian stock market OR NSE OR Sensex&language=en&sortBy=publishedAt&apiKey={API_KEY}"

try:
    news = requests.get(url).json()
    articles = news["articles"][:5]

    for a in articles:
        st.subheader(a["title"])
        st.write(a["source"]["name"])
        st.write(a["description"])
        st.write(a["url"])
        st.write("---")

except:
    st.write("News not available")

# -------- STOCK SCANNER --------
st.header("Stocks Likely to Rise")

stocks = [
"RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
"ITC.NS","SBIN.NS","LT.NS","AXISBANK.NS","KOTAKBANK.NS"
]

results = []

if st.button("Run Market Scan"):
    for stock in stocks:
        try:
            data = yf.download(stock, period="1y", progress=False)

            data['Return'] = data['Close'].pct_change()
            data['MA10'] = data['Close'].rolling(10).mean()
            data['MA50'] = data['Close'].rolling(50).mean()
            data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

            data = data.dropna()

            X = data[['Return','MA10','MA50']]
            y = data['Target']

            model = RandomForestClassifier(n_estimators=100)
            model.fit(X, y)

            latest = X.iloc[-1].values.reshape(1, -1)
            prob = model.predict_proba(latest)[0][1]

            signal = "BUY" if prob > 0.6 else "HOLD" if prob > 0.5 else "AVOID"

            results.append((stock, round(prob*100,2), signal))

        except:
            pass

    results = sorted(results, key=lambda x: x[1], reverse=True)

    st.subheader("Top Opportunities")
    for r in results:
        st.write(f"{r[0]} | {r[1]}% | {r[2]}")

# -------- SINGLE STOCK --------
st.header("Analyze Any Stock")

symbol = st.text_input("Enter Stock (Example: RELIANCE.NS)")

if symbol:
    data = yf.download(symbol, period="6mo")
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()

    st.line_chart(data[['Close','MA50','MA200']])
