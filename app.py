import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.title("NSE Legend Stock Intelligence")

# Major NSE Stocks (Top 50)
stocks = [
"RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
"ITC.NS","SBIN.NS","LT.NS","AXISBANK.NS","KOTAKBANK.NS",
"BHARTIARTL.NS","ASIANPAINT.NS","MARUTI.NS","HCLTECH.NS",
"SUNPHARMA.NS","TITAN.NS","WIPRO.NS","ULTRACEMCO.NS",
"BAJFINANCE.NS","BAJAJFINSV.NS","ADANIENT.NS","ADANIPORTS.NS",
"POWERGRID.NS","NTPC.NS","ONGC.NS","COALINDIA.NS",
"TATAMOTORS.NS","TATASTEEL.NS","HINDUNILVR.NS",
"INDUSINDBK.NS","JSWSTEEL.NS","GRASIM.NS",
"TECHM.NS","DIVISLAB.NS","DRREDDY.NS",
"CIPLA.NS","EICHERMOT.NS","HEROMOTOCO.NS",
"BRITANNIA.NS","NESTLEIND.NS","APOLLOHOSP.NS",
"SBILIFE.NS","HDFCLIFE.NS","ICICIPRULI.NS"
]

# RSI Function
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Scan Button
if st.button("Run Full Market Scan"):
    results = []

    for stock in stocks:
        try:
            data = yf.download(stock, period="2y", progress=False)

            # Indicators
            data['Return'] = data['Close'].pct_change()
            data['MA10'] = data['Close'].rolling(10).mean()
            data['MA50'] = data['Close'].rolling(50).mean()
            data['RSI'] = calculate_rsi(data)
            data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

            data = data.dropna()

            X = data[['Return','MA10','MA50','RSI']]
            y = data['Target']

            model = RandomForestClassifier(n_estimators=100)
            model.fit(X, y)

            latest = X.iloc[-1].values.reshape(1, -1)
            prob = model.predict_proba(latest)[0][1]

            # Trend
            trend = "Uptrend" if data['MA10'].iloc[-1] > data['MA50'].iloc[-1] else "Downtrend"

            # Signal
            if prob > 0.60 and trend == "Uptrend":
                signal = "BUY"
            elif prob > 0.50:
                signal = "HOLD"
            else:
                signal = "AVOID"

            results.append((stock, round(prob*100,2), trend, signal))

        except:
            pass

    results = sorted(results, key=lambda x: x[1], reverse=True)

    st.subheader("Top Opportunities Today")

    for r in results[:10]:
        st.write(f"{r[0]} | Probability: {r[1]}% | {r[2]} | Signal: {r[3]}")

# Individual Stock Analysis
st.subheader("Analyze Single Stock")

single = st.text_input("Enter Stock (Example: RELIANCE.NS)")

if single:
    data = yf.download(single, period="1y")

    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()
    data['RSI'] = calculate_rsi(data)

    st.line_chart(data[['Close','MA50','MA200']])
    st.write("Current RSI:", round(data['RSI'].iloc[-1],2))
