import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.title("Indian Stock Predictor")

stock = st.text_input("Enter NSE Stock (Example: RELIANCE.NS)")

if stock:
    data = yf.download(stock, period="2y")

    if len(data) > 0:
        data['Return'] = data['Close'].pct_change()
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

        data = data.dropna()

        X = data[['Return', 'MA10', 'MA50']]
        y = data['Target']

        model = RandomForestClassifier()
        model.fit(X, y)

        latest = X.iloc[-1].values.reshape(1, -1)
        prediction = model.predict(latest)[0]

        st.write("Current Price:", data['Close'].iloc[-1])

        if prediction == 1:
            st.success("Prediction: Stock may go UP")
        else:
            st.error("Prediction: Stock may go DOWN")

        st.line_chart(data['Close'])
    else:
        st.write("Invalid Stock")
