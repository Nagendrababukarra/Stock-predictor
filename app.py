import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model("Stock Predictions Model.keras")



st.header('Stock Market Predictor')


stock = "GOOG"
st.info("This AI model is trained specifically for Google (GOOG) stock.")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

data = yf.download(stock, start ,end)
if data.empty:
    st.error("Failed to fetch stock data. Please enter a valid stock symbol.")
    st.stop()


st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

scaler.fit(data_train)               # <-- ADDED
data_test_scale = scaler.transform(data_test)   # <-- CHANGED


st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig4)


# Tomorrow Price Prediction


st.subheader("Tomorrow Price Prediction")
last_100_days = data.Close.tail(100).values
last_100_scaled = scaler.transform(last_100_days.reshape(-1,1))
x_input = np.reshape(last_100_scaled, (1, 100, 1))

tomorrow_price = model.predict(x_input)

tomorrow_price = scaler.inverse_transform(tomorrow_price)

st.success(f"Predicted Tomorrow Closing Price: ${float(tomorrow_price[0][0]):.2f}")







