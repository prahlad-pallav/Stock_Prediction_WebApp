import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# st.write("Here's our first attempt at using data to create a table:")
# st.write(pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# }))

start_date = '2010-01-01'
end_date = '2019-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input("Enter Stock Ticker", "TSLA")
df = pdr.get_data_tiingo(user_input, api_key=("3ce7f8f4b1c6b1dd9258f0dcb03b739d35e5f9fd"), start=start_date, end=end_date)

st.subheader("Data from 2010-2019")

st.write(df.describe())

# Visualizations
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize = (12, 6))
plt.plot(df['close'].to_numpy(), label='Close')
st.pyplot(fig)


st.subheader("Closing Price vs Time Chart with 100MA")
ma100 = df.close.rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100.to_numpy(), label="ma100")
plt.plot(df['close'].to_numpy(), label='Close')
st.pyplot(fig)


st.subheader("Closing Price vs Time Chart with 100MA & 200MA")
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100.to_numpy(), label="ma100")
plt.plot(ma200.to_numpy(), label="ma100")
plt.plot(df['close'].to_numpy(), label='Close')
st.pyplot(fig)


data_training = pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df)*0.70) : int(len(df))])

scaler = MinMaxScaler(feature_range = (0, 1))

data_training_array = scaler.fit_transform(data_training)


#Loading stock_model
model = load_model('stock_model.h5')

past_100_days = data_training.tail(100)
# final_df = past_100_days.concat(data_testing, ignore_index = True)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)



x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scaled_factor = 1/scaler[0]
y_predicted = y_predicted * scaled_factor
y_test = y_test * scaled_factor


st.subheader("Predictions vs Original")
fig1 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label = "Original Price")
plt.plot(y_predicted, 'r', label = "Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig1)

