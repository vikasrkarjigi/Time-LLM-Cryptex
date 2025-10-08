import pandas as pd
import matplotlib.pyplot as plt
"""
This script plots the inference results for the ohlcv_h_4m_inf dataset.
"""


csv_path = "dataset/cryptex/inference/ohlcv_h_4m_inf.csv"


data = pd.read_csv(csv_path)
print(data.head())
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
data.sort_index(inplace=True)


data.plot(y=['close', 'close_predicted_1'], color=['blue', 'red'])

plt.show()


dif = data['close'] - data['close_predicted_2']

print(dif.mean())
print(dif.std())
print(dif.min())
print(dif.max())

mse = (dif*dif).mean()
print("MSE: ", mse)

mae = dif.abs().mean()
print("MAE: ", mae)

