
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv('bat_with_temp.csv')
df['battery'] = df['battery'].interpolate()
df['temp'] = df['temp'].interpolate()

scaler = MinMaxScaler(feature_range=(0, 1))
df[['battery_scaled', 'temp_scaled']] = scaler.fit_transform(df[['battery', 'temp']])

df['Date'] = pd.to_datetime(df['Date'])

def prepare_multifeature_data(data, features, time_steps):
    X, y = [], []
    # Check if data has enough rows
    if len(data) <= time_steps:
        print("Not enough data entries for the given time_steps.")
        return np.array(X), np.array(y)

    for i in range(len(data) - time_steps):
        end_ix = i + time_steps
        if end_ix > len(data) - 1:
            break  # Ensure we do not go out of index
        X.append(data[features].iloc[i:end_ix].values)
        y.append(data['battery_scaled'].iloc[end_ix])
    return np.array(X), np.array(y)

features = ['battery_scaled', 'temp_scaled']
time_steps = 12
splits = TimeSeriesSplit(n_splits=10)

all_y_pred = []
all_y_test = []

for train_index, test_index in splits.split(df):
    train = df.iloc[train_index]
    test = df.iloc[test_index]

    X_train, y_train = prepare_multifeature_data(train, features, time_steps)
    X_test, y_test = prepare_multifeature_data(test, features, time_steps)

    model = Sequential()
    model.add(LSTM(units=50, input_shape=(time_steps, len(features))))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    y_pred = model.predict(X_test)
    all_y_pred.extend(y_pred)
    all_y_test.extend(y_test)

all_y_pred_inv = scaler.inverse_transform(np.column_stack([all_y_pred, np.zeros(len(all_y_pred))]))[:, 0]
all_y_test_inv = scaler.inverse_transform(np.column_stack([all_y_test, np.zeros(len(all_y_test))]))[:, 0]

rmse = np.sqrt(mean_squared_error(all_y_test_inv, all_y_pred_inv))
print("Root Mean Squared Error:", rmse)

plt.figure(figsize=(12, 6))
plt.plot(df['Date'][len(df) - len(all_y_test_inv):], all_y_test_inv, label='Actual')
plt.plot(df['Date'][len(df) - len(all_y_pred_inv):], all_y_pred_inv, label='Predicted')
plt.title('Battery Prediction with LSTM Considering Temperature')
plt.xlabel('Date')
plt.ylabel('Battery Level')
plt.legend()
plt.show()