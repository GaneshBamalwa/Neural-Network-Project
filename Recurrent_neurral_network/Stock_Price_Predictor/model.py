'''
Includes all the visualization I used for the dataset 

'''
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('bank_nifty.csv')
data.columns = data.columns.str.strip()
data["Date"] = pd.to_datetime(data["Date"], format="%d-%b-%Y", errors="coerce")
print(data.head(1))
print(data.tail(1))

# Plot High and Low prices over time
plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['High'], label='High', color='green')
plt.plot(data['Date'], data['Low'], label='Low', color='red')
plt.legend()
plt.grid()
plt.title("High and Low Prices Over Time")

# Correlation heatmap for numeric features
numeric_data = data.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(12,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.grid()

# Prepare dataset for LSTM
stock_close = data["Close"]
dataset = stock_close.values.reshape(-1, 1)
training_data_len = int(np.ceil(len(dataset) * 0.80))

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len]

X_train, y_train = [], []

# Create sliding window sequences
for i in range(60, training_data_len):
    X_train.append(train_data[i-60:i])
    y_train.append(train_data[i])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model = keras.models.Sequential()

# First LSTM layer: captures sequential patterns
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(60,1)))

# Second LSTM layer: further processing
model.add(keras.layers.LSTM(64, return_sequences=False))

# Dense layer: interprets features
model.add(keras.layers.Dense(128, activation='relu'))

# Dropout layer: prevents overfitting
model.add(keras.layers.Dropout(0.5))

# Output layer: predicts closing price
model.add(keras.layers.Dense(1))

# Compile model
model.compile(
    optimizer='adam', 
    loss='mae',  
    metrics=[keras.metrics.RootMeanSquaredError(), 'mae', 'mse']
)

# Train model
training = model.fit(X_train, y_train, batch_size=32, epochs=25)

# Prepare test dataset
test_data = scaled_data[training_data_len-120:]
X_test, y_test = [], []

for i in range(120, len(test_data)):
    X_test.append(test_data[i-120:i])
    y_test.append(test_data[i])

X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Combine train and test for plotting
train = data[:training_data_len]
test = data[training_data_len:]
test = test.copy()
test['Predictions'] = predictions

# Plot training and testing results
plt.figure(figsize=(12,8))
plt.plot(train['Date'], train['Close'], label='Training Data', color='black')
plt.plot(test['Date'], test['Close'], label='Testing Data', color='orange')
plt.plot(test['Date'], test['Predictions'], label='Predicted', color='blue')
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title("Stock Price Predictions")
plt.legend()
plt.grid()
plt.show()

# Plot September data
september_data = test[test['Date'].dt.month == 9]

plt.figure(figsize=(14,6))
plt.plot(september_data['Date'], september_data['Close'], label='Actual Close', color='orange')
plt.plot(september_data['Date'], september_data['Predictions'], label='Predicted Close', color='blue')
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title("Stock Price Predictions vs Actual (September)")
plt.legend()
plt.xticks(september_data['Date'], rotation=90)

y_min = september_data[['Close', 'Predictions']].min().min()
y_max = september_data[['Close', 'Predictions']].max().max()
plt.yticks(range(int(y_min), int(y_max)+150, 150))
plt.grid(True)
plt.show()
