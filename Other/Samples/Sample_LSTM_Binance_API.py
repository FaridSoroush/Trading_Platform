import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Initialize the ccxt exchange
exchange = ccxt.binanceus()
symbol = 'BTC/USD'

# Fetch market data
limit = 1000
bars = exchange.fetch_ohlcv(symbol, '1s', limit=limit)
df = pd.DataFrame(bars[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Preprocess the data
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df.drop(['volume'], axis=1, inplace=True)

# Define the number of time steps to consider
n_steps = 5

# Prepare the data for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

X = []
y = []
for i in range(n_steps, len(scaled_data)):
    X.append(scaled_data[i - n_steps:i])
    y.append(scaled_data[i, 3])  # Corrected index for close price
X = np.array(X)
y = np.array(y)


# Split the data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Set the hyperparameters
input_size = X_train.shape[2]
hidden_size = 64
output_size = 1
num_epochs = 100
learning_rate = 0.001

# Instantiate the LSTM model
model = LSTM(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the LSTM model
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Predict on the training set
with torch.no_grad():
    model.eval()
    train_predictions = model(X_train_tensor).numpy()

# Inverse transform the predictions and actual values
train_predictions = scaler.inverse_transform(np.concatenate((X_train[:, -1, :-1], train_predictions), axis=1))
y_train_actual = scaler.inverse_transform(np.concatenate((X_train[:, -1, :-1], y_train.reshape(-1, 1)), axis=1))

# Calculate the mean squared error
mse = mean_squared_error(y_train_actual[:, -1], train_predictions[:, -1])

# Plotting the predicted and actual values
plt.figure(figsize=(12, 6))
plt.plot(df.index[n_steps:train_size + n_steps], y_train_actual[:, -1], label='Actual')
plt.plot(df.index[n_steps:train_size + n_steps], train_predictions[:, -1], label='Predicted')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Price')
plt.legend()
plt.show()

print("Mean Squared Error:", mse)