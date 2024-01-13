import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from joblib import dump, load
from tqdm import tqdm
import time
from torch.nn import DataParallel
import os
import ta

n_steps = 10

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load, convert and sort the data
if device == torch.device("cuda"):
    input_filename = "BTCUSDT4Y1MKline.csv"

elif device == torch.device("cpu"):
    os.chdir('/Users/faridsoroush/Documents/GitHub/Trading-Software/Prediction_Model/Transformer/')
    # print("Current working directory: {0}".format(os.getcwd()))
    input_filename = "BTCUSDT4Y1MKline.csv"

else:
    print("Device not found")

df = pd.read_csv(input_filename)

# Check if all values in the last column are zero
if (df.iloc[:, -1] == 0).all():
    # Drop the last column
    df = df.iloc[:, :-1]

# Drop 'Open' and 'Close time' columns
df = df.drop(['Open', 'Close Time'], axis=1)

df['Open Time'] = pd.to_datetime(df['Open Time'])
df['Open Time'] = df['Open Time'].values.astype(np.int64) // 10 ** 9
df = df.sort_values('Open Time')

# Add moving average
df['Moving_Avg_Close_1000X_n_steps'] = df['Close'].rolling(1000*n_steps).mean()
df['Moving_Avg_Close_100X_n_steps'] = df['Close'].rolling(100*n_steps).mean() 
df['Moving_Avg_Close_10X_n_steps'] = df['Close'].rolling(10*n_steps).mean()   
for i in range(1, 10*n_steps):
    df.loc[i, 'Moving_Avg_Close_10X_n_steps'] = df.loc[:i, 'Close'].mean()
for i in range(1, 100*n_steps):
    df.loc[i, 'Moving_Avg_Close_100X_n_steps'] = df.loc[:i, 'Close'].mean()
for i in range(1, 1000*n_steps):
    df.loc[i, 'Moving_Avg_Close_1000X_n_steps'] = df.loc[:i, 'Close'].mean()

print("Moving Averages added")

# Add MACD, RSI
df['MACD'] = ta.trend.MACD(df['Close']).macd()
df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

print("MACD, RSI added")

# Set the window size for local high/low calculation
window_size = 10*n_steps # you can adjust this value
df['High_RollingMax'] = df['High'].rolling(window=window_size, min_periods=1).max()
df['Low_RollingMin'] = df['Low'].rolling(window=window_size, min_periods=1).min()
# Initialize columns for Fibonacci levels
df['Fibonacci_Level_1'] = np.nan
df['Fibonacci_Level_2'] = np.nan
df['Fibonacci_Level_3'] = np.nan
# Calculate Fibonacci levels for each period
for i in range(len(df)):
    high = df.loc[i, 'High_RollingMax']
    low = df.loc[i, 'Low_RollingMin']
    difference = high - low
    df.loc[i, 'Fibonacci_Level_1'] = high - difference * 0.236
    df.loc[i, 'Fibonacci_Level_2'] = high - difference * 0.382
    df.loc[i, 'Fibonacci_Level_3'] = high - difference * 0.618

print("Fibonacci levels added")

# Compute Bollinger Bands
df['Bollinger_High'] = df['Close'].rolling(window_size, min_periods=1).max()
df['Bollinger_Middle'] = df['Close'].rolling(window_size, min_periods=1).mean()
df['Bollinger_Low'] = df['Close'].rolling(window_size, min_periods=1).min()

print("Bollinger Bands added")

# Compute Stochastic Oscillator
low_min  = df['Low'].rolling(window_size, min_periods=1).min()
high_max = df['High'].rolling(window_size, min_periods=1).max()
df['Stochastic_Oscillator'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100

print("Stochastic Oscillator added")

# Compute Average True Range
df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
# Print the length of the DataFrame before drop operation
print(f"Before drop: {len(df)} rows")
# Drop rows with NaN values
df_dropped = df.dropna()
# Print the length of the DataFrame after drop operation
print(f"After drop: {len(df_dropped)} rows")
# Calculate and print the number of dropped rows
num_dropped = len(df) - len(df_dropped)
print(f"Dropped {num_dropped} rows because of NaN values")
df=df_dropped

print("Average True Range added")

# Print the number of rows and columns
print(f"The final dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
# Print the first two rows
print(df.head(10)) 

# Save the dataframe
df.to_csv("BTCUSDT4Y1MKline_cleaned_24features_nSteps"+str(n_steps)+".csv", index=False)

print("Importing, cleaning, sorting, adding features: Done")
