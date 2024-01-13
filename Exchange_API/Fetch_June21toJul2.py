import requests
import urllib.parse
import hashlib
import hmac
import time
import csv
import os
import json
import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd



api_url = "https://api.binance.us"
uri_path = "/api/v3/klines"
csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_keys.csv')

with open(csv_file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        api_key = row['api_key']
        secret_key = row['secret_key']

def get_binanceus_signature(data, secret):
    postdata = urllib.parse.urlencode(data)
    message = postdata.encode()
    byte_key = bytes(secret, 'UTF-8')
    mac = hmac.new(byte_key, message, hashlib.sha256).hexdigest()
    return mac

def binanceus_request(uri_path, data, api_key, api_sec):
    headers = {}
    headers['X-MBX-APIKEY'] = api_key
    signature = get_binanceus_signature(data, api_sec)
    payload={
        **data,
        "signature": signature,
        }
    req = requests.post((api_url + uri_path), headers=headers, data=payload)
    return req.text

symbol = 'BTCUSDT'
interval = '1m'
limit = 1000

#Sample Data Output:
# [
#   [
#     1499040000000,      // Open time
#     "0.00386200",       // Open
#     "0.00386200",       // High
#     "0.00386200",       // Low
#     "0.00386200",       // Close
#     "0.47000000",  // Volume
#     1499644799999,      // Close time
#     "0.00181514",    // Quote asset volume
#     1,                // Number of trades
#     "0.47000000",    // Taker buy base asset volume
#     "0.00181514",      // Taker buy quote asset volume
#     "0" // Ignore.
#   ]
# ]

# 1937793 2023-06-21 05:18:00  28830.01  0.012250
# 1937794 2023-06-21 05:19:00  28847.01  0.301840

# Calculate timestamp for each year
end_time = int(datetime.datetime.now().timestamp() * 1000)

# Create a DataFrame to hold all data
all_data = pd.DataFrame()


# Start time - the timestamp for "2023-06-21 05:19:00"
start_time = int(datetime.datetime(2023, 6, 21, 5, 19).timestamp() * 1000)
end_time = int(datetime.datetime.now().timestamp() * 1000)


all_data = pd.DataFrame()

while end_time > start_time:

    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
        'endTime': end_time
    }

    resp = requests.get(api_url + uri_path, params=params)
    data = resp.json()

    if data:  # Check if data is not empty
        df_temp = pd.DataFrame(data)
        all_data = pd.concat([all_data, df_temp])
        end_time = data[0][0] - 1  # Update end_time to the timestamp of the earliest candlestick received, minus 1 millisecond to avoid duplicates
    else:
        print(f"\rtime:{end_time}", end='')
        end_time -= 60000*1000  # If no data is received, subtract 1 minute*1000 (limit rate) from end_time and continue

# Rename and convert columns
all_data.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore']
all_data['Open Time'] = pd.to_datetime(all_data['Open Time'], unit='ms')

# Save all data into a single CSV
all_data.to_csv('Data_from_June21toNow.csv', index=False)

