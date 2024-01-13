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


#############################################################################
# fetch historic data for BTC/USDT for the past X years, every 1 min, Kline #
#############################################################################


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
X=6  # Total years

# Calculate timestamp for each year
end_time = int(datetime.datetime.now().timestamp() * 1000)

# Create a DataFrame to hold all data
all_data = pd.DataFrame()

# Fetch data for each year
for i in range(3, X):
    year_start = int((datetime.datetime.now() - datetime.timedelta(days=(i+1)*365)).timestamp() * 1000)
    year_end = int((datetime.datetime.now() - datetime.timedelta(days=i*365)).timestamp() * 1000)

    data_for_year = []
    total_duration = year_end - year_start
    while year_end > year_start:

        elapsed_time = year_end - year_start
        progress_percentage = 100 * elapsed_time / total_duration
        print(f"\rProgress for year {i+1}: {100 - progress_percentage:.2f}%", end='')

        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
            'endTime': year_end
        }

        resp = requests.get(api_url + uri_path, params=params)
        data = resp.json()

        if data:  # Check if data is not empty
            data_for_year.extend(data)
            year_end = data[0][0] - 1  # Update end_time to the timestamp of the earliest candlestick received, minus 1 millisecond to avoid duplicates
        else:
            print(f"\rtime:{year_end}", end='')
            year_end -= 60000*1000  # If no data is received, subtract 1 minute*1000 (limit rate) from year_end and continue

    df_year = pd.DataFrame(data_for_year, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
    df_year['Open Time'] = pd.to_datetime(df_year['Open Time'], unit='ms')
    df_year.to_csv(f'data_for_year_{i+1}.csv', index=False)

    all_data = pd.concat([all_data, df_year])

    print(f"\nFinished fetching data for year {i+1} out of {X}")

    # Pause for a minute before fetching next year's data
    time.sleep(1)


# Save all data into a single CSV
all_data.to_csv('all_data.csv', index=False)


#############################################
# plot closing price for the entire dataset #
#############################################

# # Extract the closing prices and the dates
# closing_prices = [float(item[4]) for item in all_data]  # closing prices are at index 4
# dates = [datetime.datetime.fromtimestamp(item[0]/1000) for item in all_data]  # date (open time) is at index 0

# # Plot the data
# plt.figure(figsize=(14, 7))
# plt.plot(dates, closing_prices)
# plt.title('BTCUSDT Closing Prices')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.grid(True)
# plt.show(block=False)

# Now all_data should contain data for the last X years
# print(len(all_data))


################
# plot candles #
################

# # Get the data for the last week
# one_week_ago = int((datetime.datetime.now() - datetime.timedelta(days=7)).timestamp() * 1000)
# last_week_data = [item for item in all_data if item[0] >= one_week_ago]

# # Prepare the data for the candlestick plot
# # The OHLC data is at indices 1, 2, 3, 4 respectively
# ohlc_data = []
# for item in last_week_data:
#     date = datetime.datetime.fromtimestamp(item[0]/1000)
#     open_price = float(item[1])
#     high_price = float(item[2])
#     low_price = float(item[3])
#     close_price = float(item[4])
#     ohlc_data.append([date, open_price, high_price, low_price, close_price])

# # Make a dataframe for the OHLC data
# ohlc_df = pd.DataFrame(ohlc_data, columns=['Date', 'Open', 'High', 'Low', 'Close'])
# ohlc_df.set_index('Date', inplace=True)

# # Plot the candlestick chart
# mpf.plot(ohlc_df, type='candle', style='charles', title='BTCUSDT Last Week Candlestick', ylabel='Price')

# # Show the plot
# plt.show()