import csv
import matplotlib.pyplot as plt
import ccxt
import urllib.parse
import hashlib
import hmac
import base64
import requests
import time
import csv
import json
import time
import pandas as pd
import numpy as np 
import datetime
import urllib.parse
import hashlib
import hmac
import base64
import requests
import time
import csv
import json
import urllib.parse
import hashlib
import hmac
import base64
import requests
import time
import json

# this script will cancel all open orders that are Y sigma away from the average price in the past X hours
# the script checks this condition every wait_time seconds

X_hours = 3
Y_sigma = 3
wait_time = 15 * 60 # 15 min

while True:

    api_key = 'LD5eq9nHMz51lvNxA4Yk4Npm02kS09oZU7tdvvkQKpcmLhdoCmUPKEv9xebfHavB'
    secret_key = 'DtWIdtZRbtBDzzZPFVVgH39nr3VTNhz7njsb3HHGrYoHJcVl5fB1Jpdc5HoJJTfo'
    api_url = "https://api.binance.us"

    # get binanceus signature
    def get_binanceus_signature(data, secret):
        postdata = urllib.parse.urlencode(data)
        message = postdata.encode()
        byte_key = bytes(secret, 'UTF-8')
        mac = hmac.new(byte_key, message, hashlib.sha256).hexdigest()
        return mac

    # Attaches auth headers and returns results of a GET request
    def binanceus_get_request(uri_path, data, api_key, api_sec):
        headers = {}
        headers['X-MBX-APIKEY'] = api_key
        signature = get_binanceus_signature(data, api_sec)
        params={
            **data,
            "signature": signature,
            }
        req = requests.get((api_url + uri_path), params=params, headers=headers)
        return req.text

    # Attaches auth headers and returns results of a DELETE request
    def binanceus_delete_request(uri_path, data, api_key, api_sec):
        headers = {}
        headers['X-MBX-APIKEY'] = api_key
        signature = get_binanceus_signature(data, api_sec)
        params={
            **data,
            "signature": signature,
            }
        req = requests.delete((api_url + uri_path), params=params, headers=headers)
        return req.text

    # Fetch past X hours of data
    exchange = ccxt.binanceus()
    symbol = 'BTC/USDT'
    X_minutes = X_hours * 60
    bars = exchange.fetch_ohlcv(symbol, '1m', limit=X_minutes)
    # Get closing prices
    prices = [bar[4] for bar in bars]
    # Calculate average price and standard deviation (sigma)
    average_price = np.mean(prices)
    sigma = np.std(prices)

    print(f"Average price in the past {X_hours} hours: {average_price}")
    print(f"Standard deviation (sigma) in the past {X_hours} hours: {sigma}")

    symbol = 'BTCUSDT' # adjusting symbol to binanceus format

    # Get all of the open positions
    uri_path = "/api/v3/openOrders"
    data = {
        "timestamp": int(round(time.time() * 1000))
    }

    result = binanceus_get_request(uri_path, data, api_key, secret_key)
    result = json.loads(result)

    # Iterate over the result and check each open position
    for open_position in result:
        order_id = open_position['orderId']
        price = float(open_position['price'])
        if price == 0:  # Market orders
            cummulative_quote_qty = float(open_position['cummulativeQuoteQty'])
            executed_qty = float(open_position['executedQty'])
            price = cummulative_quote_qty / executed_qty  # price per unit

        # If the position is Y sigma away from the average price, cancel the position
        if abs(price - average_price) > Y_sigma * sigma:
            uri_path = f"/api/v3/order"
            data = {
                "symbol": symbol,
                "orderId": order_id,
                "timestamp": int(round(time.time() * 1000))
            }
            print(f"Order {order_id} at price {price} is "+str(Y_sigma)+" sigma away from the average price. Cancelling...")
            result = binanceus_delete_request(uri_path, data, api_key, secret_key)
            print(result)

    print("Sleeping for "+str(wait_time/60)+" minutes...")
    time.sleep(wait_time) 

