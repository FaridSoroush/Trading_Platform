import pandas as pd
import numpy as np
# from Get_balance import get_free_btc, get_free_usd, get_free_usdt
from joblib import dump, load
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import deque
import requests
import urllib.parse
import hashlib
import hmac
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from datetime import datetime


# Exchange API
api_key = 'ZxOQ7t6879VgwLQzhxIixxB39iIAUX2355YMjFu1yQzDgUmpquOq3Gyb3XESVJBs'
secret_key = 'tjuC6WfLgKNabCbC21KGJq0uF3XjEzvuXVBU1rPuW7FIrbra4IfOfw4pqILLHT5i'
api_url = "https://api.binance.us"
uri_path = "/api/v3/account"
uri_path_market = "/api/v3/klines"
uri_path_trading = "/api/v3/order"
data={'timestamp': int(round(time.time() * 1000))}


# Trading Parameters
symbol = 'BTCUSDT'
quantity=0.027/100*5      # initial: 0.00193116
type="LIMIT"
side="SELL"

def get_binanceus_signature(data, secret):
    postdata = urllib.parse.urlencode(data)
    message = postdata.encode()
    byte_key = bytes(secret, 'UTF-8')
    mac = hmac.new(byte_key, message, hashlib.sha256).hexdigest()
    return mac

def binanceus_request(uri_path_trading, data, api_key, api_sec):
    headers = {}
    headers['X-MBX-APIKEY'] = api_key
    signature = get_binanceus_signature(data, api_sec)
    payload={
        **data,
        "signature": signature,
        }
    req = requests.post((api_url + uri_path_trading), headers=headers, data=payload)
    return req.text


# data = {
#     "symbol": symbol,
#     "side": side,
#     "type": type,
#     "quantity": quantity,
#     "timestamp": int(round(time.time() * 1000))
# }

data = {
    "symbol": symbol,
    "side": side,
    "type": type,
    "quantity": quantity,
    "price": float(31350.00),
    "timeInForce": "GTC",
    "timestamp": int(round(time.time() * 1000))
}

result = binanceus_request(uri_path_trading, data, api_key, secret_key)
print("POST {}: {}".format(uri_path_trading, result))