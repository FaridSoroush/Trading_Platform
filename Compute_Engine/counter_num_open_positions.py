
import urllib.parse
import hashlib
import hmac
import base64
import requests
import time
import csv
import json

api_key = 'ZxOQ7t6879VgwLQzhxIixxB39iIAUX2355YMjFu1yQzDgUmpquOq3Gyb3XESVJBs'
secret_key = 'tjuC6WfLgKNabCbC21KGJq0uF3XjEzvuXVBU1rPuW7FIrbra4IfOfw4pqILLHT5i'
api_url = "https://api.binance.us"

# get binanceus signature
def get_binanceus_signature(data_count_num_open_positions, secret):
    postdata = urllib.parse.urlencode(data_count_num_open_positions)
    message = postdata.encode()
    byte_key = bytes(secret, 'UTF-8')
    mac = hmac.new(byte_key, message, hashlib.sha256).hexdigest()
    return mac

# Attaches auth headers and returns results of a POST request
def binanceus_request(uri_path, data, api_key, api_sec):
    headers = {}
    headers['X-MBX-APIKEY'] = api_key
    signature = get_binanceus_signature(data, api_sec)
    params={
        **data,
        "signature": signature,
        }
    req = requests.get((api_url + uri_path), params=params, headers=headers)
    return req.text

uri_path_count_num_open_positions = "/api/v3/openOrders"

# count number of open positions
def count_num_open_positions():
    data_count_num_open_positions = {
        "timestamp": int(round(time.time() * 1000))
    }
    result_count_num_open_positions = binanceus_request(uri_path_count_num_open_positions, data_count_num_open_positions, api_key, secret_key)
    result_count_num_open_positions = json.loads(result_count_num_open_positions)
    count = 0
    for _ in result_count_num_open_positions:
        count += 1
    return count

# print(count_num_open_positions())