import urllib.parse
import hashlib
import hmac
import base64
import requests
import time
import json
import csv
import os
print(os.getcwd())


# Path to the directory containing the CSV files
csv_file_path = "/Users/faridsoroush/Documents/GitHub/Trading-Software/Exchange_API/api_keys.csv"

# Read the data from the CSV file
with open(csv_file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        api_key = row['api_key']
        secret_key = row['secret_key']

api_url = "https://api.binance.us"

# get binanceus signature
def get_binanceus_signature(data, secret):
    postdata = urllib.parse.urlencode(data)
    message = postdata.encode()
    byte_key = bytes(secret, 'UTF-8')
    mac = hmac.new(byte_key, message, hashlib.sha256).hexdigest()
    return mac

# Attaches auth headers and returns results of a GET request
def binanceus_request(uri_path, data, api_key, api_sec):
    headers = {}
    headers['X-MBX-APIKEY'] = api_key
    signature = get_binanceus_signature(data, api_sec)
    params = {
        **data,
        "signature": signature,
    }
    req = requests.get((api_url + uri_path), params=params, headers=headers)
    return req.json()  # Return response as JSON

uri_path = "/api/v3/account"
data = {
    "timestamp": int(round(time.time() * 1000)),
}

result = binanceus_request(uri_path, data, api_key, secret_key)

# Find the balance information for BTC and USDT and USD
balances = {}
for asset in result["balances"]:
    if asset["asset"] in ["BTC", "USDT", "USD"]:
        balances[asset["asset"]] = {
            "free": asset["free"],
            "locked": asset["locked"]
        }

# Print the balance information for BTC and USDT and USD
for asset, balance in balances.items():
    output = {
        "asset": asset,
        "free": balance["free"],
        "locked": balance["locked"]
    }
    print(json.dumps(output, indent=4))

