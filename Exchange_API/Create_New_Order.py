import urllib.parse
import hashlib
import hmac
import base64
import requests
import time
import csv

# Specify the file path of the CSV file
csv_file_path = 'api_keys.csv'

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

# Attaches auth headers and returns results of a POST request
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

symbol="BTCUSD"
side="SELL"
type="MARKET"
quantity=0.0001

#print("api key:", api_key)
#print("secret key:", secret_key)


uri_path = "/api/v3/order"
data = {
    "symbol": symbol,
    "side": side,
    "type": type,
    "quantity": quantity,
    "timestamp": int(round(time.time() * 1000))
}

result = binanceus_request(uri_path, data, api_key, secret_key)
print("POST {}: {}".format(uri_path, result))

#print(data)
