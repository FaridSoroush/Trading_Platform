import urllib.parse
import hashlib
import hmac
import base64
import requests
import time
import json

api_key = 'LD5eq9nHMz51lvNxA4Yk4Npm02kS09oZU7tdvvkQKpcmLhdoCmUPKEv9xebfHavB'
secret_key = 'DtWIdtZRbtBDzzZPFVVgH39nr3VTNhz7njsb3HHGrYoHJcVl5fB1Jpdc5HoJJTfo'
api_url = "https://api.binance.us"
symbol='BTCUSDT'

orderId=1040470329


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
    params={
        **data,
        "signature": signature,
        }
    req = requests.delete((api_url + uri_path), params=params, headers=headers)
    return req.text



uri_path = "/api/v3/order"
data = {
  "orderId": orderId,
  "symbol": symbol,
  "timestamp": int(round(time.time() * 1000))
}

result = binanceus_request(uri_path, data, api_key, secret_key)
# print("DELETE {}: {}".format(uri_path, result))

result = json.loads(result)
# print("GET {}: {}".format(uri_path, result))
# prtin the result with 4 indent
print(json.dumps(result, indent=4))