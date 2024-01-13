import urllib.parse
import hashlib
import hmac
import requests
import time
import json
import csv
import os

api_key = 'LD5eq9nHMz51lvNxA4Yk4Npm02kS09oZU7tdvvkQKpcmLhdoCmUPKEv9xebfHavB'
secret_key = 'DtWIdtZRbtBDzzZPFVVgH39nr3VTNhz7njsb3HHGrYoHJcVl5fB1Jpdc5HoJJTfo'
api_url = "https://api.binance.us"

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
    params = {
        **data,
        "signature": signature,
    }
    req = requests.get((api_url + uri_path), params=params, headers=headers)
    return req.json()

def get_balance(asset_name):

    uri_path = "/api/v3/account"
    data = {
        "timestamp": int(round(time.time() * 1000)),
    }

    result = binanceus_request(uri_path, data, api_key, secret_key)

    balances = {}
    for asset in result["balances"]:
        if asset["asset"] == asset_name:
            balances[asset["asset"]] = {
                "free": asset["free"],
                "locked": asset["locked"]
            }
    
    return float(balances[asset_name]["free"]) if asset_name in balances else 0

def get_locked_balance(asset_name):

    uri_path = "/api/v3/account"
    data = {
        "timestamp": int(round(time.time() * 1000)),
    }

    result = binanceus_request(uri_path, data, api_key, secret_key)

    balances = {}
    for asset in result["balances"]:
        if asset["asset"] == asset_name:
            balances[asset["asset"]] = {
                "free": asset["free"],
                "locked": asset["locked"]
            }
    
    return float(balances[asset_name]["locked"]) if asset_name in balances else 0


def get_free_btc():
    return get_balance("BTC")

def get_free_usd():
    return get_balance("USD")

def get_free_usdt():
    return get_balance("USDT")

def get_locked_btc():
    return get_locked_balance("BTC")

def get_locked_usd():
    return get_locked_balance("USD")

def get_locked_usdt():
    return get_locked_balance("USDT")

def get_total_btc():
    return get_free_btc() + get_locked_btc()

def get_total_usd():
    return get_free_usd() + get_locked_usd()

def get_total_usdt():
    return get_free_usdt() + get_locked_usdt()

# print("get_locked_usdt(): ", get_locked_usdt())

# print("get_locoked_usdt", get_locked_balance("USDT"))