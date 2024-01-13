import pandas as pd
import time
import numpy as np
from collections import deque
import requests
import urllib.parse
import hashlib
import hmac
import time
import pandas as pd
from Get_balance import get_free_btc, get_free_usd, get_free_usdt
from joblib import dump, load
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib


delta_t_trading = 60 # seconds

n_steps = 30

model_parameters_file = '../Prediction_Model/GradientBoosting/GB_model_parameters_10000_1000_2_1_10_10.pkl'

# Exchange API
api_key = 'ZxOQ7t6879VgwLQzhxIixxB39iIAUX2355YMjFu1yQzDgUmpquOq3Gyb3XESVJBs'
secret_key = 'tjuC6WfLgKNabCbC21KGJq0uF3XjEzvuXVBU1rPuW7FIrbra4IfOfw4pqILLHT5i'
api_url = "https://api.binance.us"
uri_path = "/api/v3/account"
uri_path_market = "/api/v3/klines"
data={'timestamp': int(round(time.time() * 1000))}

# trading parameters
paper_trading=True
symbol = 'BTC/USDT'
Exchange_Comission=0
position_ratio=0.02
quantity=0
paper_account_profit=0
type="MARKET"
trade_approved = False
num_approved_trades=0
trade_comission=0

def get_binanceus_signature(data, secret):
    postdata = urllib.parse.urlencode(data)
    message = postdata.encode()
    byte_key = bytes(secret, 'UTF-8')
    mac = hmac.new(byte_key, message, hashlib.sha256).hexdigest()
    return mac

def binanceus_request(uri_path_market, data, api_key, api_sec):
    headers = {}
    headers['X-MBX-APIKEY'] = api_key
    signature = get_binanceus_signature(data, api_sec)
    payload={
        **data,
        "signature": signature,
        }
    req = requests.post((api_url + uri_path), headers=headers, data=payload)
    return req.text

def fetch_live_market_data(limit):

    params = {
    'symbol': 'BTCUSDT',
    'interval': '1m',
    'limit': limit
    }

    data = []
    resp = requests.get(api_url + uri_path_market, params=params)
    data = resp.json()
    df = pd.DataFrame(data, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
    df['Open Time'] = pd.to_datetime(df['Open Time'])
    df['Open Time'] = df['Open Time'].values.astype(np.int64) // 10 ** 9
    df = df.sort_values('Open Time')

    return df

model = GradientBoostingRegressor(n_estimators=10000,  
                                  max_depth=1000,      
                                  min_samples_split=2,
                                  learning_rate=1, 
                                  min_samples_leaf=10,
                                  n_iter_no_change=10, 
                                  loss='squared_error')

model = joblib.load(model_parameters_file)

def trade_evaluation():

    global num_approved_trades

    side=None

    if paper_trading==True:
        U=1
        BTC_balance=0.00193116*U
        USDT_balance=47.79670558*U
    
    elif paper_trading==False:
        BTC_balance=get_free_btc()
        USDT_balance=get_free_usdt()

    
    # position ratio: 0.02
    # BTC_balance=0.00193116

    # quantity in BTC 
    quantity_long=position_ratio*BTC_balance
    quantity_short=position_ratio*USDT_balance/current_price

    price_gap_percent=abs((predicted_price-current_price)/current_price*100)

    # comission in USDT
    # trade_comission_long=(current_price+predicted_price)*(quantity_long)*Exchange_Comission
    # trade_comission_short=(current_price+predicted_price)*(quantity_short)*Exchange_Comission
    trade_comission_long=0
    trade_comission_short=0
    # # MSE_error=predicted_price*0.0006 # max [sqr(338^0.5/30000) , sqr(7.7e-6^0.5)] # math is messed up, double check

    # potential profit in USDT
    potential_profit_long=(predicted_price-current_price)*quantity_long
    potential_profit_short=-(predicted_price-current_price)*quantity_short

    if potential_profit_long > (trade_comission_long):
        side="BUY"
        trade_approved=True
        num_approved_trades+=1
        print("trading now...")

    elif potential_profit_short> (trade_comission_short):
        side='SELL'
        trade_approved=True
        num_approved_trades+=1
        print("trading now...")

    else:
        side='None'
        trade_approved=False


    if side=='BUY':
        quantity=quantity_long
        trade_comission=trade_comission_long
    elif side=='SELL':
        quantity=quantity_short
        trade_comission=trade_comission_short
    else:
        quantity=0
        trade_comission=0

    
    print(f'{"trade approved:":<30}{trade_approved}')
    print(f'{"side:":<30}{side}')
    print(f'{"predicted price: $":<30}{predicted_price}')
    print(f'{"current price: $":<30}{current_price}')
    print(f'{"price gap %:":<30}{price_gap_percent}')
    print(f'{"quantity (in BTC):":<30}{quantity}')
    print(f'{"potential profit long: $":<30}{potential_profit_long}')
    print(f'{"potential profit short: $":<30}{potential_profit_short}')
    # print(f'{"trade comission long: $":<30}{trade_comission_long}')
    # print(f'{"trade comission short: $":<30}{trade_comission_short}')
    # print(f'{"trade comission $":<30}{trade_comission}')
    print(f'{"number of approved trades:":<30}{num_approved_trades}')
    print("----------------------------------------------------")

    return trade_approved, side, quantity


prediction = []

while True:

    df = fetch_live_market_data(n_steps+1) 
    df=df[['Open Time','Close', 'Volume']]
    predicting_index = df.columns.get_loc('Close')

    current_price = float(df.iloc[-1]['Close'])

    scaled_data = df.values

    X = []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i])
    X = np.array(X)
    
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = X.shape[2]
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # flatten input
    n_input = X.shape[1] * X.shape[2]
    X_flat = X.reshape((X.shape[0], n_input))

    predicted_price = model.predict(X_flat)

    trade_approved, side, quantity = trade_evaluation()

    if trade_approved:
        
        if paper_trading==True:

            if side=='SELL':

                sold_price=float(fetch_live_market_data(limit=1).iloc[-1]['Close'])
                print("sold price:", sold_price)
                time.sleep(delta_t_trading)
                SecTrade_bought_price=float(fetch_live_market_data(limit=1).iloc[-1]['Close'])
                print("bought price:", SecTrade_bought_price)
                paper_account_profit+=(-(SecTrade_bought_price-sold_price)*(quantity)*(1-Exchange_Comission))
                print("executed the Sell-Buy trade")
                print("profit of this trade:", (-(SecTrade_bought_price-sold_price)*(quantity)*(1-Exchange_Comission)))

            elif side=='BUY':

                purchased_price=float(fetch_live_market_data(limit=1).iloc[-1]['Close'])
                print("bought price:", purchased_price)
                time.sleep(delta_t_trading)
                SecTrade_sold_price=float(fetch_live_market_data(limit=1).iloc[-1]['Close'])
                print("sold price:", SecTrade_sold_price)
                paper_account_profit+=(+(SecTrade_sold_price-purchased_price)*(quantity)*(1-Exchange_Comission))
                print("executed the Buy-Sell trade")
                print("profit of this trade:", (+(SecTrade_sold_price-purchased_price)*(quantity)*(1-Exchange_Comission)))

            print("paper account profit:", paper_account_profit)
            print("----------------------------------------------------")

        elif paper_trading==False:

            uri_path_trading = "/api/v3/order"
            data = {
                "symbol": symbol,
                "side": side,
                "type": type,
                "quantity": quantity,
                "timestamp": int(round(time.time() * 1000))
            }

            result = binanceus_request(uri_path_trading, data, api_key, secret_key)
            print("POST {}: {}".format(uri_path, result))

    time.sleep(1)