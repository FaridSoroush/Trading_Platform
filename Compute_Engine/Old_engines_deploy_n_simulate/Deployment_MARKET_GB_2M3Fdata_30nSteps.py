# model GB 2M3F, only the last 5K, with stop loss $0.05 n 30min wait, expoential growth, 5% position ratio

import pandas as pd
import numpy as np
from Get_balance import get_free_btc, get_free_usd, get_free_usdt
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
import json
import math

# making sure that the script runs at the beginning of 
# each minute (since precition is based on 1min intervals of cloasing price)
now = time.time()
initial_wait_time = 60 - (now % 60)
print("initial_wait_time: ", initial_wait_time)
time.sleep(initial_wait_time)

# stop loss
stop_loss = 0.005 # in US dollars
wait_time_steps = 10 * 60 # in seconds

step_size = 0.00001 # this value is from Binance API documentation but I think it
                    # is wrong. I think it actually is 1e-5 (added an if at the 
                    # end of the trade_eval function to account for this)

n_steps = 30
predicting_index = 1

delta_t_trading = 60 - 1 # in seconds

# historic_data_input_filename = "Data_from_June21toNow.csv"
# historic_df = pd.read_csv(historic_data_input_filename)
# historic_df=historic_df[['Open Time','Close', 'Volume']]
# historic_df['Open Time'] = pd.to_datetime(historic_df['Open Time'])
# historic_df['Open Time'] = historic_df['Open Time'].values.astype(np.int64) // 10 ** 9
# historic_df = historic_df.sort_values('Open Time')

# account balance on Jul 2nd, 2023 at 9:26pm PST:
# {
#     "asset": "BTC",
#     "free": "0.00193116",
#     "locked": "0.00000000"
# }
# {
#     "asset": "USDT",
#     "free": "47.79670558",
#     "locked": "0.00000000"
# }
# {
#     "asset": "USD",
#     "free": "0.00017500",
#     "locked": "0.00000000"
# }


# Exchange API
api_key = 'ZxOQ7t6879VgwLQzhxIixxB39iIAUX2355YMjFu1yQzDgUmpquOq3Gyb3XESVJBs'
secret_key = 'tjuC6WfLgKNabCbC21KGJq0uF3XjEzvuXVBU1rPuW7FIrbra4IfOfw4pqILLHT5i'
api_url = "https://api.binance.us"
uri_path = "/api/v3/account"
uri_path_market = "/api/v3/klines"
uri_path_trading = "/api/v3/order"
data={'timestamp': int(round(time.time() * 1000))}

# trading parameters
paper_trading=False # IMPORTANT
symbol = 'BTCUSDT'
Exchange_Comission= 0
position_ratio=0.05
quantity= 0.0
print("quantity: ", quantity)
print("quantity type: ", type(quantity))
type="MARKET"
trade_approved = False
num_approved_trades=0
trade_comission=0
global initial_run
initial_run = True

model = GradientBoostingRegressor(n_estimators=10000,  
                                  max_depth=1000,      
                                  min_samples_split=2,
                                  learning_rate=1, 
                                  min_samples_leaf=10,
                                  n_iter_no_change=10, 
                                  loss='squared_error')
model_parameters_file = '../Prediction_Model/GradientBoosting/GB_model_data_2M_main_parameters_10000_1000_2_1_10_10.pkl'
model = joblib.load(model_parameters_file)


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
    df = df [['Open Time', 'Close', 'Volume']]
    df['Open Time'] = pd.to_datetime(df['Open Time'])
    df['Open Time'] = df['Open Time'].values.astype(np.int64) // 10 ** 9
    df = df.sort_values('Open Time')

    return df

def trade_evaluation():

    global num_approved_trades
    global BTC_balance
    global USDT_balance

    quantity=0.0

    side='NONE'

    if paper_trading==True and initial_run==True:
        print("Paper Trading")
    
    elif paper_trading==False:
        BTC_balance=get_free_btc()
        USDT_balance=get_free_usdt()

    quantity_long=position_ratio*BTC_balance
    quantity_short=position_ratio*USDT_balance/current_price

    price_gap_percent=abs((predicted_price-current_price)/current_price*100)

    potential_profit_long=(predicted_price-current_price)*quantity_long
    potential_profit_short=-(predicted_price-current_price)*quantity_short

    if potential_profit_long > 0 and quantity_long > step_size:

        quantity=quantity_long

        if (quantity - (quantity % step_size)) > step_size:

            quantity= math.floor(quantity * 1e5) / 1e5
            side='BUY'
            trade_approved=True
            num_approved_trades+=1
            profit__per_trade_predicted = potential_profit_long
            profit_per_trade_predicted_list.append(profit__per_trade_predicted)
            print("tading now...")

        else:
            quantity=0

    elif potential_profit_short > 0 and quantity_short > step_size:

        quantity=quantity_short

        if (quantity - (quantity % step_size)) > step_size:

            quantity= math.floor(quantity * 1e5) / 1e5
            side='SELL'
            trade_approved=True
            num_approved_trades+=1
            profit__per_trade_predicted = potential_profit_short
            profit_per_trade_predicted_list.append(profit__per_trade_predicted)
            print("tading now...")

        else:
            quantity=0

    else:
        side='None'
        quantity=0
        trade_approved=False

    if quantity != 0 and quantity < 1e-4:
        quantity=1e-4


    if side == 'BUY' or side == 'SELL':
        # print(f"quantitiy divisible? ", quantity % step_size == 0)
        print(f'{"trade approved:":<30}{trade_approved}')
        print(f'{"side:":<30}{side}')
        print(f'{"predicted price: $":<30}{predicted_price}')
        print(f'{"current price: $":<30}{current_price}')
        print(f'{"price gap %:":<30}{price_gap_percent}')
        print(f'{"quantity (in BTC):":<30}{quantity}')
        print(f'{"potential profit long: $":<30}{potential_profit_long}')
        print(f'{"potential profit short: $":<30}{potential_profit_short}')
        print(f'{"number of approved trades:":<30}{num_approved_trades}')
        print("----------------------------------------------------")

    return trade_approved, side, quantity

prediction = []

profit_per_trade_predicted_list = [] # profit_predicted_list is the storage of the profit of each trade over time
profit_per_trade_predicted = 0 # profit_predicted is the profit of the current trade
profit_per_trade_actual_list = [] # profit_per_trade_list is the profit of each trade
profit_per_trade_actual = 0 # profit_per_trade is the profit of the current trade
paper_account_profit_list = [] # paper_account_profit_list is the accumulated profit of the paper account
paper_account_profit=0 # paper_account_profit is the profit of the paper account at the current time step


# real account trading
df = fetch_live_market_data(n_steps+1) 
current_price = float(df.iloc[-1]['Close'])
BTC_balance=get_free_btc()
USDT_balance=get_free_usdt()
main_account_balance_list = [] # main_account_balance_list is the accumulated balance of the main account
main_account_balance = BTC_balance*current_price+USDT_balance # main_account_balance is the balance of the main account at the current time step

print("----------------------------------------------------")
print("Information of the account:")
print("BTC_balance: {}".format(BTC_balance))
print("USDT_balance: {}".format(USDT_balance))
print("main_account_balance: {}".format(main_account_balance))
print("Local Time",datetime.now())
print("----------------------------------------------------")

while True:

    df = fetch_live_market_data(n_steps+1) 
    df = df.sort_values('Open Time')
    
    current_price = float(df.iloc[-1]['Close'])

    scaled_data = df.values

    X = []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i])
    X = np.array(X)

    n_features = X.shape[2]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    n_input = X.shape[1] * X.shape[2]
    X_flat = X.reshape((X.shape[0], n_input))

    predicted_price = float(model.predict(X_flat)[0])

    trade_approved, side, quantity = trade_evaluation()

    if trade_approved:
        
        if paper_trading==True:
            
            print("paper trading...")

        elif paper_trading==False:

            if side=='SELL':

                print("executing Sell-Buy tade...")

                print("sending SELL order...")

                data = {
                    "symbol": symbol,
                    "side": side,
                    "type": type,
                    "quantity": quantity,
                    "timestamp": int(round(time.time() * 1000))
                }

                # print value of quantity key in data dictionary and its type:
                print("quantity: {}".format(data["quantity"]))
                # print("quantity: {} ({})".format(data["quantity"], type(data["quantity"])))

                result = binanceus_request(uri_path_trading, data, api_key, secret_key)
                print("POST {}: {}".format(uri_path_trading, result))
                result_json = json.loads(result)
                sold_price = result_json["fills"][0]["price"]
                sold_price = float(sold_price)

                print("SELL order sent")


                # waiting 1 minutes
                print("waiting 1 minutes...")
                time.sleep(delta_t_trading)


                print("sending BUY order...")

                side = 'BUY' # IMPORTANT

                data = {
                    "symbol": symbol,
                    "side": side,
                    "type": type,
                    "quantity": quantity,
                    "timestamp": int(round(time.time() * 1000))
                }

                result = binanceus_request(uri_path_trading, data, api_key, secret_key)
                print("POST {}: {}".format(uri_path_trading, result))
                result_json = json.loads(result)
                SecTrade_bought_price = result_json["fills"][0]["price"]
                SecTrade_bought_price = float(SecTrade_bought_price)

                print("BUY order sent")

                side = 'NONE' # IMPORTANT

                print("executed Sell-Buy trade.")

                profit__per_trade_actual = (sold_price - SecTrade_bought_price)*quantity
                profit_per_trade_actual_list.append(profit__per_trade_actual)

                print("profit of the trade: $", profit__per_trade_actual)

                # stop loss
                if profit__per_trade_actual < -stop_loss:
                    print("stop loss triggered, waiting 10 mintues...")
                    time.sleep(wait_time_steps)
                    continue  



            elif side=='BUY':

                print("executing Buy-Sell tade...")

                print("sending BUY order...")

                uri_path_trading = "/api/v3/order"
                data = {
                    "symbol": symbol,
                    "side": side,
                    "type": type,
                    "quantity": quantity,
                    "timestamp": int(round(time.time() * 1000))
                }

                # print value of quantity key in data dictionary and its type:
                print("quantity: {}".format(data["quantity"]))
                # print("quantity: {} ({})".format(data["quantity"], type(data["quantity"])))

                result = binanceus_request(uri_path_trading, data, api_key, secret_key)
                print("POST {}: {}".format(uri_path_trading, result))
                result_json = json.loads(result)
                purchased_price = result_json["fills"][0]["price"]
                purchased_price = float(purchased_price)

                print("BUY order sent")


                # waiting 1 minutes
                print("waiting 1 minutes...")
                time.sleep(delta_t_trading)


                print("sending SELL order...")

                side = 'SELL' # IMPORTANT

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
                result_json = json.loads(result)
                SecTrade_sold_price = result_json["fills"][0]["price"]
                SecTrade_sold_price = float(SecTrade_sold_price)

                print("SELL order sent")

                side = 'NONE' # IMPORTANT

                print("executed Buy-Sell trade.")

                profit__per_trade_actual = (SecTrade_sold_price - purchased_price)*quantity
                profit_per_trade_actual_list.append(profit__per_trade_actual)

                print("profit of this trade:", profit__per_trade_actual)

                # stop loss
                if profit__per_trade_actual < -stop_loss:
                    print("stop loss triggered, waiting 10 mintues...")
                    time.sleep(wait_time_steps)
                    continue  


        # fetching balance
        BTC_balance=get_free_btc()
        USDT_balance=get_free_usdt()
        main_account_balance = BTC_balance*current_price+USDT_balance

        # updating data
        paper_account_profit_list.append(paper_account_profit)
        profit_per_trade_actual_list.append(profit_per_trade_actual)
        main_account_balance_list.append(main_account_balance)

        # printing data
        print("----------------------------------------------------")
        print("Information of the account:")
        print("BTC_balance: {}".format(BTC_balance))
        print("USDT_balance: {}".format(USDT_balance))
        print("main_account_balance: {}".format(main_account_balance))
        print("Local Time",datetime.now())
        print("----------------------------------------------------")

        # saving data
        profit_account_actual_df = pd.DataFrame(paper_account_profit_list, columns=['Profit account actual'])
        profit_per_trade_predicted_df = pd.DataFrame(profit_per_trade_predicted_list, columns=['Profit per tradepredicted'])
        profit_per_trade_actual_df = pd.DataFrame(profit_per_trade_actual_list, columns=['Profit per Trade Actual'])
        main_account_balance_df = pd.DataFrame(main_account_balance_list, columns=['Profit per trade actual'])
        profit_account_actual_df.to_csv('Deployed_profit_account_actual.csv', index=False)
        profit_per_trade_predicted_df.to_csv('Deployed_profit_per_trade_predicted.csv', index=False)
        profit_per_trade_actual_df.to_csv('Deployed_profit_per_trade_actual.csv', index=False)
        main_account_balance_df.to_csv('Deployed_main_account_balance.csv', index=False)



    time.sleep(1)    


