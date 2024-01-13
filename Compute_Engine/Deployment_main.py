# model GB 2M3F, only the last 5K, with stop loss $0.05 n 30min wait, expoential growth, 5% position ratio

import pandas as pd
import numpy as np
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
import csv
# personal modules
from realtime_price_BTCUSDT_call import get_BTCUSDT_realtime_price
from counter_num_open_positions import count_num_open_positions
from Get_balance import get_free_btc, get_free_usd, get_free_usdt, get_locked_btc, get_locked_usd, get_locked_usdt, get_total_btc, get_total_usd, get_total_usdt
import subprocess

# # Command to open a new terminal window, activate the environment, and run the script
# command = 'tell application "Terminal" to do script "source activate myenv; python3 /Users/faridsoroush/Documents/GitHub/Trading-Software/Compute_Engine/Cancel_6sigma_open_orders.py"'
# # Use osascript with -e option to execute the AppleScript command
# subprocess.Popen(['osascript', '-e', command])


# record the start time of the code
start_time_code = datetime.now()
# record it into a csv file
with open('start_time.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([start_time_code])

account_checking_counter = 0

stop_loss = 0.05 # in US dollars # stop loss

wait_time_steps = 10 * 60 # in seconds # wait timer after stop loss is triggered

slippage = 0.01635/100   # 0.0163% slippage (mean n median measured by me was 0.02578% 
                        # and 0.01635%, max was 0.082%) 

step_size = 0.00001     # this value is from Binance API documentation but I think it is wrong. 
                        # I think it actually is 1e-5 (added an if at the end of the trade_eval 
                        # function to account for this)
n_steps = 30
predicting_index = 1

# Exchange API
api_key = 'LD5eq9nHMz51lvNxA4Yk4Npm02kS09oZU7tdvvkQKpcmLhdoCmUPKEv9xebfHavB'
secret_key = 'DtWIdtZRbtBDzzZPFVVgH39nr3VTNhz7njsb3HHGrYoHJcVl5fB1Jpdc5HoJJTfo'
api_url = "https://api.binance.us"
uri_path = "/api/v3/account"
uri_path_market = "/api/v3/klines"
uri_path_trading = "/api/v3/order"
data={'timestamp': int(round(time.time() * 1000))}
max_num_open_positions = 200

# trading parameters
paper_trading=False # IMPORTANT
symbol = 'BTCUSDT'
Exchange_Comission= 0
position_ratio_long=0.05
position_ratio_short=0.05

quantity= 0 
type="MARKET"
trade_approved = False
num_approved_trades=0
trade_comission=0
global initial_run
initial_run = True

prediction = []

profit_per_trade_predicted_list = [] 
profit_per_trade_predicted = 0 

executed_per_trade_price=0
executed_per_trade_price_list=[]
current_per_trade_price=0
current_per_trade_price_list=[]

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
    df['Open Time'] = df['Open Time'].values.astype(np.int64) // 10 ** 3
    df = df.sort_values('Open Time')

    return df

def trade_evaluation():

    global num_approved_trades
    global BTC_balance
    global USDT_balance
    global quantity
    global side
    global trade_approved
    global account_checking_counter
    
    quantity=0.0
    side='NONE'
    trade_approved=False
    BTC_balance=get_free_btc()
    USDT_balance=get_free_usdt()

    quantity_long=position_ratio_long*USDT_balance/current_price
    quantity_short=position_ratio_short*BTC_balance

    price_gap_percent=abs((predicted_price-current_price)/current_price*100)

    # long and short profit considering slippage:
    potential_profit_long = (predicted_price-current_price)*quantity_long*(1-slippage)
    potential_profit_short = -(predicted_price-current_price)*quantity_short*(1-slippage)

    # checking for long trade:
    if potential_profit_long > 0 and \
        quantity_long > step_size and \
        USDT_balance > current_price*quantity_long:

        print("thinking about long trade...")

        quantity=quantity_long

        print("USDT balance more than quantity*current_price: ", USDT_balance > current_price*quantity_long)
        print("quantity residue more than step size: ", (quantity - (quantity % step_size)) > step_size)

        if (quantity - (quantity % step_size)) > step_size and USDT_balance > current_price*quantity:

            print("quantity in trade_eval on buy side: ", quantity)
            print("USDT balance in trade_eval on buy side: ", USDT_balance)
            quantity= math.floor(quantity * 1e5) / 1e5
            side='BUY'
            trade_approved=True
            num_approved_trades+=1
            profit__per_trade_predicted = potential_profit_long
            profit_per_trade_predicted_list.append(profit__per_trade_predicted)
            print("tading Long now...")

        else:
            print("not enough USDT balance or quantity residue less than step size")
            quantity=0
            side='NONE'
            trade_approved=False

    # checking for short trade
    elif potential_profit_short > 0 and \
        quantity_short > step_size and \
        BTC_balance > quantity_short:

        print("thinking about short trade...")

        quantity=quantity_short

        print("BTC balance more than quantity: ", BTC_balance > quantity)
        print("quantity residue more than step size: ", (quantity - (quantity % step_size)) > step_size)

        if (quantity - (quantity % step_size)) > step_size and BTC_balance > quantity:

            print("quantity in trade_eval on sell side: ", quantity)
            print("BTC balance in trade_eval on sell side: ", BTC_balance)
            quantity= math.floor(quantity * 1e5) / 1e5
            side='SELL'
            trade_approved=True
            num_approved_trades+=1
            profit__per_trade_predicted = potential_profit_short
            profit_per_trade_predicted_list.append(profit__per_trade_predicted)
            print("tading Short now...")

##############################################################################################################
        # else:
        #     print("not enough BTC balance or quantity residue less than step size")
        #     quantity=0
        #     side='NONE'
        #     trade_approved=False


    # if you turn the lower and upper block (elses) OFF, it's gonna TRADE NO MATTER
    # WHAT (the next ifs block) until quantitiy of the position ratio*asset reaches
    # step_size (at the moment step_size=1e-5, pracitcally zero). For example, if
    # the quantity of the position ratio*asset is 1e-6, it's gonna pass BUY/SELL with
    # quantity of 1e-6 in the upper if conditions, and then change it to 1e-4 and trade
    # with quantitiy=1e-4 in the lower block.
    # (this is useful for measuring slippage, general logic of trading, testing models, etc. 
    # since it just lets you trade no matter what)

    # When it's ON, it's gonna check if the quantity of the (potentially 
    # profitable) trade is greater than the minimum trade size (step_size), so 
    # it only trades if there's a potential profit & the quantity is greater, so something
    # like quantity=1e-6 is not gonna be traded, but quantity=1e-3 is gonna be traded.

    # else:
        # print("no trade opportunity because of insufficient balance or predicted price is too close to current price")
        # side='NONE'
        # quantity=0
        # trade_approved=False
##############################################################################################################

    # TRADING NO MATTER WHAT
    if quantity != 0 and quantity < 1e-4 and (side == 'BUY' or side == 'SELL'):  
        print("quantity is less than 1e-4 and side is BUY or SELL")

        if BTC_balance > 1e-4 and side=='SELL': # if the balance is greater than the minimum trade size
            quantity=1e-4
            print("BTC balance more than 1e-4 and side is SELL")

        elif USDT_balance > current_price*1e-4 and side=='BUY': # if the balance is greater than the minimum trade size
            quantity=1e-4
            print("USDT balance more than current_price*1e-4 and side is BUY")

        else:
            quantity=0
            side = 'NONE'
            trade_approved=False
            print("not enough balance for 1e-4 BTC trade size")


    # total_account_balance = get_total_usdt() + get_total_btc()*current_price
    # if side == 'BUY' or side == 'SELL':
    # print(f"quantitiy divisible? ", quantity % step_size == 0)
    print(f'{"trade approved:":<30}{trade_approved}')
    print(f'{"side:":<30}{side}')
    print(f'{"predicted price: $":<30}{predicted_price}')
    print(f'{"current price: $":<30}{current_price}')
    print(f'{"price gap %:":<30}{price_gap_percent}')
    print(f'{"quantity (in BTC):":<30}{quantity}')
    print(f'{"quantity long:":<30}{quantity_long}')
    print(f'{"quantity short:":<30}{quantity_short}')
    print(f'{"potential profit long: $":<30}{potential_profit_long}')
    print(f'{"potential profit short: $":<30}{potential_profit_short}')
    print(f'{"number of approved trades:":<30}{num_approved_trades}')
    print(f'{"BTC balance: $":<30}{BTC_balance}')
    print(f'{"USDT balance: $":<30}{USDT_balance}')
    # print(f'{"free account balance: $":<30}{main_account_balance}')
    # print(f'{"total account balance: $":<30}{total_account_balance}')
    account_checking_counter += 1
    print("----------------------------------------------------")

    return trade_approved, side, quantity

# real account trading
df = fetch_live_market_data(n_steps+1) 
current_price = float(df.iloc[-1]['Close'])
BTC_balance=get_free_btc()
USDT_balance=get_free_usdt()
main_account_balance_list = [] # main_account_balance_list is the accumulated balance of the main account
main_account_balance = BTC_balance*current_price+USDT_balance # main_account_balance is the balance of the main account at the current time step

# record BTC price at start time of the code
BTC_price_at_start_time = current_price
BTC_balance_at_start_time = get_total_btc()
USDT_balance_at_start_time = get_total_usdt()

# record it into a csv file
with open('BTC_price_n_Account_Balance_n_start_time.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(["BTC_price_at_start_time", "BTC_balance_at_start_time","USDT_balance_at_start_time", "start_time_code"])
    writer.writerow([BTC_price_at_start_time, BTC_balance_at_start_time, USDT_balance_at_start_time, start_time_code])

print("----------------------------------------------------")
print("Information of the account:")
print("BTC_balance: {}".format(BTC_balance))
print("USDT_balance: {}".format(USDT_balance))
print("main_account_balance: {}".format(main_account_balance))
print("Local Time",datetime.now())
print("----------------------------------------------------")

while True:

    now = time.time()
    match_unix_wait_time = 60 - (now % 60)
    print("time to match UNIX time: ", match_unix_wait_time)
    time.sleep(match_unix_wait_time)

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
        
        if count_num_open_positions() <= (max_num_open_positions - 2): 

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

                print("quantity: {}".format(data["quantity"]))

                result = binanceus_request(uri_path_trading, data, api_key, secret_key)
                print("POST {}: {}".format(uri_path_trading, result))
                result_json = json.loads(result)
                sold_price = result_json["fills"][0]["price"]
                sold_price = float(sold_price)

                # measuring slippage
                current_per_trade_price = get_BTCUSDT_realtime_price()
                current_per_trade_price_list.append(current_price)
                executed_per_trade_price = sold_price
                executed_per_trade_price_list.append(executed_per_trade_price)

                print("SELL order sent")

                print("sending BUY order (LIMIT)...")

                side = 'BUY' # IMPORTANT
                type = 'LIMIT' # IMPORTANT

                # reducing margin by 20% or just sell with little profit
                if predicted_price < sold_price:
                    predicted_price = predicted_price + abs(sold_price - predicted_price) * 0.2
                else:
                    predicted_price =  sold_price * (1 - 0.0001) # 0.01% profit

                print("predicted_price: {}".format(round(predicted_price),2))

                data = {
                    "symbol": symbol,
                    "side": side,
                    "type": type,
                    "quantity": quantity,
                    "price": float(round(predicted_price,2)),
                    "timeInForce": "GTC",
                    "timestamp": int(round(time.time() * 1000))
                }

                result = binanceus_request(uri_path_trading, data, api_key, secret_key)
                print("POST {}: {}".format(uri_path_trading, result))

                print("BUY order sent (LIMIT)")

                side = 'NONE' # IMPORTANT
                type = 'MARKET' # IMPORTANT

                print("executed Sell-Buy trade.")

                # stop loss
                current_price = get_BTCUSDT_realtime_price()
                potential_per_trade_profit = (sold_price-current_price)*quantity
                if potential_per_trade_profit < -stop_loss:
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

                print("quantity: {}".format(data["quantity"]))

                result = binanceus_request(uri_path_trading, data, api_key, secret_key)
                print("POST {}: {}".format(uri_path_trading, result))
                result_json = json.loads(result)
                purchased_price = result_json["fills"][0]["price"]
                purchased_price = float(purchased_price)

                # measuring slippage
                current_per_trade_price = get_BTCUSDT_realtime_price()
                current_per_trade_price_list.append(current_price)
                executed_per_trade_price = purchased_price
                executed_per_trade_price_list.append(executed_per_trade_price)

                print("BUY order sent")

                print("sending SELL order (LIMIT)...")

                side = 'SELL' # IMPORTANT
                type = 'LIMIT' # IMPORTANT

                # reducing margin by 20% or just sell with little profit
                if predicted_price > purchased_price:
                    predicted_price = predicted_price - abs(predicted_price - purchased_price) * 0.2
                else:
                    predicted_price =  purchased_price * 1.0001 # 0.01% profit

                print("predicted_price: {}".format(round(predicted_price,2)))

                uri_path_trading = "/api/v3/order"
                data = {
                    "symbol": symbol,
                    "side": side,
                    "type": type,
                    "quantity": quantity,
                    "price": float(round(predicted_price,2)),
                    "timeInForce": "GTC",
                    "timestamp": int(round(time.time() * 1000))
                }

                result = binanceus_request(uri_path_trading, data, api_key, secret_key)
                print("POST {}: {}".format(uri_path, result))

                print("SELL order sent (LIMIT))")

                side = 'NONE' # IMPORTANT
                type = 'MARKET' # IMPORTANT

                print("executed Buy-Sell trade.")

                # stop loss
                current_price = get_BTCUSDT_realtime_price()
                potential_per_trade_profit = (current_price - purchased_price)*quantity
                if potential_per_trade_profit < -stop_loss:
                    print("stop loss triggered, waiting 10 mintues...")
                    time.sleep(wait_time_steps)
                    continue

        else:
            print("Maximum number of open-positions ({}) per symbol reached".format(max_num_open_positions))
            print("waiting 5 minutes...")
            time.sleep(60*5) # waiting 5 minutes
