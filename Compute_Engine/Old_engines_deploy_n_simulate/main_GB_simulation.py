import pandas as pd
import numpy as np
import pandas as pd
from Get_balance import get_free_btc, get_free_usd, get_free_usdt
from joblib import dump, load
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import matplotlib.pyplot as plt

n_steps = 30

delta_t_trading = 60

historic_data_input_filename = "../Prediction_Model/LSTM/BTCUSDT4Y1MKline_cleaned_24features_nSteps"+str(n_steps)+".csv"
historic_df = pd.read_csv(historic_data_input_filename)
historic_df=historic_df[['Open Time','Close', 'Volume']]

model_parameters_file = '../Prediction_Model/GradientBoosting/GB_model_parameters_10000_1000_2_1_10_10.pkl'

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

def fetch_historic_data(start, end):
    return historic_df[start : end].copy()

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

    quantity_long=position_ratio*BTC_balance
    quantity_short=position_ratio*USDT_balance/current_price

    price_gap_percent=abs((predicted_price-current_price)/current_price*100)

    trade_comission_long=0
    trade_comission_short=0

    potential_profit_long=(predicted_price-current_price)*quantity_long
    potential_profit_short=-(predicted_price-current_price)*quantity_short

    if potential_profit_long > (trade_comission_long):
        side="BUY"
        trade_approved=True
        num_approved_trades+=1

    elif potential_profit_short> (trade_comission_short):
        side='SELL'
        trade_approved=True
        num_approved_trades+=1

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

    return trade_approved, side, quantity

prediction = []

start_point = len(historic_df) - 5000
end_point = start_point + n_steps + 1
iteration_step = 1

paper_account_profit_list = []

while end_point < len(historic_df):

    df = fetch_historic_data(start_point, end_point) 
    df=df[['Open Time','Close', 'Volume']]

    predicting_index = df.columns.get_loc('Close')

    current_price = float(df.iloc[-1]['Close'])

    scaled_data = df.values

    X = []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i])
    X = np.array(X)

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

                sold_price=float(fetch_historic_data(end_point, end_point + delta_t_trading).iloc[-1]['Close'])
                SecTrade_bought_price=float(fetch_historic_data(end_point + delta_t_trading, end_point + 2*delta_t_trading).iloc[-1]['Close'])
                paper_account_profit+=(-(SecTrade_bought_price-sold_price)*(quantity)*(1-Exchange_Comission))

            elif side=='BUY':

                purchased_price=float(fetch_historic_data(end_point, end_point + delta_t_trading).iloc[-1]['Close'])
                SecTrade_sold_price=float(fetch_historic_data(end_point + delta_t_trading, end_point + 2*delta_t_trading).iloc[-1]['Close'])
                paper_account_profit+=(+(SecTrade_sold_price-purchased_price)*(quantity)*(1-Exchange_Comission))
                
    paper_account_profit_list.append(paper_account_profit)

    start_point += iteration_step
    end_point += iteration_step
    print("\rstart_point: {}".format(start_point), end="")

plt.plot(paper_account_profit_list)
plt.title('Paper Account Profit Over Time')
plt.xlabel('Time Step')
plt.ylabel('Profit')
plt.show()

profit_df = pd.DataFrame(paper_account_profit_list, columns=['Profit'])
profit_df.to_csv('profit_over_time.csv', index=False)