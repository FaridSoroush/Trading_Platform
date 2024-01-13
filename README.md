# Trading-Software
<img width="1298" alt="image" src="https://github.com/FaridSoroush/Trading-Software/assets/45682607/b11a8aa1-d32c-472a-929a-fbe583a44b83">

https://github.com/FaridSoroush/Trading-Platform/blob/main/paper%20account%20profit%202.png

# Data
 - BTC, USDT (1:1 with USD)
 - 6 years of one-every-minute datapoints from Binance
 - Each datapoint contains high, low, open, close, volumn
 - Each X consists of 10 minute datapoints
 - Each y is the close price of the next minute

TODO:
- Add moving averages (hourly, daily, weekly etc.)
- Add other coins
- Add other financial indices (e.g. S&P 500, interest rate etc.)
- Experiment with other y's (e.g. highest price in the next 10 minutes or whether there will be a profitable rise in 10 minutes)

# Model
- Linear layers + nn.transformer + linear layers

TODO:
- Experiment of nn.transformer parameter (encoder layer, decoder layer, number of heads etc.)
- Experiment with emsemble learning (weighted outputs from several models)

# Training
- Data is split into first 80\% training, last 20\% validation and test, no shuffling
- Batch size 128, learning rate 0.005
- Every 5 epochs, evaluate with validation set and save current best model weights, test at the end

TODO:
- Experiment with validation and test sets across different time periods, but be careful not to let the model see predicted value
  
# Execution
- Pull real time data every seconds (empirically found that per minute data changes within a minute)
- Do inference on most recent 10 minutes data
- If next minute predicted price is profitable (after commision fee), buy/sell part of account balance (1\%)

TODO:
- If predicting max price in next 10 minutes or whether there will be a profitable rise in 10 minutes, buy at current time and sell at future profitable price
- Keep a stack for bought-in price and amount, sell from inventory accordingly whenever profitable
