# Trading-Software
<img width="1298" alt="image" src="https://github.com/FaridSoroush/Trading-Software/assets/45682607/b11a8aa1-d32c-472a-929a-fbe583a44b83">

![Example Image](Plots/Screenshot 2024-01-13 at 11.43.29 AM.png)

![Example Image](Plots/Screenshot 2024-01-13 at 11.45.55 AM.png)

![Example Image](Plots/Screenshot 2024-01-13 at 11.48.50 AM.png)

![Example Image](Plots/Screenshot 2024-01-13 at 11.52.06 AM.png)

![Example Image](Plots/all_orders_last_run.png)

![Example Image](Plots/filled_orders_short_term.png)

![Example Image](Plots/indicators.png)

![Example Image](Plots/order_history.png)

![Example Image](Plots/paper account profit 1.png)

![Example Image](Plots/paper account profit 2.png)

![Example Image](Plots/paper account profit 3.png)

![Example Image](Plots/prediction_local.png)

![Example Image](Plots/prediction_local_and_long_term.png)

![Example Image](Plots/prediction_long_term.png)

![Example Image](Plots/short_term_orders.png)


# Data
 - BTC, USDT (1:1 with USD)
 - 6 years of one-every-minute datapoints from Binance
 - Each datapoint contains high, low, open, close, volumn
 - Each X consists of 10 minute datapoints
 - Each y is the close price of the next minute

# Model
- Linear layers + nn.transformer + linear layers

# Training
- Data is split into first 80\% training, last 20\% validation and test, no shuffling
- Batch size 128, learning rate 0.005
- Every 5 epochs, evaluate with validation set and save current best model weights, test at the end
  
# Execution
- Pull real time data every seconds (empirically found that per minute data changes within a minute)
- Do inference on most recent 10 minutes data
- If next minute predicted price is profitable (after commision fee), buy/sell part of account balance (1\%)
