# Trading-Software

* **Main components of the system:**

<img src="image/README/1705182855841.png" alt="Example Image" width="850"/>

* **Examples of technical trading indicators:**

<img src="image/README/indicators.png" alt="Example Image" width="850"/>

* **Example of modeling result different time frames:**

<img src="image/README/prediction_local_and_long_term.png" alt="Example Image" width="850"/>

<img src="image/README/prediction_long_term.png" alt="Example Image" width="850"/>

* **Result of trading simulation:**

<img src="image/README/paper_account_profit_1.png" alt="Example Image" width="850"/>

<img src="image/README/paper_account_profit_2.png" alt="Example Image" width="850"/>

<img src="image/README/paper_account_profit_3.png" alt="Example Image" width="850"/>

<img src="image/README/simulation_result.png" alt="Example Image" width="850"/>


* **Examples of GUI for history of orders (all orders and filled orders here):**

<img src="image/README/all_orders_last_run.png" alt="Example Image" width="850"/>

<img src="image/README/filled_orders_short_term.png" alt="Example Image" width="850"/>

* **Example of GUI for live trading:**

<img src="image/README/GUI.png" alt="Example Image" width="850"/>

<img src="image/README/GUI1.png" alt="Example Image" width="850"/>

<img src="image/README/GUI3.png" alt="Example Image" width="850"/>

# Data

- BTC, USDT (1:1 with USD)
- 6 years of one-every-minute datapoints from Binance
- Each datapoint contains high, low, open, close, volumn
- Each X consists of 10 minute datapoints
- Each y is the close price of the next minute

# Model

- Linear layers + nn.transformer + linear layers
- Gradient Boosting
- LSTM

# Training

- Data is split into first 80\% training, last 20\% validation and test, no shuffling
- Batch size 128, learning rate 0.005
- Every 5 epochs, evaluate with validation set and save current best model weights, test at the end

# Execution

- Pull real time data every seconds (empirically found that per minute data changes within a minute)
- Do inference on most recent 10 minutes data
- If next minute predicted price is profitable (after commision fee), buy/sell part of account balance (1\%)
