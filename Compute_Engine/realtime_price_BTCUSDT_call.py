import ccxt

# Initialize the ccxt exchange
exchange = ccxt.binanceus()
symbol = 'BTC/USDT'

# Function to fetch and update the Bitcoin price
def get_BTCUSDT_realtime_price():
    bars = exchange.fetch_ohlcv(symbol, '1m', limit=2)
    price = bars[-1][4]  # Use the close price
    return price

# Test call to get_btc_price
if __name__ == "__main__":
    print(get_btc_price())
