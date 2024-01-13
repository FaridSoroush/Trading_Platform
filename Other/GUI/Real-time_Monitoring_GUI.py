import ccxt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from threading import Thread

# Initialize the ccxt exchange
exchange = ccxt.binanceus()
symbol = 'BTC/USD'

# Create a Tkinter window
window = tk.Tk()
window.title("Real-Time Bitcoin Price")
window.geometry("800x600")

# Create a figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line, = ax.plot([], [], label='Bitcoin Price')

# Define the x and y data for the plot
x_data = []
y_data = []

# Variables for zooming
zoom_factor = 1.2
base_scale = 2.0

# Function to fetch and update the Bitcoin price
def update_price(frame_data):
    bars = exchange.fetch_ohlcv(symbol, '1m', limit=2)
    price = bars[-1][4]  # Use the close price
    x_data.append(len(x_data) + 1)
    y_data.append(price)
    line.set_data(x_data, y_data)

    # Adjust the plot axis range
    ax.relim()
    ax.autoscale_view()

    ax.set_title(f"Bitcoin Price: {price} USD")
    fig.canvas.draw()

# Function to handle mouse scroll event for zooming
def scroll(event):
    if event.button == 'up':
        scale_factor = 1 / zoom_factor
    else:
        scale_factor = zoom_factor

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xwidth = xmax - xmin
    ywidth = ymax - ymin

    xcenter = (xmin + xmax) / 2
    ycenter = (ymin + ymax) / 2

    new_xwidth = xwidth * scale_factor
    new_ywidth = ywidth * scale_factor

    new_xmin = xcenter - new_xwidth / 2
    new_xmax = xcenter + new_xwidth / 2
    new_ymin = ycenter - new_ywidth / 2
    new_ymax = ycenter + new_ywidth / 2

    ax.set_xlim(new_xmin, new_xmax)
    ax.set_ylim(new_ymin, new_ymax)
    fig.canvas.draw()

# Function to start monitoring the Bitcoin price
def start_monitoring():
    ani = animation.FuncAnimation(fig, update_price, interval=500)
    fig.canvas.mpl_connect('scroll_event', scroll)
    canvas.draw()

# Function to stop the program
def stop_program():
    window.quit()

# Create a thread for monitoring the Bitcoin price
monitor_thread = Thread(target=start_monitoring)

# Create a canvas for embedding the plot in the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().pack()

# Create Start and Stop buttons
start_button = tk.Button(window, text="Start Monitoring", command=monitor_thread.start)
start_button.pack()

stop_button = tk.Button(window, text="Stop Program", command=stop_program)
stop_button.pack()

# Start the Tkinter event loop
window.mainloop()
