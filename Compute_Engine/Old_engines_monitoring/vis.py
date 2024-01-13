import pandas as pd
import matplotlib.pyplot as plt
import time

plt.ion()

while True:
    # load the data
    main_account_balance_df = pd.read_csv('Deployed_main_account_balance.csv')

    # plot the data
    plt.figure(figsize=(15,10))

    plt.plot(main_account_balance_df)
    plt.title('Main Account Balance')

    # Calculate min and max, expanded by 20%
    y_min = main_account_balance_df.iloc[:, 0].min()
    y_max = main_account_balance_df.iloc[:, 0].max()
    y_range = y_max - y_min
    plt.ylim(y_min - 0.2*y_range, y_max + 0.2*y_range)

    # Format y-axis tick labels
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)
    
    print("plots updated at " + time.strftime("%H:%M:%S"))
    
    # Clear the plot for the next draw
    plt.clf()
    
    time.sleep(1)
