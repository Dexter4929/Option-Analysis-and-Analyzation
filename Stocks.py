import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Function to fetch options data for a given stock ticker
def fetch_option_data(ticker):
    stock = yf.Ticker(ticker)
    expirations = stock.options[:7]  # Select the first 7 expiration dates
    all_otm_calls = []
    all_otm_puts = []

    for expiration in expirations:
        opt_chain = stock.option_chain(expiration)
        calls = opt_chain.calls
        puts = opt_chain.puts
        current_price = stock.history(period="1d")['Close'][0]

        # Filter and select top 5 OTM calls
        otm_calls = calls[calls['strike'] > current_price].nlargest(5, 'volume')

        # Filter and select top 5 OTM puts, starting Underneath the current price.
        otm_puts = puts[puts['strike'] < current_price].nlargest(5, 'volume')
        
        # Add expiration date to the DataFrame
        otm_calls['expiration'] = expiration
        otm_puts['expiration'] = expiration

        all_otm_calls.append(otm_calls)
        all_otm_puts.append(otm_puts)
    
    combined_otm_calls = pd.concat(all_otm_calls)
    combined_otm_puts = pd.concat(all_otm_puts)

    return combined_otm_calls, combined_otm_puts, expirations, current_price

# Function to create the combined plots
def create_combined_plots(ticker, expirations, current_price, combined_otm_calls, combined_otm_puts):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # 2 rows, 2 columns

    # Plot 1: Net Call and Put Option Volume
    data = yf.Ticker(ticker)
    call_volumes, put_volumes, open_interest = [], [], []

    for expiration in expirations:
        calls = data.option_chain(expiration).calls
        puts = data.option_chain(expiration).puts
        call_volumes.append(calls['volume'].sum())
        put_volumes.append(puts['volume'].sum())
        open_interest.append(calls['openInterest'].sum() + puts['openInterest'].sum())

    labels = expirations
    axs[0, 0].bar(labels, open_interest, color='lightgray', alpha=0.5, label='Open Interest', zorder=1)
    axs[0, 0].plot(labels, call_volumes, marker='o', label='Net Call Volume', color='green', zorder=2)
    axs[0, 0].plot(labels, put_volumes, marker='o', label='Net Put Volume', color='red', zorder=2)
    axs[0, 0].set_xticks(range(len(labels)))
    axs[0, 0].set_xticklabels(labels, rotation=45)
    axs[0, 0].legend(loc="upper left")

    # Plot 2: Bar Chart for Call Volume and Open Interest
    sns.set(style="whitegrid")
    selected_calls = combined_otm_calls[['strike', 'volume', 'openInterest']]
    selected_calls = selected_calls.nlargest(5, 'volume')
    data_to_plot_calls = pd.melt(selected_calls, id_vars=['strike'], var_name='Metric', value_name='Value')
    
    sns.barplot(x='strike', y='Value', hue='Metric', data=data_to_plot_calls, palette="muted", ax=axs[1, 0])
    axs[1, 0].set_xlabel('Strike Price')
    axs[1, 0].legend(loc="upper right")  # Show legend
    axs[1, 0].tick_params(axis='y', labelsize=10)  # Show y-ticks

    # Plot 3: Bubble Plot for Top OTM Call Volumes
    bubble = sns.scatterplot(x='expiration', y='strike', size='volume', sizes=(50, 1000),
                             hue='volume', palette='coolwarm', data=combined_otm_calls, ax=axs[0, 1], legend=False)
    axs[0, 1].set_xlabel('Expiration Date')
    axs[0, 1].set_ylabel('Strike Price')
    axs[0, 1].tick_params(axis='x', rotation=45)

    # Plot 4: Bar Chart for Put Volume and Open Interest (Top 5 OTM Puts)
    selected_puts = combined_otm_puts[['strike', 'volume', 'openInterest']]
    selected_puts = selected_puts.nlargest(5, 'volume')  # Ensure only top 5 are displayed

    data_to_plot_puts = pd.melt(selected_puts, id_vars=['strike'], var_name='Metric', value_name='Value')

    sns.barplot(x='strike', y='Value', hue='Metric', data=data_to_plot_puts, palette="muted", ax=axs[1, 1])
    axs[1, 1].set_xlabel('Strike Price')
    axs[1, 1].legend(loc="upper right")  # Show legend
    axs[1, 1].tick_params(axis='y', labelsize=10)  # Show y-ticks

    plt.tight_layout()
    plt.show()

# Main function to execute the process
def main():
    ticker = input("Enter the stock ticker: ")
    combined_otm_calls, combined_otm_puts, expirations, current_price = fetch_option_data(ticker)
    create_combined_plots(ticker, expirations, current_price, combined_otm_calls, combined_otm_puts)

# Execute the main function
if __name__ == "__main__":
    main()
