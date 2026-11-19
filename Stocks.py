import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# Function to fetch options data and today's intraday price data
def fetch_option_data(ticker):
    stock = yf.Ticker(ticker)
    expirations = stock.options[:7]  # Select the first 7 expiration dates
    all_calls = []
    all_puts = []

    # Fetch today's intraday price data (5-minute intervals)
    intraday = stock.history(period="1d", interval="5m")
    if intraday.empty:
        # Fallback to most recent trading day if market hasn't opened yet
        intraday = stock.history(period="5d", interval="15m")
        last_day = intraday.index[-1].date()
        intraday = intraday[intraday.index.date == last_day]

    current_price = float(intraday['Close'].iloc[-1])

    for expiration in expirations:
        opt_chain = stock.option_chain(expiration)
        calls = opt_chain.calls.copy()
        puts = opt_chain.puts.copy()

        calls['expiration'] = expiration
        puts['expiration'] = expiration

        all_calls.append(calls)
        all_puts.append(puts)

    combined_calls = pd.concat(all_calls, ignore_index=True)
    combined_puts = pd.concat(all_puts, ignore_index=True)

    return combined_calls, combined_puts, expirations, current_price, intraday

# Helper to plot Top 10 Strikes with yellow dashed spot price line
def plot_top_strikes(ax, df, current_price, title, bar_palette):
    agg = (
        df.fillna({'volume': 0, 'openInterest': 0})
        .groupby('strike', as_index=False)[['volume', 'openInterest']].sum()
        .nlargest(10, 'volume')
        .sort_values('strike')
        .reset_index(drop=True)
    )

    strikes_num = agg['strike'].values
    agg['strike_label'] = [f"{s:g}" for s in strikes_num]
    data_to_plot = pd.melt(agg, id_vars=['strike_label'], value_vars=['volume', 'openInterest'], var_name='Metric', value_name='Value')

    sns.barplot(
        x='strike_label', y='Value', hue='Metric',
        data=data_to_plot, palette=bar_palette, ax=ax
    )

    # Interpolate exact x-position of current_price across the sorted strike ticks
    x_indices = np.arange(len(strikes_num))
    spot_x = np.interp(current_price, strikes_num, x_indices) if len(strikes_num) > 1 else 0.0

    ax.axvline(
        spot_x, color='#eab308', linestyle='--', linewidth=2.2,
        label=f'Spot (${current_price:.2f})', zorder=5
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Metric')
    ax.grid(True, linestyle='--', alpha=0.5)

# Function to create the combined plots
def create_combined_plots(ticker, expirations, current_price, intraday, combined_calls, combined_puts):
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))

    data = yf.Ticker(ticker)
    call_volumes, put_volumes, open_interest = [], [], []

    for expiration in expirations:
        calls = data.option_chain(expiration).calls.fillna(0)
        puts = data.option_chain(expiration).puts.fillna(0)
        call_volumes.append(calls['volume'].sum())
        put_volumes.append(puts['volume'].sum())
        open_interest.append(calls['openInterest'].sum() + puts['openInterest'].sum())

    labels = expirations

    # --- PLOT 1 (Top Left): Open Interest vs Option Volume ---
    ax1 = axs[0, 0]
    ax2 = ax1.twinx()

    ax1.bar(labels, open_interest, color='lightgray', alpha=0.5, label='Open Interest')
    ax2.plot(labels, call_volumes, marker='o', label='Call Volume', color='green')
    ax2.plot(labels, put_volumes, marker='o', label='Put Volume', color='red')

    ax1.set_ylabel('Open Interest', fontsize=10)
    ax2.set_ylabel('Option Volume', fontsize=10)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_title('Open Interest vs Option Volume by Expiration', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.grid(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left')

    bar_palette = {'volume': '#111827', 'openInterest': '#9ca3af'}

    # --- PLOT 2 (Bottom Left): Top 10 Call Strikes ---
    plot_top_strikes(axs[1, 0], combined_calls, current_price, f'Top 10 Call Strikes ({ticker.upper()})', bar_palette)

    # --- PLOT 3 (Top Right): TODAY'S PRICE VS NORMALIZED (CALLS - PUTS) ---
    ax_p3_price = axs[0, 1]
    ax_p3_sent = ax_p3_price.twinx()

    front_chain = data.option_chain(expirations[0])
    total_calls = front_chain.calls['volume'].fillna(0).sum()
    total_puts = front_chain.puts['volume'].fillna(0).sum()
    total_vol = max(total_calls + total_puts, 1)

    net_ratio = (total_calls - total_puts) / total_vol

    prices = intraday['Close'].values
    vols = intraday['Volume'].replace(0, 1).values
    x = np.arange(len(prices))

    price_dev = (prices - prices[0]) / max(prices.std(), 1e-6)
    vol_weight = vols / vols.mean()
    raw_sentiment = net_ratio + 0.25 * np.tanh(price_dev * vol_weight)
    norm_sentiment = np.clip(raw_sentiment / max(np.max(np.abs(raw_sentiment)), 1.0), -1.0, 1.0)

    time_labels = intraday.index.strftime('%H:%M')
    ax_p3_price.plot(x, prices, color='#1f2937', linewidth=2, label='Stock Price', zorder=3)
    ax_p3_price.set_ylabel('Stock Price', fontsize=10, color='black')
    ax_p3_price.set_xlabel('Time (Today)', fontsize=10, color='black')
    ax_p3_price.grid(True, linestyle='--', alpha=0.6)

    tick_idx = np.linspace(0, len(x) - 1, min(7, len(x)), dtype=int)
    ax_p3_price.set_xticks(tick_idx)
    ax_p3_price.set_xticklabels(time_labels[tick_idx], rotation=45)

    x_dense, y_dense = [], []
    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = norm_sentiment[i], norm_sentiment[i + 1]
        x_dense.append(x0)
        y_dense.append(y0)
        if (y0 > 0 and y1 < 0) or (y0 < 0 and y1 > 0):
            t = -y0 / (y1 - y0)
            x_dense.append(x0 + t * (x1 - x0))
            y_dense.append(0.0)
    x_dense.append(x[-1])
    y_dense.append(norm_sentiment[-1])

    points = np.array([x_dense, y_dense]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_midpoints = 0.5 * (np.array(y_dense[:-1]) + np.array(y_dense[1:]))
    seg_colors = ['green' if val >= 0 else 'red' for val in seg_midpoints]

    lc = LineCollection(segments, colors=seg_colors, linewidths=2.2, zorder=4)
    ax_p3_sent.add_collection(lc)
    ax_p3_sent.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.8)
    ax_p3_sent.set_ylim(-1.1, 1.1)
    ax_p3_sent.set_ylabel('Normalized (Calls - Puts)', fontsize=10, color='black')
    ax_p3_sent.grid(False)

    ax_p3_price.set_title(f"Today's Price vs Net Sentiment ({ticker.upper()})", fontsize=12)

    legend_handles = [
        Line2D([0], [0], color='#1f2937', lw=2, label='Stock Price'),
        Line2D([0], [0], color='green', lw=2, label='Calls > Puts (> 0)'),
        Line2D([0], [0], color='red', lw=2, label='Puts > Calls (< 0)')
    ]
    ax_p3_price.legend(handles=legend_handles, loc='upper left', fontsize=9)

    # --- PLOT 4 (Bottom Right): Top 10 Put Strikes ---
    plot_top_strikes(axs[1, 1], combined_puts, current_price, f'Top 10 Put Strikes ({ticker.upper()})', bar_palette)

    plt.tight_layout()
    plt.show()

# Main function
def main():
    ticker = input("Enter the stock ticker: ")
    combined_calls, combined_puts, expirations, current_price, intraday = fetch_option_data(ticker)
    create_combined_plots(ticker, expirations, current_price, intraday, combined_calls, combined_puts)

if __name__ == "__main__":
    main()
