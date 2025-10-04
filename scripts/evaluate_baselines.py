# evaluate_baselines.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def buy_and_hold(df, initial_balance=10000):
    """
    Simulates the Buy and Hold strategy.

    Args:
        df (pd.DataFrame): DataFrame with daily asset prices.
        initial_balance (int): The starting capital.

    Returns:
        pd.Series: A Series containing the portfolio value for each day.
    """
    print("--- Simulating Buy and Hold ---")
    n_assets = len(df.columns)

    # Invest an equal amount in each asset at the beginning
    initial_investment_per_asset = initial_balance / n_assets

    # Get the initial prices
    initial_prices = df.iloc[0]

    # Calculate the number of shares bought for each asset
    shares = initial_investment_per_asset / initial_prices

    # Calculate the portfolio value for each day
    portfolio_values = df.dot(shares)

    print(f"Initial Investment: ${initial_balance:.2f}")
    print(f"Final Portfolio Value: ${portfolio_values.iloc[-1]:.2f}")

    return portfolio_values

def equally_weighted_rebalanced(df, initial_balance=10000, rebalance_freq='M', transaction_cost_pct=0.001):
    """
    Simulates an Equally Weighted Portfolio with periodic rebalancing.

    Args:
        df (pd.DataFrame): DataFrame with daily asset prices.
        initial_balance (int): The starting capital.
        rebalance_freq (str): The rebalancing frequency ('M' for monthly, 'Q' for quarterly).
        transaction_cost_pct (float): The transaction cost as a percentage.

    Returns:
        pd.Series: A Series containing the portfolio value for each day.
    """
    print(f"\n--- Simulating Equally Weighted Portfolio (Rebalanced {rebalance_freq}) ---")
    n_assets = len(df.columns)

    # Set the initial weights to be equal
    weights = np.full(n_assets, 1/n_assets)

    portfolio_value = initial_balance
    portfolio_values = pd.Series(index=df.index)

    last_rebalance_date = None

    for date, prices in df.iterrows():
        # Store the portfolio value for the day before any changes
        portfolio_values[date] = portfolio_value

        # Determine if it's a rebalancing day
        # Rebalance on the first day of the new period (month, quarter)
        if last_rebalance_date is None or (date.month != last_rebalance_date.month and rebalance_freq == 'M'):

            # Calculate the value of trades to rebalance
            target_asset_values = portfolio_value * (1/n_assets)
            current_asset_values = weights * portfolio_value
            trades = target_asset_values - current_asset_values

            # Apply transaction costs
            transaction_costs = np.sum(np.abs(trades)) * transaction_cost_pct
            portfolio_value -= transaction_costs

            # Reset weights to be equal
            weights = np.full(n_assets, 1/n_assets)
            last_rebalance_date = date

        # Calculate portfolio value for the *next* day before the market opens
        # Get price changes from today to the next trading day
        today_prices = df.loc[date]
        next_day_index = df.index.get_loc(date) + 1
        if next_day_index < len(df):
            next_day_prices = df.iloc[next_day_index]
            price_change_ratio = next_day_prices / today_prices

            # Update portfolio value based on price changes
            portfolio_value = np.sum( (weights * portfolio_value) * price_change_ratio )

            # Update weights due to market drift
            new_asset_values = (weights * portfolio_value) * price_change_ratio
            weights = new_asset_values / np.sum(new_asset_values)

    print(f"Initial Investment: ${initial_balance:.2f}")
    print(f"Final Portfolio Value: ${portfolio_values.iloc[-1]:.2f}")

    return portfolio_values.dropna()


def main():
    # Load the test data
    test_df = pd.read_csv('data/test.csv', index_col='Date', parse_dates=True)

    # --- Run Baseline Strategies ---
    bnh_values = buy_and_hold(test_df)
    ewp_values = equally_weighted_rebalanced(test_df)

    # --- Plot the results ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(bnh_values.index, bnh_values, label='Buy and Hold', color='blue', linewidth=2)
    ax.plot(ewp_values.index, ewp_values, label='Equally Weighted (Rebalanced Monthly)', color='green', linewidth=2)

    ax.set_title('Baseline Strategy Performance (2021-2023)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(fontsize=12)

    # Format the y-axis to show currency
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig('baseline_performance.png')
    plt.show()

if __name__ == '__main__':
    main()