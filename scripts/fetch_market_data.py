import argparse
import os
import pandas as pd
import yfinance as yf
from datetime import date

def fetch_data(start_date, end_date, output_filename):
    """
    Fetches, cleans, and saves historical market data for a given date range.

    Args:
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        output_filename (str): The path and name of the file to save the data.
    """
    print(f"--- Fetching data from {start_date} to {end_date} ---")

    # Define the base list of tickers
    tickers = ["AAPL", "MSFT", "SPY", "TLT", "BTC-USD"]
    
    # Smartly remove Bitcoin if the period is before its existence (e.g., before 2013)
    if pd.to_datetime(start_date).year < 2013:
        print("Note: Bitcoin (BTC-USD) did not exist for the requested period and will be excluded.")
        tickers.remove("BTC-USD")

    # Download data from Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date)
    close_data = data['Close'].copy()

    # Data Cleaning
    print(f"\nMissing values before cleaning:\n{close_data.isnull().sum()}")
    close_data.ffill(inplace=True)
    close_data.bfill(inplace=True)
    
    # Drop any columns that are still all NaN (like BTC in the 2008 data)
    close_data.dropna(axis=1, how='all', inplace=True)
    
    print(f"\nMissing values after cleaning:\n{close_data.isnull().sum()}")

    # Ensure data directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    close_data.to_csv(output_filename)
    print(f"\n✅ Data successfully saved to {output_filename}")


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Fetch historical market data for specified periods.")
    
    parser.add_argument(
        "--start",
        type=str,
        default="2018-01-01",
        help="Start date in YYYY-MM-DD format. Default is for the 2018 stress test."
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2019-12-31",
        help="End date in YYYY-MM-DD format. Default is for the 2018 stress test."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="data/stress_test_2018.csv",
        help="Output file name (e.g., 'data/my_data.csv')."
    )

    args = parser.parse_args()

    # Use 'today' as the end date if specified
    end_date = date.today().strftime('%Y-%m-%d') if args.end.lower() == 'today' else args.end

    fetch_data(start_date=args.start, end_date=end_date, output_filename=args.filename)