import yfinance as yf
import pandas as pd
import os

# --- Configuration ---
# Asset tickers
TICKERS = ["AAPL", "MSFT", "SPY", "TLT", "BTC-USD"]
# Time periods for training and testing
TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2020-12-31"
TEST_START_DATE = "2021-01-01"
TEST_END_DATE = "2023-12-31"

# Directory to save the data
DATA_DIR = "data"
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

# --- Data Fetching and Processing ---

def fetch_and_prepare_data(start_date, end_date, tickers):
    """
    Fetches historical data for the given tickers and processes it.
    Returns a DataFrame with 'Close' prices for each ticker.
    """
    print(f"Fetching data from {start_date} to {end_date} for {tickers}...")
    
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # CHANGE: Add .copy() to explicitly create a new DataFrame and avoid warnings.
    close_data = data['Close'].copy()
    
    print("\nData Head:")
    print(close_data.head())
    
    print("\nMissing values before cleaning:")
    print(close_data.isnull().sum())
    
    # Now, all inplace operations are safely performed on our own copy.
    close_data.ffill(inplace=True)
    close_data.bfill(inplace=True)

    print("\nMissing values after cleaning:")
    print(close_data.isnull().sum())

    for col in close_data.columns:
        close_data[col] = pd.to_numeric(close_data[col], errors='coerce')

    close_data.dropna(inplace=True)
    
    return close_data

def main():
    """Main function to run the data fetching process."""
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")
        
    # Fetch, process, and save training data
    print("--- Preparing Training Data ---")
    train_data = fetch_and_prepare_data(TRAIN_START_DATE, TRAIN_END_DATE, TICKERS)
    train_data.to_csv(TRAIN_DATA_PATH)
    print(f"Training data saved to {TRAIN_DATA_PATH}")

    print("\n" + "="*50 + "\n")

    # Fetch, process, and save testing data
    print("--- Preparing Testing Data ---")
    test_data = fetch_and_prepare_data(TEST_START_DATE, TEST_END_DATE, TICKERS)
    test_data.to_csv(TEST_DATA_PATH)
    print(f"Testing data saved to {TEST_DATA_PATH}")

if __name__ == "__main__":
    main()