import requests
import pandas as pd
import datetime
from tenacity import retry, stop_after_attempt, wait_fixed
from pathlib import Path
import pytz
import time

# --- Settings ---
SYMBOL = "ETHUSDT"
INTERVAL = "1h"
START_YEAR = 2018
DATA_DIR = Path("binance_data")

# --- Create data folder if it doesn't exist ---
DATA_DIR.mkdir(exist_ok=True)


@retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
def get_binance_data(symbol, interval, start_dt, end_dt):
    """Get historical K-line data from Binance."""
    url = "https://api.binance.com/api/v3/klines"

    # Convert dates to millisecond timestamp and UTC
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    all_data = []
    current_start_ts = start_ts

    print(
        f"Getting {symbol} from {start_dt.strftime('%Y-%m-%d %H:%M:%S')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S')}..."
    )

    while current_start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start_ts,
            "endTime": end_ts,
            "limit": 1000,
        }
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            klines = response.json()

            if not klines:
                print("  No more data found for this period.")
                break

            all_data.extend(klines)
            # Update start time to the last candle's time + 1 millisecond
            current_start_ts = klines[-1][0] + 1

            # Small delay between requests
            time.sleep(0.1)

            print(
                f"    {len(klines)} candles received, new start: {pd.to_datetime(current_start_ts, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M:%S')}"
            )

        except requests.exceptions.RequestException as e:
            print(f"  Request failed: {e}. Retrying...")
            raise

    if not all_data:
        print("No data received from API")
        return pd.DataFrame()

    print(f"Total {len(all_data)} candles received from API")

    df = pd.DataFrame(
        all_data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="first")]

    print(f"DataFrame created with {len(df)} rows")
    return df


def get_last_saved_timestamp(symbol):
    """Find the timestamp of the last saved data point."""
    files = sorted(DATA_DIR.glob(f"binance_data_{symbol}_*.csv"))

    # Ignore 'combined' files
    files = [f for f in files if "combined" not in f.name]

    if not files:
        print("No existing data files found. Starting from 2018.")
        return None

    # Try to read the most recent file
    last_file = files[-1]
    try:
        df = pd.read_csv(last_file, index_col="open_time", parse_dates=True)

        # Ensure the index is timezone-aware UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        if not df.empty:
            last_timestamp = df.index[-1]
            print(f"Last saved timestamp found: {last_timestamp}")
            return last_timestamp
        else:
            print("Last file is empty. Starting from 2018.")
            return None

    except Exception as e:
        print(f"Error reading {last_file}: {e}. Starting from 2018.")
        return None


def save_incremental_data(df, symbol):
    """Save data incrementally by year to avoid memory issues."""
    if df.empty:
        print("No data to save.")
        return

    print(f"Saving {len(df)} data points incrementally by year...")
    print(f"Data time range: {df.index.min()} to {df.index.max()}")

    # Group by year and process each year separately
    for year, year_df in df.groupby(df.index.year):
        file_path = DATA_DIR / f"binance_data_{symbol}_{year}.csv"
        print(f"\nProcessing year {year} with {len(year_df)} rows...")

        if file_path.exists():
            # Read existing data for this year
            try:
                existing_df = pd.read_csv(
                    file_path, index_col="open_time", parse_dates=True
                )
                if existing_df.index.tz is None:
                    existing_df.index = existing_df.index.tz_localize("UTC")
                else:
                    existing_df.index = existing_df.index.tz_convert("UTC")

                # Combine with new data
                combined_df = pd.concat([existing_df, year_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
                combined_df.sort_index(inplace=True)

                print(
                    f"  Combined {len(existing_df)} existing + {len(year_df)} new = {len(combined_df)} total"
                )

                # Save immediately
                combined_df.to_csv(file_path)
                print(f"  ✓ Immediately saved: {file_path}")

            except Exception as e:
                print(f"  Error processing existing file: {e}")
                # Save new data directly if there's an error reading existing file
                year_df.to_csv(file_path)
                print(f"  ✓ Saved new data: {file_path}")
        else:
            # Save new year data immediately
            year_df.to_csv(file_path)
            print(f"  ✓ Created new file: {file_path}")


def update_binance_data_incremental(symbol, interval, batch_size_days=30):
    """Download and update Binance data in smaller batches to avoid memory issues."""
    utc = pytz.UTC
    start_date_abs = datetime.datetime(START_YEAR, 1, 1, tzinfo=utc)
    end_date = datetime.datetime.now(utc)

    # Find the last saved timestamp
    last_timestamp = get_last_saved_timestamp(symbol)
    start_date = last_timestamp if last_timestamp else start_date_abs

    if start_date >= end_date:
        print("Data is already up-to-date.")
        return

    print(f"Downloading from: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Downloading to: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

    current_start = start_date
    batch_count = 0

    while current_start < end_date:
        # Calculate batch end date (current_start + batch_size_days, but not beyond end_date)
        batch_end = min(
            current_start + datetime.timedelta(days=batch_size_days), end_date
        )

        print(
            f"\n--- Batch {batch_count + 1}: {current_start.strftime('%Y-%m-%d')} to {batch_end.strftime('%Y-%m-%d')} ---"
        )

        # Get data for this batch
        batch_df = get_binance_data(symbol, interval, current_start, batch_end)

        if not batch_df.empty:
            print(f"Batch {batch_count + 1}: Downloaded {len(batch_df)} rows")
            # Save this batch immediately
            save_incremental_data(batch_df, symbol)
        else:
            print(f"Batch {batch_count + 1}: No new data")

        # Move to next batch
        current_start = batch_end
        batch_count += 1

        # Small delay between batches
        time.sleep(1)

    print(f"\nCompleted {batch_count} batches")


def combine_yearly_data(symbol, data_dir):
    """Combines all yearly data files into a single CSV file."""
    print("\nStarting the process of combining yearly files...")
    files_to_combine = sorted(data_dir.glob(f"binance_data_{symbol}_*.csv"))

    # Remove 'combined' files from the list so they are not used in re-combination
    files_to_combine = [f for f in files_to_combine if "combined" not in f.name]

    if not files_to_combine:
        print("No data files found to combine.")
        return

    all_dfs = []
    for file in files_to_combine:
        try:
            print(f"  Reading: {file.name}")
            df = pd.read_csv(file, index_col="open_time", parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            all_dfs.append(df)
            print(f"    ✓ Loaded {len(df)} rows from {file.name}")
        except Exception as e:
            print(f"    ✗ Error reading {file.name}: {e}")

    if not all_dfs:
        print("No data was read to combine.")
        return

    print("  Combining data...")
    combined_df = pd.concat(all_dfs)

    print("  Sorting and removing duplicates...")
    combined_df.sort_index(inplace=True)
    combined_df = combined_df[~combined_df.index.duplicated(keep="last")]

    # Create filename with the current date
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    combined_file_name = data_dir / f"binance_data_{symbol}_combined_{today_str}.csv"

    print(f"  Saving combined file to: {combined_file_name}")
    combined_df.to_csv(combined_file_name)
    print(f"✓ Combined file with {len(combined_df)} rows of data saved successfully.")


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Binance Data Download and Combination Script ---")

    # Step 1: Update yearly data incrementally
    update_binance_data_incremental(SYMBOL, INTERVAL, batch_size_days=30)

    # Step 2: Combine yearly files into a single master file
    combine_yearly_data(SYMBOL, DATA_DIR)

    print("--- End of Script ---")
