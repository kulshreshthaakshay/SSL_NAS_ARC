"""
Data Collection Script for SSL-NAS Research
Low-Data Regime with Multiple Stocks

This script collects financial data for self-supervised learning in low-data scenarios.
Uses multiple stocks from different sectors to ensure generalization.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append('..')
from financial_considerations import FinancialFeatureEngine, MarketRegimeDetector

# Setup paths (relative to project root)
project_root = Path(__file__).parent.parent
data_raw_dir = project_root / "data" / "raw"
data_processed_dir = project_root / "data" / "processed"

# Create directories if they don't exist
data_raw_dir.mkdir(parents=True, exist_ok=True)
data_processed_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION - LOW DATA REGIME
# ============================================================================

# Multiple stocks from different sectors for generalization
TICKERS = {
    'tech': 'AAPL',        # Technology
    'finance': 'JPM',      # Financial Services
    'energy': 'XOM',       # Energy
    'healthcare': 'JNJ',   # Healthcare
    'retail': 'WMT'        # Retail
}

# Low-data regime: 1 year of data
START_DATE = "2022-01-01"
END_DATE = "2023-01-01"

# Window configuration
WINDOW_SIZE = 30  # 30-day windows

# Train/Val/Test split ratios (time-series split, no shuffling)
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2  # Remaining

logger.info("=" * 80)
logger.info("SSL-NAS Data Collection - Low Data Regime")
logger.info("=" * 80)
logger.info(f"Stocks: {list(TICKERS.values())}")
logger.info(f"Date Range: {START_DATE} to {END_DATE}")
logger.info(f"Window Size: {WINDOW_SIZE} days")
logger.info("=" * 80)

# ============================================================================
# FUNCTIONS
# ============================================================================

def download_and_process_stock(ticker, start_date, end_date):
    """
    Download and process a single stock.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with processed features
    """
    logger.info(f"\nProcessing {ticker}...")
    
    try:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            logger.warning(f"No data downloaded for {ticker}")
            return None
            
        logger.info(f"  Downloaded {len(data)} days of data")
        
        # Save raw data
        raw_file = data_raw_dir / f"{ticker}.csv"
        data.to_csv(raw_file)
        logger.info(f"  Saved raw data to {raw_file}")
        
        # Prepare dataframe
        df = data.reset_index()
        
        # Handle multi-level columns from yfinance (tuples) or regular columns (strings)
        if isinstance(df.columns[0], tuple):
            # Flatten multi-level columns by taking first element
            df.columns = [col[0].lower() if isinstance(col, tuple) else str(col).lower() for col in df.columns]
        else:
            df.columns = [str(col).lower() for col in df.columns]
        
        # Convert to numeric
        for col in df.columns:
            if col not in ['date', 'datetime']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Feature engineering
        ffe = FinancialFeatureEngine()
        features = ffe.create_features(df)
        features_df = pd.DataFrame(features)
        
        # Drop all-NaN columns
        all_nan_cols = [col for col in features_df.columns if features_df[col].isna().all()]
        if all_nan_cols:
            logger.info(f"  Dropping all-NaN columns: {all_nan_cols}")
            features_df = features_df.drop(columns=all_nan_cols)
        
        # Handle rolling window features
        rolling_cols = [col for col in ['rsi', 'macd', 'bollinger_bands', 'realized_vol', 'garch_vol'] 
                       if col in features_df.columns]
        if rolling_cols:
            first_valids = [features_df[col].first_valid_index() or 0 for col in rolling_cols]
            min_valid_idx = max(first_valids) if first_valids else 0
            features_df = features_df.iloc[min_valid_idx:].reset_index(drop=True)
            logger.info(f"  Removed {min_valid_idx} rows due to rolling window initialization")
        
        # Forward fill then zero fill (conservative imputation)
        features_df = features_df.ffill().fillna(0)
        
        # Remove any remaining NaN or Inf
        num_before = len(features_df)
        features_df = features_df[~features_df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        num_after = len(features_df)
        if num_after < num_before:
            logger.info(f"  Removed {num_before - num_after} rows with NaN/Inf values")
        
        logger.info(f"  Final feature shape: {features_df.shape}")
        logger.info(f"  Features: {list(features_df.columns)}")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return None


def create_train_val_test_split(features_df, train_ratio, val_ratio):
    """
    Create time-series train/val/test split.
    
    Args:
        features_df: DataFrame with features
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    total_len = len(features_df)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    
    train_df = features_df.iloc[:train_end].copy()
    val_df = features_df.iloc[train_end:val_end].copy()
    test_df = features_df.iloc[val_end:].copy()
    
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def normalize_splits(train_df, val_df, test_df):
    """
    Normalize data splits (fit scaler on train only to avoid look-ahead bias).
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        
    Returns:
        Tuple of (train_norm, val_norm, test_norm, scaler)
    """
    scaler = MinMaxScaler()
    
    # Fit on train only (NO LOOK-AHEAD BIAS)
    train_norm = scaler.fit_transform(train_df)
    
    # Transform val and test using train statistics
    val_norm = scaler.transform(val_df)
    test_norm = scaler.transform(test_df)
    
    # Convert back to DataFrames
    train_norm_df = pd.DataFrame(train_norm, columns=train_df.columns)
    val_norm_df = pd.DataFrame(val_norm, columns=val_df.columns)
    test_norm_df = pd.DataFrame(test_norm, columns=test_df.columns)
    
    return train_norm_df, val_norm_df, test_norm_df, scaler


def create_windows(data, window_size):
    """
    Create sliding windows from time series data.
    
    Args:
        data: DataFrame or array with features
        window_size: Size of each window
        
    Returns:
        Array of windows with shape (num_windows, window_size, num_features)
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    windows = []
    for i in range(len(data) - window_size):
        window = data[i:i+window_size]
        windows.append(window)
    
    return np.array(windows)


def save_windows(windows, filename, split_name):
    """Save windows to file and log statistics."""
    filepath = data_processed_dir / filename
    np.save(filepath, windows)
    logger.info(f"  Saved {split_name}: {filepath}")
    logger.info(f"    Shape: {windows.shape}")
    logger.info(f"    Size: {windows.nbytes / 1024:.2f} KB")


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    """Main data collection pipeline."""
    
    all_train_windows = []
    all_val_windows = []
    all_test_windows = []
    
    # Process each stock
    for sector, ticker in TICKERS.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {ticker} ({sector.upper()})")
        logger.info(f"{'='*80}")
        
        # Download and process
        features_df = download_and_process_stock(ticker, START_DATE, END_DATE)
        
        if features_df is None or len(features_df) < WINDOW_SIZE + 10:
            logger.warning(f"Skipping {ticker} - insufficient data")
            continue
        
        # Split into train/val/test (time-series split)
        logger.info("\nSplitting data...")
        train_df, val_df, test_df = create_train_val_test_split(
            features_df, TRAIN_RATIO, VAL_RATIO
        )
        
        # Normalize (fit on train only - NO LOOK-AHEAD BIAS)
        logger.info("\nNormalizing data (fit on train only)...")
        train_norm, val_norm, test_norm, scaler = normalize_splits(
            train_df, val_df, test_df
        )
        
        # Create windows
        logger.info(f"\nCreating windows (size={WINDOW_SIZE})...")
        train_windows = create_windows(train_norm, WINDOW_SIZE)
        val_windows = create_windows(val_norm, WINDOW_SIZE)
        test_windows = create_windows(test_norm, WINDOW_SIZE)
        
        logger.info(f"  Train windows: {train_windows.shape}")
        logger.info(f"  Val windows: {val_windows.shape}")
        logger.info(f"  Test windows: {test_windows.shape}")
        
        # Collect windows from all stocks
        all_train_windows.append(train_windows)
        all_val_windows.append(val_windows)
        all_test_windows.append(test_windows)
    
    # Combine windows from all stocks
    logger.info(f"\n{'='*80}")
    logger.info("COMBINING DATA FROM ALL STOCKS")
    logger.info(f"{'='*80}")
    
    combined_train = np.concatenate(all_train_windows, axis=0)
    combined_val = np.concatenate(all_val_windows, axis=0)
    combined_test = np.concatenate(all_test_windows, axis=0)
    
    # Shuffle within each split (optional, but maintains temporal order per stock)
    # For SSL, shuffling is generally okay since we're learning representations
    np.random.seed(42)
    np.random.shuffle(combined_train)
    np.random.shuffle(combined_val)
    np.random.shuffle(combined_test)
    
    # Save combined windows
    logger.info("\nSaving combined windows...")
    save_windows(combined_train, "windows_train.npy", "Train")
    save_windows(combined_val, "windows_val.npy", "Val")
    save_windows(combined_test, "windows_test.npy", "Test")
    
    # Also save the main windows.npy for backward compatibility (train + val)
    combined_train_val = np.concatenate([combined_train, combined_val], axis=0)
    save_windows(combined_train_val, "windows.npy", "Train+Val (legacy)")
    
    # Final statistics
    logger.info(f"\n{'='*80}")
    logger.info("FINAL DATA STATISTICS")
    logger.info(f"{'='*80}")
    logger.info(f"Total stocks processed: {len(TICKERS)}")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Window size: {WINDOW_SIZE} days")
    logger.info(f"\nCombined Dataset:")
    logger.info(f"  Train: {combined_train.shape[0]} windows")
    logger.info(f"  Val: {combined_val.shape[0]} windows")
    logger.info(f"  Test: {combined_test.shape[0]} windows")
    logger.info(f"  Total: {combined_train.shape[0] + combined_val.shape[0] + combined_test.shape[0]} windows")
    logger.info(f"  Features per window: {combined_train.shape[2]}")
    logger.info(f"\n✅ LOW-DATA REGIME: {combined_train.shape[0]} training samples")
    logger.info(f"{'='*80}")
    
    # Verify low-data regime
    if combined_train.shape[0] > 1000:
        logger.warning("⚠️  WARNING: Training set has >1000 samples - not truly low-data!")
    else:
        logger.info("✅ Confirmed: Low-data regime (<1000 training samples)")


if __name__ == "__main__":
    main()
