"""
Data collection and preprocessing for the corrected SSL-NAS pipeline.

The script keeps validation/test samples chronological, creates a separate
architecture-search split, and saves per-window metadata for deterministic
financial evaluation.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from financial_considerations import FinancialFeatureEngine

project_root = Path(__file__).parent.parent
data_raw_dir = project_root / "data" / "raw"
data_processed_dir = project_root / "data" / "processed"
cache_dir = project_root / ".cache" / "yfinance"

data_raw_dir.mkdir(parents=True, exist_ok=True)
data_processed_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)
if hasattr(yf, "set_tz_cache_location"):
    yf.set_tz_cache_location(str(cache_dir))

TICKERS = {
    "tech": "AAPL",
    "finance": "JPM",
    "energy": "XOM",
    "healthcare": "JNJ",
    "retail": "WMT",
}

START_DATE = "2020-01-01"
END_DATE = "2023-01-01"
WINDOW_SIZE = 30
FORECAST_HORIZON = 30
MAX_WEIGHT_WINDOWS = 1000
TASK_NAME = f"future_{FORECAST_HORIZON}d_direction"

WEIGHT_RATIO = 0.50
ARCH_RATIO = 0.10
VAL_RATIO = 0.20
TEST_RATIO = 0.20


def _flatten_yfinance_columns(df):
    if isinstance(df.columns[0], tuple):
        df.columns = [col[0].lower() if isinstance(col, tuple) else str(col).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    return df


def download_and_process_stock(ticker, start_date, end_date):
    logger.info(f"Processing {ticker}")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        logger.warning(f"No data downloaded for {ticker}")
        return None

    raw_file = data_raw_dir / f"{ticker}.csv"
    data.to_csv(raw_file)

    df = _flatten_yfinance_columns(data.reset_index())
    date_col = "date" if "date" in df.columns else "datetime"
    for col in df.columns:
        if col != date_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    close = pd.to_numeric(df["close"], errors="coerce")
    dates = pd.to_datetime(df[date_col])
    features_df = pd.DataFrame(FinancialFeatureEngine().create_features(df))
    all_nan_cols = [col for col in features_df.columns if features_df[col].isna().all()]
    if all_nan_cols:
        features_df = features_df.drop(columns=all_nan_cols)

    first_valids = [features_df[col].first_valid_index() or 0 for col in features_df.columns]
    first_valid = max(first_valids) if first_valids else 0
    features_df = features_df.iloc[first_valid:].reset_index(drop=True)
    close = close.iloc[first_valid:].reset_index(drop=True)
    dates = dates.iloc[first_valid:].reset_index(drop=True)

    features_df = features_df.ffill().fillna(0)
    valid_mask = np.isfinite(features_df.to_numpy(dtype=float)).all(axis=1) & np.isfinite(close.to_numpy(dtype=float))
    features_df = features_df.loc[valid_mask].reset_index(drop=True)
    close = close.loc[valid_mask].reset_index(drop=True)
    dates = dates.loc[valid_mask].reset_index(drop=True)

    logger.info(f"{ticker}: features={features_df.shape}, close={close.shape}")
    return features_df, close.to_numpy(dtype=float), dates


def split_stock_data(features_df, close, dates):
    total_len = len(features_df)
    weight_end = int(total_len * WEIGHT_RATIO)
    arch_end = int(total_len * (WEIGHT_RATIO + ARCH_RATIO))
    val_end = int(total_len * (WEIGHT_RATIO + ARCH_RATIO + VAL_RATIO))

    slices = {
        "train": slice(0, weight_end),
        "arch": slice(weight_end, arch_end),
        "val": slice(arch_end, val_end),
        "test": slice(val_end, total_len),
    }
    result = {}
    for split, slc in slices.items():
        result[split] = (
            features_df.iloc[slc].reset_index(drop=True),
            close[slc],
            dates.iloc[slc].reset_index(drop=True),
            slc.start,
        )
    return result


def normalize_splits(split_data):
    scaler = MinMaxScaler()
    train_features = split_data["train"][0]
    normalized = {}
    scaler.fit(train_features)
    for split, (features, close, dates, offset) in split_data.items():
        norm = pd.DataFrame(scaler.transform(features), columns=features.columns)
        normalized[split] = (norm, close, dates, offset)
    return normalized, scaler


def create_windows_with_future_targets(
    norm_data,
    raw_close_prices,
    dates,
    window_size,
    forecast_horizon,
    ticker="UNKNOWN",
    split="unknown",
    global_offset=0,
):
    if isinstance(norm_data, pd.DataFrame):
        norm_data = norm_data.values
    raw_close_prices = np.asarray(raw_close_prices, dtype=float)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if len(norm_data) != len(raw_close_prices) or len(norm_data) != len(dates):
        raise ValueError("Feature, close, and date arrays must be aligned")

    windows = []
    labels = []
    future_returns = []
    metadata = []
    max_start = len(norm_data) - window_size - forecast_horizon + 1
    for i in range(max(0, max_start)):
        prediction_idx = i + window_size - 1
        target_idx = prediction_idx + forecast_horizon
        entry_price = raw_close_prices[prediction_idx]
        future_price = raw_close_prices[target_idx]
        future_return = 0.0 if entry_price == 0 else (future_price - entry_price) / entry_price

        windows.append(norm_data[i:i + window_size])
        labels.append(1 if future_return > 0 else 0)
        future_returns.append(future_return)
        metadata.append({
            "ticker": ticker,
            "split": split,
            "window_start_idx": int(global_offset + i),
            "prediction_idx": int(global_offset + prediction_idx),
            "target_idx": int(global_offset + target_idx),
            "prediction_date": dates.iloc[prediction_idx].strftime("%Y-%m-%d"),
            "target_date": dates.iloc[target_idx].strftime("%Y-%m-%d"),
        })

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(future_returns, dtype=np.float64),
        pd.DataFrame(metadata),
    )


def validate_future_target_logic():
    synthetic_features = np.arange(100, dtype=float).reshape(100, 1)
    synthetic_close = np.arange(100, dtype=float) + 100.0
    synthetic_dates = pd.date_range("2020-01-01", periods=100, freq="D")
    windows, labels, future_returns, metadata = create_windows_with_future_targets(
        synthetic_features,
        synthetic_close,
        synthetic_dates,
        window_size=30,
        forecast_horizon=30,
        ticker="TEST",
        split="test",
    )
    expected_count = 100 - 30 - 30 + 1
    expected_first_return = (synthetic_close[59] - synthetic_close[29]) / synthetic_close[29]
    assert len(windows) == expected_count
    assert len(windows) == len(labels) == len(future_returns) == len(metadata)
    assert np.isclose(future_returns[0], expected_first_return)
    assert labels[0] == 1
    assert metadata.iloc[0]["prediction_idx"] == 29
    assert metadata.iloc[0]["target_idx"] == 59


def save_split(split, windows, labels, returns, metadata):
    np.save(data_processed_dir / f"windows_{split}.npy", windows)
    np.save(data_processed_dir / f"labels_{split}.npy", labels)
    np.save(data_processed_dir / f"future_returns_{split}.npy", returns)
    metadata.to_csv(data_processed_dir / f"sample_metadata_{split}.csv", index=False)
    logger.info(
        f"Saved {split}: windows={windows.shape}, labels={labels.shape}, "
        f"returns={returns.shape}, metadata={metadata.shape}"
    )


def main():
    validate_future_target_logic()
    collected = {split: {"windows": [], "labels": [], "returns": [], "metadata": []}
                 for split in ["train", "arch", "val", "test"]}

    for _, ticker in TICKERS.items():
        processed = download_and_process_stock(ticker, START_DATE, END_DATE)
        if processed is None:
            continue
        features_df, close, dates = processed
        if len(features_df) < WINDOW_SIZE + FORECAST_HORIZON + 10:
            logger.warning(f"Skipping {ticker}: insufficient rows after feature engineering")
            continue

        split_data = split_stock_data(features_df, close, dates)
        normalized, _ = normalize_splits(split_data)
        for split, (norm, split_close, split_dates, offset) in normalized.items():
            windows, labels, future_returns, metadata = create_windows_with_future_targets(
                norm,
                split_close,
                split_dates,
                WINDOW_SIZE,
                FORECAST_HORIZON,
                ticker=ticker,
                split=split,
                global_offset=offset,
            )
            if len(windows) == 0:
                logger.warning(f"{ticker} {split}: no usable windows")
                continue
            collected[split]["windows"].append(windows)
            collected[split]["labels"].append(labels)
            collected[split]["returns"].append(future_returns)
            collected[split]["metadata"].append(metadata)

    if not collected["train"]["windows"]:
        raise RuntimeError("No usable training windows were generated")

    for split, parts in collected.items():
        windows = np.concatenate(parts["windows"], axis=0)
        labels = np.concatenate(parts["labels"], axis=0)
        returns = np.concatenate(parts["returns"], axis=0)
        metadata = pd.concat(parts["metadata"], ignore_index=True)

        if split == "train" and len(windows) > MAX_WEIGHT_WINDOWS:
            windows = windows[:MAX_WEIGHT_WINDOWS]
            labels = labels[:MAX_WEIGHT_WINDOWS]
            returns = returns[:MAX_WEIGHT_WINDOWS]
            metadata = metadata.iloc[:MAX_WEIGHT_WINDOWS].reset_index(drop=True)

        assert len(windows) == len(labels) == len(returns) == len(metadata)
        save_split(split, windows, labels, returns, metadata)

    logger.info(f"Completed data generation for task={TASK_NAME}")


if __name__ == "__main__":
    main()
