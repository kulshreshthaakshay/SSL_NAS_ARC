import numpy as np
import pandas as pd


class FinancialFeatureEngine:
    """Past-only financial features for daily OHLCV data."""

    def create_features(self, df):
        close = self._numeric(df, "close", fallback_col=0)
        features = {
            "returns": close.pct_change(),
            "log_returns": np.log(close / close.shift(1)),
            "rsi": self.calculate_rsi(close),
            "macd": self.calculate_macd(close),
            "bollinger_bands": self.calculate_bollinger(close),
            "realized_vol": close.pct_change().rolling(20).std(),
            "momentum_10": close.diff(10),
            "vwap": self.calculate_vwap(df),
        }

        if "volume" in df.columns:
            volume = pd.to_numeric(df["volume"], errors="coerce")
            returns = features["returns"].fillna(0)
            features["volume_zscore"] = (
                (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)
            )
            features["on_balance_volume"] = (np.sign(returns) * volume).cumsum()
        else:
            features["volume_zscore"] = np.nan
            features["on_balance_volume"] = np.nan

        return features

    @staticmethod
    def _numeric(df, column, fallback_col=None):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
        if fallback_col is not None:
            return pd.to_numeric(df.iloc[:, fallback_col], errors="coerce")
        raise KeyError(column)

    def calculate_rsi(self, close, window=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, close, span1=12, span2=26, signal=9):
        ema1 = close.ewm(span=span1, adjust=False).mean()
        ema2 = close.ewm(span=span2, adjust=False).mean()
        macd = ema1 - ema2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def calculate_bollinger(self, close, window=20):
        sma = close.rolling(window).mean()
        std = close.rolling(window).std()
        return (sma + 2 * std) - (sma - 2 * std)

    def calculate_vwap(self, df, window=14):
        required_cols = ["high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            return pd.Series(np.nan, index=df.index)
        typical_price = (
            pd.to_numeric(df["high"], errors="coerce")
            + pd.to_numeric(df["low"], errors="coerce")
            + pd.to_numeric(df["close"], errors="coerce")
        ) / 3
        volume = pd.to_numeric(df["volume"], errors="coerce")
        return (typical_price * volume).rolling(window).sum() / volume.rolling(window).sum()
