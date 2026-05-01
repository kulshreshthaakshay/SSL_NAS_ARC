import numpy as np
import logging
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)
from sklearn.metrics import accuracy_score, f1_score
from arch import arch_model
# Critical financial domain considerations missing:

# 1. Financial Feature Engineering
class FinancialFeatureEngine:
    """You need proper financial features beyond OHLCV"""
    
    def create_features(self, df):
        features = {}
        # Defensive: handle missing columns gracefully and ensure numeric
        close = df['close'] if 'close' in df.columns else df.iloc[:, 0]
        close = pd.to_numeric(close, errors='coerce')
        features['returns'] = close.pct_change()
        features['log_returns'] = np.log(close / close.shift(1))
        
        # Technical indicators
        features['rsi'] = self.calculate_rsi(close)
        features['macd'] = self.calculate_macd(close)
        features['bollinger_bands'] = self.calculate_bollinger(close)
        
        # Volatility measures
        features['realized_vol'] = features['returns'].rolling(20).std()
        features['garch_vol'] = self.calculate_garch_vol(features['returns'])
        
        # Market microstructure (handle missing columns)
        # NOTE: bid-ask spread requires 'ask', 'bid', and 'mid' columns (microstructure data, not present in typical OHLCV). Will be np.nan if missing.
        if all(col in df.columns for col in ['ask', 'bid', 'mid']):
            ask = pd.to_numeric(df['ask'], errors='coerce')
            bid = pd.to_numeric(df['bid'], errors='coerce')
            mid = pd.to_numeric(df['mid'], errors='coerce')
            features['bid_ask_spread'] = (ask - bid) / mid
        else:
            features['bid_ask_spread'] = np.nan
        # Removed: features['order_flow_imbalance'] and macroeconomic features
        # Momentum indicator (rate of change)
        features['momentum_10'] = close.diff(10)
        # Volume-based indicators
        if 'volume' in df.columns:
            volume = pd.to_numeric(df['volume'], errors='coerce')
            features['volume_zscore'] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)
            features['on_balance_volume'] = (np.sign(features['returns'].fillna(0)) * volume).cumsum()
        else:
            features['volume_zscore'] = np.nan
            features['on_balance_volume'] = np.nan
        # Removed: macroeconomic features (cpi, gdp, unemployment)
        # TODO: Ensure rolling VWAP calculation does not introduce look-ahead bias when applied in datacollection.py. Feature calculation should ideally happen after data splitting or on a per-window basis using only past data.
        features['vwap'] = self.calculate_vwap(df)
        
        return features

    # Dummy implementations for missing methods to avoid AttributeError
    def calculate_rsi(self, close, window=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        # Wilder's EMA for average gain/loss
        avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, close, span1=12, span2=26, signal=9):
        ema1 = close.ewm(span=span1, adjust=False).mean()
        ema2 = close.ewm(span=span2, adjust=False).mean()
        macd = ema1 - ema2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line
    
    def calculate_bollinger(self, close, window=20):
        sma = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return upper - lower
    
    def calculate_garch_vol(self, returns):
        """Calculate conditional volatility using a GARCH(1,1) model (arch library)."""
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("The 'arch' library is required for GARCH volatility. Install with 'pip install arch'.")
        # Drop NaNs for fitting
        returns_clean = returns.dropna()
        if len(returns_clean) < 30:
            # Not enough data to fit GARCH, return NaNs
            return pd.Series(np.nan, index=returns.index)
        # Fit GARCH(1,1)
        am = arch_model(returns_clean, vol='Garch', p=1, q=1, rescale=False)
        res = am.fit(disp='off')
        cond_vol = res.conditional_volatility
        # Align output to input index (NaN where input was NaN)
        full_vol = pd.Series(np.nan, index=returns.index)
        full_vol.loc[cond_vol.index] = cond_vol
        return full_vol
    
    def calculate_vwap(self, df, window=14):
        # Calculate rolling VWAP (Volume Weighted Average Price) over a given window
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            # Return NaNs if required columns are missing
            return pd.Series(np.nan, index=df.index)
        typical_price = (pd.to_numeric(df['high'], errors='coerce') +
                         pd.to_numeric(df['low'], errors='coerce') +
                         pd.to_numeric(df['close'], errors='coerce')) / 3
        volume = pd.to_numeric(df['volume'], errors='coerce')
        tp_volume = typical_price * volume
        vwap = tp_volume.rolling(window).sum() / volume.rolling(window).sum()
        return vwap

# 2. Market Regime Detection
class MarketRegimeDetector:
    """Financial markets have different regimes - your model should account for this"""
    
    def detect_regimes(self, returns):
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise ImportError("hmmlearn and scikit-learn are required. Please install them with 'pip install hmmlearn scikit-learn'.")
        # Handle NaNs and normalize returns before HMM
        returns = np.nan_to_num(returns)
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns.reshape(-1, 1))
        model = hmm.GaussianHMM(n_components=3)  # Bull, Bear, Sideways
        model.fit(returns_scaled)
        regimes = model.predict(returns_scaled)
        return regimes

# 3. Risk-Adjusted Evaluation
class FinancialEvaluator:
    """Standard ML metrics aren't enough for finance"""
    
    def evaluate_model(self, predictions, labels, returns, prices):
        metrics = {}
        
        # Traditional ML metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['f1_score'] = f1_score(labels, predictions)
        
        # Financial metrics with error handling
        try:
            portfolio_returns = self.calculate_portfolio_returns(predictions, returns)
            metrics['sharpe_ratio'] = self.calculate_sharpe(portfolio_returns)
            metrics['max_drawdown'] = self.calculate_max_drawdown(portfolio_returns)
            metrics['calmar_ratio'] = metrics['sharpe_ratio'] / (abs(metrics['max_drawdown']) + 1e-6)
        except Exception as e:
            logger.warning(f"Error calculating financial metrics: {e}")
            metrics['sharpe_ratio'] = float('-inf')
            metrics['max_drawdown'] = 0.0
            metrics['calmar_ratio'] = float('-inf')
        
        return metrics

    def calculate_portfolio_returns(self, predictions, returns):
        """Calculate portfolio returns with proper handling of edge cases."""
        if len(predictions) != len(returns):
            raise ValueError(f"Length mismatch: predictions ({len(predictions)}) != returns ({len(returns)})")
        
        # Ensure predictions are valid
        predictions = np.array(predictions)
        if not np.any(predictions != 0):  # Check if all predictions are 0
            raise ValueError("All predictions are 0, leading to no trading signals")
            
        # Convert to trading signals: 1 for long, -1 for short
        signals = np.where(predictions > 0, 1, -1)
        
        # Align signals with future returns (avoid lookahead bias)
        if len(signals) > 1:
            portfolio_returns = signals[:-1] * returns[1:]
        else:
            raise ValueError("Not enough data points for calculating returns")
            
        return portfolio_returns

    def calculate_sharpe(self, returns):
        """Calculate annualized Sharpe ratio with proper error handling."""
        if len(returns) == 0:
            raise ValueError("Empty returns array")
            
        # Remove any infinite or NaN values
        returns = returns[np.isfinite(returns)]
        if len(returns) == 0:
            raise ValueError("No valid returns after filtering")
            
        # Calculate mean and std with safeguards
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            if mean > 0:
                return float('inf')
            elif mean < 0:
                return float('-inf')
            else:
                return 0.0
                
        # Annualize (assuming daily returns)
        annualized_sharpe = mean / (std + 1e-8) * np.sqrt(252)
        
        # Clip extreme values
        return np.clip(annualized_sharpe, -100, 100)

    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown with proper error handling."""
        if len(returns) == 0:
            return 0.0
            
        # Remove any infinite or NaN values
        returns = returns[np.isfinite(returns)]
        if len(returns) == 0:
            return 0.0
            
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - peak) / peak
        
        return float(np.min(drawdowns))

# 4. Transaction Cost Modeling
class TransactionCostModel:
    """Real trading has costs - your backtest should include them"""
    
    def __init__(self, spread_bps=5, commission=0.001, slippage_bps=1):
        self.spread_bps = spread_bps
        self.commission = commission
        self.slippage_bps = slippage_bps
    
    def apply_costs(self, signals, prices):
        # Apply bid-ask spread and commission costs
        spread_cost = np.abs(np.diff(signals)) * (self.spread_bps / 10000)
        commission_cost = np.abs(np.diff(signals)) * self.commission
        # Slippage cost: proportional to price and trade
        slippage_cost = np.abs(np.diff(signals)) * (self.slippage_bps / 10000) * prices[:-1]
        total_cost = spread_cost + commission_cost + slippage_cost
        return total_cost

# 5. Data Leakage Prevention
class DataLeakageValidator:
    """Critical for financial data - no future information should leak"""
    
    def validate_features(self, features, timestamps=None, target_variable=None):
        # Check for look-ahead bias
        for feature_name, feature_values in features.items():
            # Heuristic: warn if feature name suggests rolling window or technical indicator
            if any(key in feature_name.lower() for key in ['rsi', 'macd', 'rolling', 'bollinger', 'ema', 'sma', 'std', 'vol', 'momentum']):
                print(f"[WARNING] Feature '{feature_name}' appears to be calculated with a rolling window or technical indicator. Manually verify that its calculation does NOT use future data (no look-ahead bias). Ensure all rolling windows, pct_change, etc. only use past data up to time t.")
            if target_variable is not None:
                self.has_future_information(feature_values, target_variable)

    def has_future_information(self, feature_series, target_series, feature_timestamps=None, label_timestamps=None):
        """
        Checks for look-ahead bias by ensuring no feature timestamp is >= the corresponding label timestamp.
        feature_series: pd.Series or np.ndarray of feature values
        target_series: pd.Series or np.ndarray of target values
        feature_timestamps: pd.Series or np.ndarray of timestamps for features (optional)
        label_timestamps: pd.Series or np.ndarray of timestamps for labels (optional)
        """
        if feature_timestamps is None or label_timestamps is None:
            raise ValueError("Timestamps must be provided for both features and labels to check for look-ahead bias.")
        # Ensure same length
        if len(feature_timestamps) != len(label_timestamps):
            raise ValueError("Feature and label timestamps must be the same length for robust leakage check.")
        # Check for any feature timestamp >= label timestamp
        for i, (ft, lt) in enumerate(zip(feature_timestamps, label_timestamps)):
            if ft >= lt:
                raise ValueError(f"[LEAKAGE DETECTED] Feature at index {i} uses timestamp {ft} >= label timestamp {lt}. This indicates look-ahead bias. Feature engineering must only use information up to (not including) the label time.")
        # If no issues found
        print("[Leakage Check] No look-ahead bias detected based on timestamps.")