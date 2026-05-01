import numpy as np
from scipy.interpolate import interp1d

class TimeSeriesAugmentor:
    """
    Finance-appropriate augmentations that preserve critical temporal patterns.
    Avoids distortions that would invalidate financial time series properties.
    """
    def __init__(self, seed=None):
        """Initialize augmentation with optional random seed."""
        if seed is not None:
            np.random.seed(seed)
    
    def trend_preserving_warp(self, window, warp_strength=0.1):
        """
        Warp time series while preserving trend direction.
        Critical for financial data where trend is important.
        """
        augmented = window.copy()
        seq_len = len(window)
        
        # Calculate local trend
        trend = np.polyfit(np.arange(seq_len), window[:, 0], 1)[0]
        
        # Generate smooth warping that preserves trend sign
        warp = np.random.randn(seq_len) * warp_strength
        warp = np.convolve(warp, np.ones(5)/5, mode='same')  # Smooth
        
        # Apply warp while preserving trend direction
        for i in range(window.shape[1]):
            if trend > 0:
                augmented[:, i] *= (1 + np.abs(warp))
            else:
                augmented[:, i] *= (1 - np.abs(warp) * 0.5)
        
        return augmented
    
    def add_calibrated_noise(self, window, noise_level=0.01):
        """
        Add noise calibrated to the local volatility of the series.
        This preserves the heteroskedastic nature of financial data.
        
        Args:
            window (np.ndarray): Input time series of shape (sequence_length, features)
            noise_level (float): Scaling factor for noise (0.01 = 1% of local std)
        """
        # Calculate rolling standard deviation for each feature
        augmented = window.copy()
        
        for i in range(window.shape[1]):
            # Use local volatility (last 5 points) to calibrate noise
            local_std = np.std(window[-5:, i]) if len(window) >= 5 else np.std(window[:, i])
            
            # Add noise proportional to local volatility
            if local_std > 0:
                noise = np.random.normal(0, local_std * noise_level, len(window))
                augmented[:, i] += noise
                
        return augmented
    
    def magnitude_warp(self, window, sigma=0.2, knot_ratio=0.1):
        """
        Warp the magnitude of the time series while preserving temporal structure.
        This simulates different volatility regimes without distorting timing.
        
        Args:
            window (np.ndarray): Input time series
            sigma (float): Standard deviation of the random warping
            knot_ratio (float): Ratio of knot points to sequence length
        """
        seq_len = len(window)
        augmented = window.copy()
        
        # Generate smooth magnitude scaling curve
        num_knots = max(3, int(seq_len * knot_ratio))
        knot_positions = np.linspace(0, seq_len - 1, num_knots)
        
        # Generate scaling factors (centered around 1.0)
        scaling_factors = np.exp(np.random.normal(0, sigma, num_knots))
        
        # Ensure reasonable bounds (avoid extreme scaling)
        scaling_factors = np.clip(scaling_factors, 0.5, 2.0)
        
        # Interpolate to get smooth scaling curve
        interp_func = interp1d(knot_positions, scaling_factors, 
                              kind='quadratic', fill_value="extrapolate")
        smooth_scaling = interp_func(np.arange(seq_len))
        
        # Apply magnitude warping to all features
        for i in range(window.shape[1]):
            augmented[:, i] *= smooth_scaling
            
        return augmented
    
    def window_slicing(self, window, slice_ratio=0.9):
        """
        Take a random slice of the window (temporal cropping).
        This simulates having slightly different observation periods.
        
        Args:
            window (np.ndarray): Input time series
            slice_ratio (float): Minimum ratio of window to keep
        """
        seq_len = len(window)
        min_len = int(seq_len * slice_ratio)
        
        if min_len >= seq_len:
            return window
            
        # Select random slice length
        slice_len = np.random.randint(min_len, seq_len)
        
        # Select random starting position
        start_idx = np.random.randint(0, seq_len - slice_len + 1)
        
        # Extract slice and pad to original length
        sliced = window[start_idx:start_idx + slice_len]
        
        # Pad with edge values to maintain shape
        if len(sliced) < seq_len:
            pad_len = seq_len - len(sliced)
            # Pad at the end with the last value
            padded = np.pad(sliced, ((0, pad_len), (0, 0)), mode='edge')
            return padded
            
        return sliced
    
    def feature_dropout(self, window, dropout_rate=0.1):
        """
        Randomly dropout features (set to zero) to improve robustness.
        This simulates missing or unreliable indicators.
        
        Args:
            window (np.ndarray): Input time series
            dropout_rate (float): Probability of dropping each feature
        """
        augmented = window.copy()
        num_features = window.shape[1]
        
        # Randomly select features to drop
        features_to_drop = np.random.random(num_features) < dropout_rate
        
        # Drop selected features for the entire window
        augmented[:, features_to_drop] = 0
        
        return augmented
    
    def add_outliers(self, window, outlier_prob=0.01, outlier_scale=3.0):
        """
        Add realistic outliers that simulate market events.
        
        Args:
            window (np.ndarray): Input time series
            outlier_prob (float): Probability of an outlier at each time step
            outlier_scale (float): Scale of outliers in terms of std deviations
        """
        augmented = window.copy()
        
        for i in range(window.shape[1]):
            # Calculate feature statistics
            feature_std = np.std(window[:, i])
            feature_mean = np.mean(window[:, i])
            
            if feature_std > 0:
                # Generate outlier mask
                outlier_mask = np.random.random(len(window)) < outlier_prob
                
                # Generate outlier values (both positive and negative)
                outlier_direction = np.random.choice([-1, 1], size=np.sum(outlier_mask))
                outlier_values = feature_mean + outlier_direction * outlier_scale * feature_std
                
                # Apply outliers
                augmented[outlier_mask, i] = outlier_values
                
        return augmented
    
    def volatility_scaling(self, window, scale_range=(0.5, 1.5)):
        """
        Scale the volatility without changing the mean.
        This simulates different market volatility regimes.
        
        Args:
            window (np.ndarray): Input time series
            scale_range (tuple): Range for volatility scaling
        """
        augmented = window.copy()
        scale = np.random.uniform(*scale_range)
        
        for i in range(window.shape[1]):
            # Center the data
            mean = np.mean(window[:, i])
            centered = window[:, i] - mean
            
            # Scale volatility
            scaled = centered * scale
            
            # Restore mean
            augmented[:, i] = scaled + mean
            
        return augmented
    
    def composite_transform(self, window, p=0.5):
        """
        Apply multiple finance-appropriate augmentations with probability p.
        
        Args:
            window (np.ndarray): Input time series
            p (float): Probability of applying each augmentation
        """
        # Ensure input is a copy
        augmented = window.copy()
        
        # Apply augmentations in order of impact (least to most disruptive)
        augmentations = [
            (self.add_calibrated_noise, {'noise_level': 0.01}),
            (self.feature_dropout, {'dropout_rate': 0.1}),
            (self.volatility_scaling, {'scale_range': (0.8, 1.2)}),
            (self.magnitude_warp, {'sigma': 0.1}),
            (self.add_outliers, {'outlier_prob': 0.005}),
        ]
        
        for aug_func, kwargs in augmentations:
            if np.random.random() < p:
                try:
                    augmented = aug_func(augmented, **kwargs)
                except Exception as e:
                    print(f"Warning: {aug_func.__name__} failed: {e}")
                    
        return augmented
    
    def get_augmentation_pair(self, window):
        """
        Generate two different augmented views for contrastive learning.
        Uses different augmentation combinations to create diverse views.
        
        Args:
            window (np.ndarray): Input time series
            
        Returns:
            tuple: (augmented_view1, augmented_view2)
        """
        # View 1: Focus on noise and dropout
        view1 = window.copy()
        view1 = self.add_calibrated_noise(view1, noise_level=0.01)
        if np.random.random() < 0.5:
            view1 = self.feature_dropout(view1, dropout_rate=0.15)
            
        # View 2: Focus on magnitude changes
        view2 = window.copy()
        view2 = self.magnitude_warp(view2, sigma=0.15)
        if np.random.random() < 0.5:
            view2 = self.volatility_scaling(view2, scale_range=(0.7, 1.3))
            
        return view1, view2