import numpy as np
from sklearn.preprocessing import StandardScaler

class Scalers:
    def __init__(self, feature_scaler, target_scaler, target_idx, clip_lower, clip_upper):
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.target_idx = target_idx
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

def fit_scalers(values: np.ndarray, target_idx: int, lower_q: float = 0.001, upper_q: float = 0.999) -> Scalers:
    # winsorize per-column before fitting so heavy-tailed features (returns, volume deltas)
    # don't let a handful of extreme outliers dominate the scale
    clip_lower = np.quantile(values, lower_q, axis=0)
    clip_upper = np.quantile(values, upper_q, axis=0)
    clipped = np.clip(values, clip_lower, clip_upper)

    feature_scaler = StandardScaler()
    feature_scaler.fit(clipped)

    target_scaler = StandardScaler()
    target_scaler.fit(clipped[:, [target_idx]])

    return Scalers(feature_scaler, target_scaler, target_idx, clip_lower, clip_upper)

def apply_scalers(values: np.ndarray, scalers: Scalers) -> np.ndarray:
    clipped = np.clip(values, scalers.clip_lower, scalers.clip_upper)
    return scalers.feature_scaler.transform(clipped)
