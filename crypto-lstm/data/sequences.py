import numpy as np

def create_sequences(values: np.ndarray, seq_len: int, target_idx: int):
    """
    Build causal sequences for LSTM/XGBoost training.
    Handles NaNs and infinities created by lagged BTC/ETH features.
    """
    # Clean NaNs and infinities BEFORE sequence slicing
    values = np.nan_to_num(values, nan=np.nan, posinf=np.nan, neginf=np.nan)

    # Drop rows that contain NaNs (lagged features create NaNs at the top)
    mask = ~np.isnan(values).any(axis=1)
    values = values[mask]

    X, y = [], []

    for i in range(len(values) - seq_len):
        seq_x = values[i:i+seq_len]
        seq_y = values[i+seq_len, target_idx]

        # Skip sequences containing NaNs (extra safety)
        if np.isnan(seq_x).any() or np.isnan(seq_y):
            continue

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)
