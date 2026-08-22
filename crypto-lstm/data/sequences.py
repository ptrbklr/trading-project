import numpy as np

def create_sequences(values: np.ndarray, seq_len: int, target_idx: int):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len, target_idx])
    return np.array(X), np.array(y)
