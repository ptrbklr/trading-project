import numpy as np
import pandas as pd

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    # strictly causal indicators (no future leakage)
    df = df.copy()
    df['ma_20'] = df['close'].rolling(window=20, min_periods=20).mean()
    df['ma_50'] = df['close'].rolling(window=50, min_periods=50).mean()
    df['vol_ma_20'] = df['volume'].rolling(window=20, min_periods=20).mean()

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['return_lag_2'] = df['log_return'].shift(1)
    df['return_lag_3'] = df['log_return'].shift(2)
    df['return_lag_5'] = df['log_return'].shift(4)
    df['volatility_20'] = df['log_return'].rolling(window=20, min_periods=20).std()
    df['volume_delta'] = df['volume'].pct_change()

    # normalize by *prior*-bar volatility (shifted) so the denominator never includes
    # the return it's normalizing, keeping the target stationary across volatility regimes
    df['return_vol_norm'] = df['log_return'] / df['volatility_20'].shift(1)

    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    df['macd'] = macd
    df['macd_signal'] = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

