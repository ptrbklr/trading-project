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


def add_technical_features_for(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # same causal indicators as add_technical_features, computed for one prefixed pair (e.g. "btc_", "btceth_")
    close = df[f'{prefix}_close']
    volume = df[f'{prefix}_volume']

    out = pd.DataFrame(index=df.index)
    out[f'{prefix}_ma_20'] = close.rolling(window=20, min_periods=20).mean()
    out[f'{prefix}_ma_50'] = close.rolling(window=50, min_periods=50).mean()
    out[f'{prefix}_vol_ma_20'] = volume.rolling(window=20, min_periods=20).mean()

    log_return = np.log(close / close.shift(1))
    out[f'{prefix}_log_return'] = log_return
    out[f'{prefix}_return_lag_2'] = log_return.shift(1)
    out[f'{prefix}_return_lag_3'] = log_return.shift(2)
    out[f'{prefix}_return_lag_5'] = log_return.shift(4)
    volatility_20 = log_return.rolling(window=20, min_periods=20).std()
    out[f'{prefix}_volatility_20'] = volatility_20
    out[f'{prefix}_volume_delta'] = volume.pct_change()
    out[f'{prefix}_return_vol_norm'] = log_return / volatility_20.shift(1)
    out[f'{prefix}_momentum_10'] = np.log(close / close.shift(10))

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    out[f'{prefix}_rsi_14'] = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    out[f'{prefix}_macd'] = macd
    out[f'{prefix}_macd_signal'] = macd.ewm(span=9, adjust=False).mean()
    out[f'{prefix}_macd_hist'] = macd - out[f'{prefix}_macd_signal']
    return out


def add_multi_pair_features(df: pd.DataFrame, prefixes) -> pd.DataFrame:
    parts = [df] + [add_technical_features_for(df, prefix) for prefix in prefixes]
    combined = pd.concat(parts, axis=1)

    # rolling correlation regime between BTC and ETH's own returns (relative-strength context)
    if 'btc_log_return' in combined.columns and 'eth_log_return' in combined.columns:
        corr = combined['btc_log_return'].rolling(window=20, min_periods=20).corr(combined['eth_log_return'])
        combined = pd.concat([combined, corr.rename('btc_eth_corr_20')], axis=1)

    combined = combined.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return combined

