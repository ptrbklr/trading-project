import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _candidate_data_dirs(data_dir: str):
    if os.path.isabs(data_dir):
        return [data_dir]

    candidates = [
        data_dir,
        os.path.join(PROJECT_ROOT, data_dir),
        os.path.join(os.getcwd(), data_dir),
        os.path.join(PROJECT_ROOT, "data", "candles"),
        os.path.join(PROJECT_ROOT, "crypto-lstm", "data", "candles"),
    ]
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def find_candle_file(cfg_data) -> str:
    filename = f"{cfg_data.symbol}_{cfg_data.interval_minutes}min.csv"
    attempted = []
    for base_dir in _candidate_data_dirs(cfg_data.dir):
        path = os.path.join(base_dir, filename)
        attempted.append(path)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Candle file not found. Looked for: {filename} in {attempted}"
    )


def load_futures_snapshots(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Futures snapshot file not found: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    return df


def load_candles(cfg_data) -> pd.DataFrame:
    path = find_candle_file(cfg_data)
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if 'time' in df.columns and 'timestamp' not in df.columns:
        df.rename(columns={'time': 'timestamp'}, inplace=True)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    lookback_hours = getattr(cfg_data, 'lookback_hours', None)
    if lookback_hours and 'timestamp' in df.columns:
        cutoff = df['timestamp'].max() - pd.Timedelta(hours=lookback_hours)
        df = df[df['timestamp'] >= cutoff]

    futures_path = getattr(cfg_data, 'futures_path', None)
    if futures_path and 'timestamp' in df.columns:
        futures_df = load_futures_snapshots(futures_path)
        futures_columns = getattr(cfg_data, 'futures_columns', None)
        if futures_columns:
            futures_df = futures_df[['timestamp'] + [c for c in futures_columns if c in futures_df.columns]]
        df = df.sort_values('timestamp')
        df = pd.merge_asof(df, futures_df, on='timestamp', direction='backward')
        futures_cols = [c for c in futures_df.columns if c != 'timestamp']
        if 'mark_price' in df.columns:
            df['futures_basis'] = df['mark_price'] - df['close']
            futures_cols.append('futures_basis')
        # funding rate/open interest update infrequently; carry the last known value forward
        df[futures_cols] = df[futures_cols].ffill()
        numeric_cols_extra = futures_cols
    else:
        numeric_cols_extra = []

    numeric_cols = [
        c for c in ['open', 'high', 'low', 'close', 'volume', 'trades'] + numeric_cols_extra
        if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])

    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    return df

def load_candles_for_training(data_dir, pair_name, interval_minutes):
    filename = f"{pair_name}_{interval_minutes}min.csv"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Candle file not found: {path}")
    df = pd.read_csv(path)
    # ensure consistent column names
    df.rename(columns=str.lower, inplace=True)
    return df


def find_candle_file_for_symbol(data_dir: str, symbol: str, interval_minutes: int) -> str:
    filename = f"{symbol}_{interval_minutes}min.csv"
    attempted = []
    for base_dir in _candidate_data_dirs(data_dir):
        path = os.path.join(base_dir, filename)
        attempted.append(path)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Candle file not found. Looked for: {filename} in {attempted}"
    )


def _load_pair_ohlcv(data_dir: str, symbol: str, interval_minutes: int) -> pd.DataFrame:
    path = find_candle_file_for_symbol(data_dir, symbol, interval_minutes)
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if 'time' in df.columns and 'timestamp' not in df.columns:
        df.rename(columns={'time': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    numeric_cols = [c for c in ['open', 'high', 'low', 'close', 'volume', 'trades'] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=numeric_cols).sort_values('timestamp')
    prefix = symbol.lower()
    df = df.rename(columns={c: f"{prefix}_{c}" for c in numeric_cols})
    return df[['timestamp'] + [f"{prefix}_{c}" for c in numeric_cols]]


def _add_synthetic_cross(df: pd.DataFrame, base_symbol: str, quote_symbol: str, cross_name: str) -> pd.DataFrame:
    # e.g. BTCETH derived as BTC/ETH from two EUR-denominated legs; used only when no native pair is available
    base = base_symbol.lower()
    quote = quote_symbol.lower()
    prefix = cross_name.lower()
    df = df.copy()
    df[f'{prefix}_open'] = df[f'{base}_open'] / df[f'{quote}_open']
    df[f'{prefix}_close'] = df[f'{base}_close'] / df[f'{quote}_close']
    df[f'{prefix}_high'] = df[f'{base}_high'] / df[f'{quote}_low']
    df[f'{prefix}_low'] = df[f'{base}_low'] / df[f'{quote}_high']
    if f'{base}_volume' in df.columns:
        df[f'{prefix}_volume'] = df[f'{base}_volume']
    return df


def _add_reciprocal_cross(df: pd.DataFrame, source_symbol: str, target_name: str) -> pd.DataFrame:
    # e.g. BTCETH derived as the reciprocal of the native ETHBTC market (ETH priced in BTC)
    src = source_symbol.lower()
    prefix = target_name.lower()
    df = df.copy()
    df[f'{prefix}_open'] = 1.0 / df[f'{src}_open']
    df[f'{prefix}_close'] = 1.0 / df[f'{src}_close']
    df[f'{prefix}_high'] = 1.0 / df[f'{src}_low']
    df[f'{prefix}_low'] = 1.0 / df[f'{src}_high']
    if f'{src}_volume' in df.columns:
        df[f'{prefix}_volume'] = df[f'{src}_volume']
    return df


def _load_funding_rate(data_dir: str, symbol: str) -> pd.DataFrame:
    for base_dir in _candidate_data_dirs(data_dir):
        path = os.path.join(base_dir, f"{symbol}_funding.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['timestamp'])
            return df[['timestamp', 'funding_rate']].rename(
                columns={'funding_rate': f'{symbol.lower()}_funding_rate'}
            )
    return pd.DataFrame(columns=['timestamp', f'{symbol.lower()}_funding_rate'])


def _add_funding_divergence(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    # funding rate history only covers the last ~year; older rows are filled neutral (0)
    # rather than dropped, so the 5-year candle history isn't discarded
    btc_funding = _load_funding_rate(data_dir, 'BTC')
    eth_funding = _load_funding_rate(data_dir, 'ETH')

    df = df.sort_values('timestamp')
    df = pd.merge_asof(df, btc_funding.sort_values('timestamp'), on='timestamp', direction='backward')
    df = pd.merge_asof(df, eth_funding.sort_values('timestamp'), on='timestamp', direction='backward')

    df['btc_funding_rate'] = df['btc_funding_rate'].fillna(0.0)
    df['eth_funding_rate'] = df['eth_funding_rate'].fillna(0.0)
    df['funding_divergence'] = df['btc_funding_rate'] - df['eth_funding_rate']
    return df


def load_multi_pair_candles(cfg_data) -> pd.DataFrame:
    symbols = list(cfg_data.symbols)
    interval = cfg_data.interval_minutes

    merged = None
    for sym in symbols:
        leg = _load_pair_ohlcv(cfg_data.dir, sym, interval)
        merged = leg if merged is None else pd.merge(merged, leg, on='timestamp', how='inner')

    target_symbol = getattr(cfg_data, 'target_symbol', None)
    reciprocal_source = getattr(cfg_data, 'reciprocal_source', None)
    if target_symbol and target_symbol.upper() not in [s.upper() for s in symbols]:
        if reciprocal_source and reciprocal_source.upper() in [s.upper() for s in symbols]:
            merged = _add_reciprocal_cross(merged, reciprocal_source, target_symbol)
        elif len(symbols) == 2:
            merged = _add_synthetic_cross(merged, symbols[0], symbols[1], target_symbol)
        else:
            raise ValueError(
                "target_symbol requires either reciprocal_source (a native pair in symbols) "
                "or exactly two input symbols to derive a synthetic cross"
            )

    if {'BTC', 'ETH'}.issubset({s.upper() for s in symbols}):
        merged = _add_funding_divergence(merged, cfg_data.dir)

    merged = merged.sort_values('timestamp').reset_index(drop=True)
    merged = merged.drop(columns=['timestamp'])
    return merged
