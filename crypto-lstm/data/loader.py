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
