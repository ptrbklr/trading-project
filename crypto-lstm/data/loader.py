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


def find_candle_file(cfg_data, symbol) -> str:
    """
    Find the candle file for a specific symbol using cfg_data.dir and cfg_data.interval_minutes.
    Accepts cfg_data (cfg.data from YAML) and an explicit symbol string.
    """
    filename = f"{symbol.upper()}_{cfg_data.interval_minutes}min.csv"
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
    os.chdir('/Volumes/ext-data/Python/Projects/trading-project/crypto-lstm')
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

def load_multi_asset_candles(cfg_data, extra_pairs):
    """
    Load main asset candles and merge additional assets (BTC, ETH)
    using merge_asof on timestamp.
    """
    # Load main asset
    df_main = load_candles(cfg_data)
    df_main = df_main.sort_values('timestamp')

    merged = df_main.copy()

    for pair in extra_pairs:
        # Build a temporary cfg object for each extra pair
        temp_cfg = cfg_data.__class__(
            symbol=pair,
            interval_minutes=cfg_data.interval_minutes,
            dir=cfg_data.dir,
            lookback_hours=getattr(cfg_data, 'lookback_hours', None),
            futures_path=None
        )

        df_extra = load_candles(temp_cfg).sort_values('timestamp')

        # Prefix columns to avoid collisions
        prefix = pair.lower()
        df_extra = df_extra.add_prefix(prefix + "_")

        # Restore timestamp column name for merge_asof
        df_extra.rename(columns={prefix + "_timestamp": "timestamp"}, inplace=True)

        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            df_extra.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

    return merged


def load_candles_for_training(data_dir, pair_name, interval_minutes):
    """
    Load SOL (or any pair) and merge BTC + ETH candles so that
    features.py can compute lagged BTC/ETH features.
    """
    class Cfg:
        def __init__(self, symbol, interval_minutes, dir):
            self.symbol = symbol
            self.interval_minutes = interval_minutes
            self.dir = dir
            self.lookback_hours = None
            self.futures_path = None

    cfg = Cfg(pair_name, interval_minutes, data_dir)

    # Merge BTC + ETH into the main pair
    df = load_multi_asset_candles(cfg, extra_pairs=["BTC", "ETH"])

    # Ensure consistent column names
    df.rename(columns=str.lower, inplace=True)

    return df


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


from pathlib import Path
import os
import pandas as pd
from functools import reduce

# helper: normalize timestamp column name and parse it
def _ensure_timestamp_col(df):
    # common timestamp column names to check
    candidates = ["timestamp", "time", "date", "datetime"]
    for c in candidates:
        if c in df.columns:
            df = df.rename(columns={c: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
    # if none found, try to infer by dtype (first datetime-like column)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df = df.rename(columns={col: "timestamp"})
            return df
    # last resort: try to parse first column
    first = df.columns[0]
    try:
        df[first] = pd.to_datetime(df[first], utc=True)
        df = df.rename(columns={first: "timestamp"})
        return df
    except Exception:
        raise KeyError("No timestamp column found or parseable in candle file")

def load_candles_for_symbol(cfg_data, symbol):
    """
    Load raw OHLCV candles for a single symbol.
    Ensures a canonical 'timestamp' column is present (datetime, UTC).
    """
    # Defensive validation
    if not hasattr(cfg_data, "dir") or not hasattr(cfg_data, "interval_minutes"):
        raise TypeError("cfg_data must be the YAML data object (cfg.data) with 'dir' and 'interval_minutes'")

    path = find_candle_file(cfg_data, symbol)
    df = pd.read_csv(path)

    # Normalize timestamp column and parse it
    df = _ensure_timestamp_col(df)

    return df

def load_multi_pair_candles(cfg_data):
    """
    Multi-asset loader that keeps a canonical 'timestamp' column and prefixes
    only the non-timestamp columns for each symbol.
    """
    if not hasattr(cfg_data, "symbols"):
        raise TypeError("cfg_data must include 'symbols' (list of symbols). Pass cfg.data from YAML.")

    symbols = list(cfg_data.symbols)
    dfs = []

    for sym in symbols:
        df_sym = load_candles_for_symbol(cfg_data, sym)

        # Keep timestamp column unprefixed, prefix all other columns
        ts = df_sym["timestamp"]
        non_ts = df_sym.drop(columns=["timestamp"])
        non_ts = non_ts.add_prefix(sym.lower() + "_")

        # Reattach timestamp as the first column
        df_prefixed = pd.concat([ts.reset_index(drop=True), non_ts.reset_index(drop=True)], axis=1)
        # Ensure timestamp column name is exactly 'timestamp'
        df_prefixed = df_prefixed.rename(columns={df_prefixed.columns[0]: "timestamp"})

        dfs.append(df_prefixed)

    # Merge all symbols on timestamp (inner join)
    df_merged = reduce(lambda left, right: left.merge(right, on="timestamp", how="inner"), dfs)

    return df_merged
   