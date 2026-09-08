"""
data/features.py

Feature engineering utilities for the crypto-lstm project.

Exports:
- add_technical_features(df, prefix, lags=None, rolling_windows=None)
- add_multi_pair_features(df, prefixes, target_prefix, lags=None, rolling_windows=None, drop_raw=True)
- build_feature_set(df, prefixes, target_prefix, lags=None, rolling_windows=None, drop_raw=True)
"""

from typing import List, Optional, Sequence
import pandas as pd
import numpy as np


def _ensure_prefix_columns(df: pd.DataFrame, prefix: str) -> None:
    close_col = f"{prefix}_close"
    if close_col not in df.columns:
        raise KeyError(f"Expected column '{close_col}' for symbol prefix '{prefix}' not found.")


def _safe_log_return(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace(0, np.nan)
    return np.log(s).diff()


def _add_base_features_for_symbol(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close_col = f"{prefix}_close"
    vol_col = f"{prefix}_volume"

    out[f"{prefix}_log_return"] = _safe_log_return(df[close_col])
    out[f"{prefix}_pct_change"] = df[close_col].pct_change()

    if vol_col in df.columns:
        out[f"{prefix}_volume_change"] = df[vol_col].pct_change()

    return out


def _add_rolling_features(df: pd.DataFrame, prefix: str, windows: Sequence[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close_col = f"{prefix}_close"
    for w in windows:
        out[f"{prefix}_ma_{w}"] = df[close_col].rolling(window=w, min_periods=1).mean()
        out[f"{prefix}_std_{w}"] = df[close_col].rolling(window=w, min_periods=1).std().fillna(0.0)
    return out


def _add_lagged_features(df: pd.DataFrame, prefix: str, lags: Sequence[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for k in lags:
        out[f"{prefix}_log_return_lag_{k}"] = df[f"{prefix}_log_return"].shift(k)
        if f"{prefix}_pct_change" in df.columns:
            out[f"{prefix}_pct_change_lag_{k}"] = df[f"{prefix}_pct_change"].shift(k)
        if f"{prefix}_volume_change" in df.columns:
            out[f"{prefix}_volume_change_lag_{k}"] = df[f"{prefix}_volume_change"].shift(k)
    return out


def build_feature_set(
    df: pd.DataFrame,
    prefixes: Sequence[str],
    target_prefix: str,
    lags: Optional[Sequence[int]] = None,
    rolling_windows: Optional[Sequence[int]] = None,
    drop_raw: bool = True,
) -> pd.DataFrame:
    """
    Build a feature set from raw, prefixed candle data.

    - df must contain 'timestamp' and per-symbol prefixed columns like 'btc_close'.
    - prefixes should be lowercased symbol prefixes.
    - target_prefix is the symbol prefix to predict (case-insensitive).
    """
    if lags is None:
        lags = [1, 5, 10]
    if rolling_windows is None:
        rolling_windows = [3, 7, 21]

    if "timestamp" not in df.columns:
        raise KeyError("Input dataframe must contain a 'timestamp' column.")
    if not isinstance(prefixes, (list, tuple)) or len(prefixes) == 0:
        raise ValueError("prefixes must be a non-empty list or tuple of symbol prefixes.")
    if target_prefix.lower() not in [p.lower() for p in prefixes]:
        raise ValueError("target_prefix must be one of the provided prefixes (case-insensitive).")

    base = df.copy().reset_index(drop=True)

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(base["timestamp"]):
        try:
            base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
        except Exception as exc:
            raise ValueError("Could not parse 'timestamp' column to datetime.") from exc

    feature_frames = []
    for p in prefixes:
        prefix = p.lower()
        _ensure_prefix_columns(base, prefix)

        base_feats = _add_base_features_for_symbol(base, prefix)
        rolling_feats = _add_rolling_features(base, prefix, rolling_windows)

        merged = pd.concat([base[["timestamp"]].reset_index(drop=True), base_feats.reset_index(drop=True), rolling_feats.reset_index(drop=True)], axis=1)
        lagged = _add_lagged_features(merged, prefix, lags)

        combined = pd.concat([merged, lagged], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]

        feature_frames.append(combined)

    from functools import reduce
    for i, f in enumerate(feature_frames):
        if "timestamp" not in f.columns:
            raise RuntimeError(f"Feature frame for prefix '{prefixes[i]}' missing 'timestamp' column.")

    merged_all = reduce(lambda left, right: left.merge(right, on="timestamp", how="inner"), feature_frames)

    if drop_raw:
        raw_cols = [c for c in merged_all.columns if c.endswith("_open") or c.endswith("_high") or c.endswith("_low") or c.endswith("_close")]
        merged_all = merged_all.drop(columns=[c for c in raw_cols if c in merged_all.columns], errors="ignore")

    target_log_col = f"{target_prefix.lower()}_log_return"
    if target_log_col not in merged_all.columns:
        close_col_candidate = f"{target_prefix.lower()}_close"
        if close_col_candidate in df.columns:
            merged_all[target_log_col] = _safe_log_return(df[close_col_candidate]).reindex(merged_all.index)
        else:
            raise KeyError(f"Required target log-return column '{target_log_col}' not found and no '{close_col_candidate}' available to compute it.")

    merged_all = merged_all.sort_values("timestamp").reset_index(drop=True)
    return merged_all


# Convenience wrapper: add technical features for a single symbol frame
def add_technical_features(
    df: pd.DataFrame,
    prefix: str,
    lags: Optional[Sequence[int]] = None,
    rolling_windows: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Compute technical features for a single symbol (keeps timestamp).
    Returns a DataFrame with timestamp plus engineered features for that prefix.
    """
    if lags is None:
        lags = [1, 5, 10]
    if rolling_windows is None:
        rolling_windows = [3, 7, 21]

    if "timestamp" not in df.columns:
        raise KeyError("Input dataframe must contain a 'timestamp' column.")
    prefix = prefix.lower()
    _ensure_prefix_columns(df, prefix)

    base_feats = _add_base_features_for_symbol(df, prefix)
    rolling_feats = _add_rolling_features(df, prefix, rolling_windows)
    merged = pd.concat([df[["timestamp"]].reset_index(drop=True), base_feats.reset_index(drop=True), rolling_feats.reset_index(drop=True)], axis=1)
    lagged = _add_lagged_features(merged, prefix, lags)
    combined = pd.concat([merged, lagged], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


# Convenience wrapper: build multi-pair features (alias for build_feature_set)
def add_multi_pair_features(
    df: pd.DataFrame,
    prefixes: Sequence[str],
    target_prefix: str,
    lags: Optional[Sequence[int]] = None,
    rolling_windows: Optional[Sequence[int]] = None,
    drop_raw: bool = True,
) -> pd.DataFrame:
    """
    Build features for multiple symbols and return merged feature frame.
    This is a thin wrapper around build_feature_set for compatibility with older imports.
    """
    return build_feature_set(df, prefixes=prefixes, target_prefix=target_prefix, lags=lags, rolling_windows=rolling_windows, drop_raw=drop_raw)


# Public API
__all__ = [
    "build_feature_set",
    "add_technical_features",
    "add_multi_pair_features",
]
