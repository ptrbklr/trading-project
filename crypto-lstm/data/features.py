"""
data/features.py

Feature engineering utilities for the crypto-lstm project.

Public API:
- build_feature_set(df, prefixes, target_prefix, lags=None, rolling_windows=None, drop_raw=True)

Design goals:
- Work with loader output: a DataFrame that contains a canonical 'timestamp' column
  and per-symbol prefixed columns like 'btc_open', 'btc_close', 'eth_volume', etc.
- Create base features (log returns, pct changes, moving averages, vol) for each symbol.
- Create lagged features for each symbol (no peeking / no future leakage).
- Ensure the target base column `{target_prefix}_log_return` exists (trainer expects it).
- Be defensive: validate inputs, keep timestamp, avoid in-place surprises.
"""

from typing import List, Optional, Sequence
import pandas as pd
import numpy as np


def _ensure_prefix_columns(df: pd.DataFrame, prefix: str) -> None:
    """Raise if expected core columns for a symbol are missing."""
    # We expect at least a close price column for each symbol
    close_col = f"{prefix}_close"
    if close_col not in df.columns:
        raise KeyError(f"Expected column '{close_col}' for symbol prefix '{prefix}' not found.")


def _safe_log_return(series: pd.Series) -> pd.Series:
    """Compute log return safely (handles zeros and NaNs)."""
    # Convert to float, replace zeros with small epsilon to avoid -inf
    s = pd.to_numeric(series, errors="coerce").replace(0, np.nan)
    return np.log(s).diff()


def _add_base_features_for_symbol(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Add base features for a single symbol (in-place on a copy).
    Creates:
      - {prefix}_log_return
      - {prefix}_pct_change (close pct change)
      - {prefix}_volume_change (pct change of volume if volume exists)
    """
    out = df.copy()
    close_col = f"{prefix}_close"
    vol_col = f"{prefix}_volume"

    # Log return on close
    out[f"{prefix}_log_return"] = _safe_log_return(out[close_col])

    # Simple pct change (redundant with log_return but sometimes useful)
    out[f"{prefix}_pct_change"] = out[close_col].pct_change()

    # Volume change if volume column exists
    if vol_col in out.columns:
        out[f"{prefix}_volume_change"] = out[vol_col].pct_change()
    else:
        # Keep column absent if no volume data
        pass

    return out[[c for c in out.columns if c.startswith(prefix + "_")]]


def _add_rolling_features(df: pd.DataFrame, prefix: str, windows: Sequence[int]) -> pd.DataFrame:
    """
    Add rolling statistics for the close price for given windows.
    Produces columns like:
      - {prefix}_ma_{w}
      - {prefix}_std_{w}
    """
    out = pd.DataFrame(index=df.index)
    close_col = f"{prefix}_close"
    for w in windows:
        out[f"{prefix}_ma_{w}"] = df[close_col].rolling(window=w, min_periods=1).mean()
        out[f"{prefix}_std_{w}"] = df[close_col].rolling(window=w, min_periods=1).std().fillna(0.0)
    return out


def _add_lagged_features(df: pd.DataFrame, prefix: str, lags: Sequence[int]) -> pd.DataFrame:
    """
    Create lagged versions of the log return and pct_change for the symbol.
    Produces columns like:
      - {prefix}_log_return_lag_{k}
      - {prefix}_pct_change_lag_{k}
    """
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

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Must contain 'timestamp' and per-symbol prefixed columns
        like 'btc_close', 'eth_open', etc.
    prefixes : sequence of str
        Symbol prefixes to process (e.g., ['btc', 'eth', 'sol']).
        These should already be lowercased and match the loader output.
    target_prefix : str
        The prefix of the symbol we want to predict (e.g., 'sol').
        This function will ensure `{target_prefix}_log_return` exists.
    lags : sequence of int, optional
        Lags to create for lagged features. Defaults to [1, 5, 10].
    rolling_windows : sequence of int, optional
        Window sizes for moving averages and std. Defaults to [3, 7, 21].
    drop_raw : bool
        If True, drop raw price columns (like '{prefix}_close') after features are created
        to reduce leakage risk and model input size.

    Returns
    -------
    pd.DataFrame
        DataFrame with timestamp plus engineered features. Index is reset to original df index.
    """
    if lags is None:
        lags = [1, 5, 10]
    if rolling_windows is None:
        rolling_windows = [3, 7, 21]

    # Defensive checks
    if "timestamp" not in df.columns:
        raise KeyError("Input dataframe must contain a 'timestamp' column.")
    if not isinstance(prefixes, (list, tuple)) or len(prefixes) == 0:
        raise ValueError("prefixes must be a non-empty list or tuple of symbol prefixes.")
    if target_prefix not in [p.lower() for p in prefixes]:
        # allow target_prefix to be provided in any case
        raise ValueError("target_prefix must be one of the provided prefixes (case-insensitive).")

    # Work on a copy to avoid surprising the caller
    base = df.copy().reset_index(drop=True)

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(base["timestamp"]):
        try:
            base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
        except Exception as exc:
            raise ValueError("Could not parse 'timestamp' column to datetime.") from exc

    feature_frames = []
    # For each symbol, compute base features, rolling stats, and lagged features
    for p in prefixes:
        prefix = p.lower()
        # Validate presence of core columns (close at minimum)
        _ensure_prefix_columns(base, prefix)

        # Base features (log returns, pct change, volume change)
        base_feats = _add_base_features_for_symbol(base, prefix)

        # Rolling features on raw close price
        rolling_feats = _add_rolling_features(base, prefix, rolling_windows)

        # Merge base feats and rolling feats into a single frame aligned with index
        merged = pd.concat([base[["timestamp"]], base_feats, rolling_feats], axis=1)

        # Add lagged features (lags operate on log_return/pct_change)
        lagged = _add_lagged_features(merged, prefix, lags)

        # Combine merged + lagged (timestamp kept once)
        combined = pd.concat([merged, lagged], axis=1)

        # Keep only one timestamp column per symbol frame
        combined = combined.loc[:, ~combined.columns.duplicated()]

        feature_frames.append(combined)

    # Now merge all symbol feature frames on timestamp (inner join)
    from functools import reduce

    # Before merging, ensure each frame has 'timestamp' as a column
    for i, f in enumerate(feature_frames):
        if "timestamp" not in f.columns:
            raise RuntimeError(f"Feature frame for prefix '{prefixes[i]}' missing 'timestamp' column.")

    merged_all = reduce(lambda left, right: left.merge(right, on="timestamp", how="inner"), feature_frames)

    # Optionally drop raw price columns to avoid leakage and reduce input size
    if drop_raw:
        raw_cols = [c for c in merged_all.columns if c.endswith("_open") or c.endswith("_high") or c.endswith("_low") or c.endswith("_close")]
        # Keep timestamp and engineered features only
        merged_all = merged_all.drop(columns=[c for c in raw_cols if c in merged_all.columns], errors="ignore")

    # Ensure the base target column exists for the target_prefix
    target_log_col = f"{target_prefix.lower()}_log_return"
    if target_log_col not in merged_all.columns:
        # If it doesn't exist, try to compute it from any remaining close column (fallback)
        close_col_candidate = f"{target_prefix.lower()}_close"
        if close_col_candidate in df.columns:
            merged_all[target_log_col] = _safe_log_return(df[close_col_candidate]).reindex(merged_all.index)
        else:
            raise KeyError(f"Required target log-return column '{target_log_col}' not found and no '{close_col_candidate}' available to compute it.")

    # Final housekeeping: sort by timestamp and reset index
    merged_all = merged_all.sort_values("timestamp").reset_index(drop=True)

    return merged_all
