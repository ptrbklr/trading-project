import numpy as np
import pandas as pd


def add_base_features(df: pd.DataFrame, prefixes) -> pd.DataFrame:
    """
    Base per-asset features: log returns, volume delta, volatility, RSI, MACD, etc.
    All prefix-safe: btc_, eth_, sol_, ...
    """
    out = df.copy()

    for prefix in prefixes:
        close = f"{prefix}_close"
        volume = f"{prefix}_volume"

        if close in out.columns:
            # Log return
            out[f"{prefix}_log_return"] = np.log(
                out[close] / out[close].shift(1)
            )

            # 20-bar volatility of log returns
            out[f"{prefix}_volatility_20"] = (
                out[f"{prefix}_log_return"]
                .rolling(window=20, min_periods=20)
                .std()
            )

            # Return normalized by prior-bar volatility
            out[f"{prefix}_return_vol_norm"] = (
                out[f"{prefix}_log_return"]
                / out[f"{prefix}_volatility_20"].shift(1)
            )

            # RSI 14
            delta = out[close].diff()
            gain = (
                delta.clip(lower=0)
                .rolling(window=14, min_periods=14)
                .mean()
            )
            loss = (
                -delta.clip(upper=0)
                .rolling(window=14, min_periods=14)
                .mean()
            )
            rs = gain / loss.replace(0, np.nan)
            out[f"{prefix}_rsi_14"] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = out[close].ewm(span=12, adjust=False).mean()
            ema_26 = out[close].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            out[f"{prefix}_macd"] = macd
            out[f"{prefix}_macd_signal"] = macd.ewm(span=9, adjust=False).mean()
            out[f"{prefix}_macd_hist"] = (
                out[f"{prefix}_macd"] - out[f"{prefix}_macd_signal"]
            )

        if volume in out.columns:
            # Volume delta (pct change)
            out[f"{prefix}_volume_delta"] = out[volume].pct_change()

            # 20-bar volume moving average
            out[f"{prefix}_vol_ma_20"] = (
                out[volume].rolling(window=20, min_periods=20).mean()
            )

    # BTC–ETH correlation regime (if both exist)
    if "btc_log_return" in out.columns and "eth_log_return" in out.columns:
        out["btc_eth_corr_20"] = (
            out["btc_log_return"]
            .rolling(window=20, min_periods=20)
            .corr(out["eth_log_return"])
        )

    return out


def add_lagged_features(df: pd.DataFrame, prefixes) -> pd.DataFrame:
    """
    Prefix-safe lag block for all assets in `prefixes`.
    Includes lags for returns, volume, volume_delta, vol_ma_20, momentum_10, trades.
    """
    out = df.copy()

    for prefix in prefixes:
        log_ret = f"{prefix}_log_return"
        volume = f"{prefix}_volume"
        vol_delta = f"{prefix}_volume_delta"
        vol_ma20 = f"{prefix}_vol_ma_20"
        momentum10 = f"{prefix}_momentum_10"
        trades = f"{prefix}_trades"

        # --- Return lags ---
        if log_ret in out.columns:
            out[f"{prefix}_return_lag_2"] = out[log_ret].shift(1)
            out[f"{prefix}_return_lag_3"] = out[log_ret].shift(2)
            out[f"{prefix}_return_lag_5"] = out[log_ret].shift(4)

            out[f"{prefix}_return_lag_10s"] = out[log_ret].shift(10)
            out[f"{prefix}_return_lag_20s"] = out[log_ret].shift(20)
            out[f"{prefix}_return_lag_30s"] = out[log_ret].shift(30)

        # --- Volume lags ---
        if volume in out.columns:
            out[f"{prefix}_volume_lag_10s"] = out[volume].shift(10)
            out[f"{prefix}_volume_lag_20s"] = out[volume].shift(20)

        # --- Volume delta lags ---
        if vol_delta in out.columns:
            out[f"{prefix}_volume_delta_lag_10s"] = out[vol_delta].shift(10)
            out[f"{prefix}_volume_delta_lag_20s"] = out[vol_delta].shift(20)

        # --- Volatility MA lags ---
        if vol_ma20 in out.columns:
            out[f"{prefix}_vol_ma20_lag_10s"] = out[vol_ma20].shift(10)
            out[f"{prefix}_vol_ma20_lag_20s"] = out[vol_ma20].shift(20)

        # --- Momentum lags ---
        if momentum10 in out.columns:
            out[f"{prefix}_momentum10_lag_10s"] = out[momentum10].shift(10)
            out[f"{prefix}_momentum10_lag_20s"] = out[momentum10].shift(20)

        # --- Trades lags ---
        if trades in out.columns:
            out[f"{prefix}_trades_lag_10s"] = out[trades].shift(10)
            out[f"{prefix}_trades_lag_20s"] = out[trades].shift(20)

    return out


def apply_target_safe_filter(df: pd.DataFrame, target_prefix: str) -> pd.DataFrame:
    """
    Remove any non-lagged (t=0) features for the target asset to avoid leakage.
    Keeps only lagged / derived features.
    """
    out = df.copy()

    # Columns that are raw target series at t=0
    raw_cols = [
        f"{target_prefix}_close",
        f"{target_prefix}_volume",
        f"{target_prefix}_trades",
        f"{target_prefix}_momentum_10",
        f"{target_prefix}_vol_ma_20",
        f"{target_prefix}_volume_delta",
        f"{target_prefix}_log_return",
        f"{target_prefix}_volatility_20",
        f"{target_prefix}_return_vol_norm",
        f"{target_prefix}_rsi_14",
        f"{target_prefix}_macd",
        f"{target_prefix}_macd_signal",
        f"{target_prefix}_macd_hist",
    ]

    to_drop = [c for c in raw_cols if c in out.columns]
    out = out.drop(columns=to_drop)

    return out


def build_feature_set(
    df: pd.DataFrame,
    prefixes=("btc", "eth", "sol"),
    target_prefix="sol",
) -> pd.DataFrame:
    """
    Main entry point: build full feature set with
    - base features
    - lagged features
    - target-safe filtering
    - leakage protection (no future shifts)
    """
    features = df.copy()

    # 1. Base features (per asset)
    features = add_base_features(features, prefixes)

    # 2. Lagged features (per asset)
    features = add_lagged_features(features, prefixes)

    # 3. Target-safe filter (remove raw target t=0 features)
    features = apply_target_safe_filter(features, target_prefix)

    # 4. Clean up: remove inf, drop rows with NaNs from shifting/rolling
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna().reset_index(drop=True)

    return features
