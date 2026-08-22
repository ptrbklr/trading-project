import time
from datetime import datetime, timezone

import pandas as pd
import requests

CHARTS_BASE_URL = "https://futures.kraken.com/api/charts/v1"
FUNDING_BASE_URL = "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates"

RESOLUTION_SECONDS = {
    1: ("1m", 60),
    5: ("5m", 300),
    15: ("15m", 900),
    60: ("1h", 3600),
    1440: ("1d", 86400),
}

MAX_CANDLES_PER_REQUEST = 2000


def _rate_limit(last_request, min_interval=1.0):
    delta = time.time() - last_request
    if delta < min_interval:
        time.sleep(min_interval - delta)
    return time.time()


def fetch_ohlc_history(symbol: str, interval_minutes: int, start: datetime, end: datetime) -> pd.DataFrame:
    if interval_minutes not in RESOLUTION_SECONDS:
        raise ValueError(f"Unsupported interval_minutes for futures backfill: {interval_minutes}")
    resolution, step_seconds = RESOLUTION_SECONDS[interval_minutes]

    chunk_span = MAX_CANDLES_PER_REQUEST * step_seconds
    from_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    all_candles = []
    last_request = 0.0
    while from_ts < end_ts:
        to_ts = min(from_ts + chunk_span, end_ts)
        last_request = _rate_limit(last_request)
        url = f"{CHARTS_BASE_URL}/trade/{symbol}/{resolution}"
        r = requests.get(url, params={"from": from_ts, "to": to_ts})
        r.raise_for_status()
        data = r.json()
        candles = data.get("candles", [])
        all_candles.extend(candles)
        from_ts = to_ts

    if not all_candles:
        return pd.DataFrame(columns=["timestamp", "mark_price", "futures_volume"])

    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates(subset=["time"]).sort_values("time")
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms").astype("datetime64[us]")
    df["mark_price"] = pd.to_numeric(df["close"])
    df["futures_volume"] = pd.to_numeric(df["volume"])
    return df[["timestamp", "mark_price", "futures_volume"]].reset_index(drop=True)


def fetch_funding_rate_history(symbol: str) -> pd.DataFrame:
    r = requests.get(FUNDING_BASE_URL, params={"symbol": symbol})
    r.raise_for_status()
    data = r.json()
    if data.get("result") != "success":
        raise RuntimeError(f"Kraken Futures funding rate API error: {data}")

    rates = data.get("rates", [])
    if not rates:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "relative_funding_rate"])

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).astype("datetime64[us]")
    df = df.rename(columns={"relativeFundingRate": "relative_funding_rate", "fundingRate": "funding_rate"})
    return df[["timestamp", "funding_rate", "relative_funding_rate"]].sort_values("timestamp").reset_index(drop=True)
