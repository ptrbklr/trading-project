import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from ingestion.kraken.client import KrakenClient
from ingestion.kraken.parser import parse_trades
from ingestion.kraken.storage import TradeStorage
from ingestion.kraken.metadata import normalize_pair_name, resolve_kraken_pair
from data.candles_builder import trades_to_df, build_candles


def fetch_range(client, pair, start, end):
    since = int(start.timestamp())
    end_ts = int(end.timestamp())
    all_trades = []
    while since < end_ts:
        raw, last = client.fetch_trades(pair, since=since)
        if not raw:
            break
        df = parse_trades(raw)
        df = df[df['time'] < end]
        if df.empty:
            break
        all_trades.extend(df.values.tolist())
        new_since = int(df['time'].max().timestamp()) + 1
        if new_since <= since:
            break
        since = new_since
        if len(all_trades) % 200000 < len(raw):
            print(f"  ...fetched {len(all_trades)} trades so far, at {df['time'].max()}")
    return all_trades


def main():
    pair_symbol = "BTC"
    pair = resolve_kraken_pair(pair_symbol)
    pair_name = normalize_pair_name(pair_symbol)

    gap_start = datetime(2026, 3, 31)
    storage = TradeStorage(PROJECT_ROOT / "data/raw", pair_name)
    existing = storage.load() or []
    existing_min = min(t[2] for t in existing) if existing else None
    gap_end = existing_min if existing_min else datetime.now(timezone.utc).replace(tzinfo=None)

    print(f"Fetching gap trades from {gap_start} to {gap_end}...")
    client = KrakenClient()
    gap_trades = fetch_range(client, pair, gap_start, gap_end)
    print(f"Fetched {len(gap_trades)} gap trades")

    all_trades = gap_trades + list(existing)
    all_trades.sort(key=lambda t: t[2])
    storage.save(all_trades)
    print(f"Saved merged raw trades: {len(all_trades)} total")

    df_trades = trades_to_df(all_trades)
    candles = build_candles(df_trades, interval_minutes=1440)
    candles = candles.rename(columns={"timestamp": "timestamp"})
    print(f"Built {len(candles)} daily candles from raw trades ({candles['timestamp'].min()} to {candles['timestamp'].max()})")

    bulk_path = PROJECT_ROOT / "data/candles/BTC_1440min.csv"
    bulk = pd.read_csv(bulk_path, parse_dates=["timestamp"])
    bulk = bulk[bulk["timestamp"] < candles["timestamp"].min()]

    combined = pd.concat([bulk, candles], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    combined.to_csv(bulk_path, index=False)
    print(f"Saved combined daily history: {len(combined)} rows ({combined['timestamp'].min()} to {combined['timestamp'].max()})")


if __name__ == "__main__":
    main()
