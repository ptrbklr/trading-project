from datetime import datetime, timedelta, timezone

if __package__:
    from .client import KrakenClient
    from .parser import parse_trades
    from .storage import TradeStorage
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from ingestion.kraken.client import KrakenClient
    from ingestion.kraken.parser import parse_trades
    from ingestion.kraken.storage import TradeStorage

class KrakenFetcher:
    def __init__(self, pair, pair_name, raw_dir):
        self.client = KrakenClient()
        self.storage = TradeStorage(raw_dir, pair_name)
        self.pair = pair

    def incremental_update(self):
        existing = self.storage.load()
        if not existing:
            return self.full_fetch(days=60)

        latest_datetime = max(t[2] for t in existing)
        latest_ts = int(latest_datetime.timestamp())
        since = latest_ts + 1

        all_trades = list(existing)
        for _ in range(500):
            raw, last = self.client.fetch_trades(self.pair, since=since)
            if not raw:
                break

            df = parse_trades(raw)
            latest_datetime = datetime.fromtimestamp(latest_ts, timezone.utc).replace(tzinfo=None)
            df_new = df[df['time'] > latest_datetime]

            if df_new.empty:
                break

            all_trades.extend(df_new.values.tolist())
            latest_ts = int(df_new['time'].max().timestamp())
            since = latest_ts + 1

        self.storage.save(all_trades)
        return all_trades

    def full_fetch(self, days=60):
        start = (datetime.now(timezone.utc) - timedelta(days=days)).replace(tzinfo=None)
        since = int(start.timestamp())

        all_trades = []
        for _ in range(500):
            raw, last = self.client.fetch_trades(self.pair, since=since)
            if not raw:
                break

            df = parse_trades(raw)
            df = df[df['time'] >= start]

            if df.empty:
                break

            all_trades.extend(df.values.tolist())
            since = int(df['time'].max().timestamp()) + 1

        self.storage.save(all_trades)
        return all_trades
