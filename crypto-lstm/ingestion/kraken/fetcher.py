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
    # safety net only; incremental_update/full_fetch/backfill_range loop until caught up to "now"
    MAX_ITERATIONS = 200_000

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
        now_ts = datetime.now(timezone.utc).timestamp()
        for _ in range(self.MAX_ITERATIONS):
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

            if latest_ts >= now_ts:
                break

        self.storage.save(all_trades)
        return all_trades

    def full_fetch(self, days=60):
        # 1. Compute start timestamp (naive datetime)
        start = (datetime.now(timezone.utc) - timedelta(days=days)).replace(tzinfo=None)
        since = int(start.timestamp())

        all_trades = []
        last_seen_ts = since
        now_ts = datetime.now(timezone.utc).timestamp()

        for _ in range(self.MAX_ITERATIONS):
            raw, last = self.client.fetch_trades(self.pair, since=last_seen_ts)

            # 2. If Kraken returns nothing, stop
            if not raw:
                break

            df = parse_trades(raw)

            # 3. Kraken often returns trades *slightly before* the requested timestamp.
            #    So we allow a small backward tolerance.
            tolerance = timedelta(seconds=5)
            df = df[df['time'] >= (start - tolerance)]

            # 4. If still empty, continue fetching instead of breaking
            if df.empty:
                # Advance cursor using Kraken's "last" field if available
                if last:
                    last_seen_ts = int(last)
                else:
                    # Fallback: move forward 1 second
                    last_seen_ts += 1
                continue

            # 5. Append new trades
            all_trades.extend(df.values.tolist())

            # 6. Move cursor forward
            last_seen_ts = int(df['time'].max().timestamp()) + 1

            if last_seen_ts >= now_ts:
                break

        # 7. Save results
        self.storage.save(all_trades)
        return all_trades

    def backfill_range(self, since_ts):
        """Fetch trades from since_ts forward, merging with any existing cache, up to now."""
        existing = self.storage.load() or []
        existing_times = {t[2] for t in existing}

        all_trades = list(existing)
        last_seen_ts = since_ts
        now_ts = datetime.now(timezone.utc).timestamp()

        for _ in range(self.MAX_ITERATIONS):
            raw, last = self.client.fetch_trades(self.pair, since=last_seen_ts)
            if not raw:
                break

            df = parse_trades(raw)
            if df.empty:
                if last:
                    last_seen_ts = int(last)
                else:
                    last_seen_ts += 1
                continue

            new_rows = [row for row in df.values.tolist() if row[2] not in existing_times]
            all_trades.extend(new_rows)
            existing_times.update(row[2] for row in new_rows)

            last_seen_ts = int(df['time'].max().timestamp()) + 1
            if last_seen_ts >= now_ts:
                break

        all_trades.sort(key=lambda t: t[2])
        self.storage.save(all_trades)
        return all_trades
