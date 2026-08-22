from datetime import datetime, timezone

if __package__:
    from .futures_client import KrakenFuturesClient
    from .futures_storage import FuturesSnapshotStorage
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from ingestion.kraken.futures_client import KrakenFuturesClient
    from ingestion.kraken.futures_storage import FuturesSnapshotStorage

class KrakenFuturesFetcher:
    def __init__(self, symbol, pair_name, raw_dir):
        self.client = KrakenFuturesClient()
        self.storage = FuturesSnapshotStorage(raw_dir, pair_name)
        self.symbol = symbol

    def poll_once(self):
        ticker = self.client.fetch_ticker(self.symbol)
        row = {
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0),
            "mark_price": ticker.get("markPrice"),
            "index_price": ticker.get("indexPrice"),
            "funding_rate": ticker.get("fundingRate"),
            "open_interest": ticker.get("openInterest"),
            "volume_24h": ticker.get("vol24h"),
        }
        self.storage.append(row)
        return row
