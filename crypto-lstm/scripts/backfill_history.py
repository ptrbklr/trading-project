import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.kraken.fetcher import KrakenFetcher
from ingestion.kraken.metadata import normalize_pair_name, resolve_kraken_pair

# bulk historical OHLCVT dump ends 2026-03-31 23:00 UTC; backfill from there to now
GAP_START_TS = 1774998000

PAIRS = ["BTC", "ETH", "ETHBTC"]

for symbol in PAIRS:
    pair = resolve_kraken_pair(symbol)
    pair_name = normalize_pair_name(symbol)
    fetcher = KrakenFetcher(pair, pair_name, PROJECT_ROOT / "data/raw")
    print(f"Backfilling {pair_name} ({pair}) from gap start...")
    trades = fetcher.backfill_range(GAP_START_TS)
    print(f"{pair_name}: {len(trades)} total trades cached")
