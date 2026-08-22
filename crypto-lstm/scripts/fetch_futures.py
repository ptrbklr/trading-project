import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.schema import load_config
from ingestion.kraken.futures_fetcher import KrakenFuturesFetcher
from ingestion.kraken.futures_metadata import resolve_futures_symbol
from ingestion.kraken.metadata import normalize_pair_name

def main():
    cfg = load_config(PROJECT_ROOT / "config/config.yaml")
    symbol = resolve_futures_symbol(cfg.data.symbol)
    pair_name = normalize_pair_name(cfg.data.symbol)

    fetcher = KrakenFuturesFetcher(symbol, pair_name, PROJECT_ROOT / "data/raw")
    row = fetcher.poll_once()
    print(f"Recorded futures snapshot: {row}")

if __name__ == "__main__":
    main()
