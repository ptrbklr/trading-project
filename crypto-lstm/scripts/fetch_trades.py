import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.schema import load_config
from ingestion.kraken.fetcher import KrakenFetcher
from ingestion.kraken.metadata import normalize_pair_name, resolve_kraken_pair

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pair', default=None, help='e.g. XRPEUR (defaults to config.yaml data.symbol)')
    return p.parse_args()

def main():
    args = parse_args()
    if args.pair:
        symbol = args.pair
    else:
        cfg = load_config(PROJECT_ROOT / "config/config.yaml")
        symbol = cfg.data.symbol

    pair = resolve_kraken_pair(symbol)
    pair_name = normalize_pair_name(symbol)

    fetcher = KrakenFetcher(pair, pair_name, PROJECT_ROOT / "data/raw")
    trades = fetcher.incremental_update()
    print(f"Fetched {len(trades)} trades for {pair_name}")

if __name__ == "__main__":
    main()
