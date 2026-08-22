import os
import pickle
import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.kraken.storage import TradeStorage
from ingestion.kraken.metadata import normalize_pair_name  # as defined earlier
from data.candles_builder import trades_to_df, build_candles

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pair', required=True, help='e.g. BTCEUR')
    p.add_argument('--raw-dir', default=str(PROJECT_ROOT / 'data/raw'))
    p.add_argument('--candles-dir', default=str(PROJECT_ROOT / 'data/candles'))
    p.add_argument(
        '--interval', type=int, nargs='+', default=[1, 5, 15, 60],
        help='One or more candle intervals in minutes (default: 1 5 15 60)',
    )
    return p.parse_args()

def main():
    args = parse_args()
    pair_name = normalize_pair_name(args.pair)
    storage = TradeStorage(args.raw_dir, pair_name)

    trades = storage.load()
    if trades is None:
        raise RuntimeError(f"No raw trades found for {pair_name}")

    df_trades = trades_to_df(trades)

    os.makedirs(args.candles_dir, exist_ok=True)
    for interval in args.interval:
        candles = build_candles(df_trades, interval_minutes=interval)
        filename = f"{pair_name}_{interval}min.csv"
        path = os.path.join(args.candles_dir, filename)
        candles.to_csv(path, index=False)

        print(f"Saved {len(candles)} candles to {path}")

if __name__ == "__main__":
    main()
