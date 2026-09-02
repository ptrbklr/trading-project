import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.kraken.futures_backfill import fetch_funding_rate_history

OUT_DIR = PROJECT_ROOT / "data" / "candles"

PAIRS = {"BTC": "PF_XBTUSD", "ETH": "PF_ETHUSD"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for symbol, futures_symbol in PAIRS.items():
        df = fetch_funding_rate_history(futures_symbol)
        out_path = OUT_DIR / f"{symbol}_funding.csv"
        df.to_csv(out_path, index=False)
        print(f"{symbol}: saved {len(df)} funding rate rows to {out_path}")


if __name__ == "__main__":
    main()
