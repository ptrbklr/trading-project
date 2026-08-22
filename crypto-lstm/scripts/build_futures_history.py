import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from ingestion.kraken.futures_backfill import fetch_ohlc_history, fetch_funding_rate_history
from ingestion.kraken.futures_metadata import resolve_futures_symbol
from ingestion.kraken.metadata import normalize_pair_name


def parse_args():
    p = argparse.ArgumentParser(description="Backfill Kraken Futures OHLC + funding rate history.")
    p.add_argument("--pair", default="BTC", help="Base asset symbol (default: BTC)")
    p.add_argument("--interval", type=int, default=15, help="Candle interval in minutes (1, 5, 15, or 60)")
    p.add_argument("--days", type=int, default=20, help="Lookback window in days (default: 20)")
    p.add_argument("--quote", default="USD", help="Convert mark_price into this quote currency (default: USD, no conversion)")
    return p.parse_args()


def main():
    args = parse_args()
    symbol = resolve_futures_symbol(args.pair)
    pair_name = normalize_pair_name(args.pair)

    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=args.days)

    print(f"Fetching {symbol} OHLC history ({args.interval}min, last {args.days} days)...")
    ohlc = fetch_ohlc_history(symbol, args.interval, start, end)
    print(f"Fetched {len(ohlc)} candles")

    print(f"Fetching {symbol} funding rate history...")
    funding = fetch_funding_rate_history(symbol)
    funding = funding[funding["timestamp"] >= start]
    print(f"Fetched {len(funding)} funding rate entries")

    merged = pd.merge_asof(ohlc.sort_values("timestamp"), funding.sort_values("timestamp"), on="timestamp", direction="backward")
    merged[["funding_rate", "relative_funding_rate"]] = merged[["funding_rate", "relative_funding_rate"]].ffill()

    if args.quote.upper() == "EUR":
        print("Fetching PF_EURUSD OHLC history for FX conversion...")
        fx = fetch_ohlc_history("PF_EURUSD", args.interval, start, end)
        fx = fx.rename(columns={"mark_price": "eurusd_rate"})[["timestamp", "eurusd_rate"]]
        merged = pd.merge_asof(merged, fx.sort_values("timestamp"), on="timestamp", direction="backward")
        merged["eurusd_rate"] = merged["eurusd_rate"].ffill()
        merged["mark_price"] = merged["mark_price"] / merged["eurusd_rate"]
        merged = merged.drop(columns=["eurusd_rate"])

    out_dir = PROJECT_ROOT / "data" / "candles"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pair_name}_futures_{args.interval}min.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
