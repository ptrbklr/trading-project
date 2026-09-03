import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.kraken.storage import TradeStorage
from data.candles_builder import trades_to_df, build_candles

DUMP_DIR = Path("/Volumes/ext-data/Kraken_OHLCVT")
CANDLES_DIR = PROJECT_ROOT / "data/candles"
RAW_DIR = PROJECT_ROOT / "data/raw"

# our_symbol -> kraken bulk-dump filename prefix
PAIR_DUMPS = {
    "BTC": "XBTEUR",
    "ETH": "ETHEUR",
    "ETHBTC": "ETHXBT",
    "BTCUSD": "XBTUSD",
    "ETHUSD": "ETHUSD",
}
INTERVALS = [1, 5, 15, 60, 1440]
YEARS_BACK = 5
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "trades"]


def load_dump(symbol: str, interval: int) -> pd.DataFrame:
    path = DUMP_DIR / f"{PAIR_DUMPS[symbol]}_{interval}.csv"
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    cutoff = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=365 * YEARS_BACK)
    df = df[df["timestamp"] >= cutoff]
    return df.sort_values("timestamp").reset_index(drop=True)


def build_recent_from_raw(symbol: str, interval: int) -> pd.DataFrame:
    storage = TradeStorage(RAW_DIR, symbol)
    trades = storage.load()
    if not trades:
        return pd.DataFrame(columns=COLUMNS)
    df_trades = trades_to_df(trades)
    return build_candles(df_trades, interval_minutes=interval)


def main():
    for symbol in PAIR_DUMPS:
        for interval in INTERVALS:
            historical = load_dump(symbol, interval)
            recent = build_recent_from_raw(symbol, interval)

            if not historical.empty and not recent.empty:
                dump_end = historical["timestamp"].max()
                recent = recent[recent["timestamp"] > dump_end]

            merged = pd.concat([historical, recent], ignore_index=True)
            merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            out_path = CANDLES_DIR / f"{symbol}_{interval}min.csv"
            merged.to_csv(out_path, index=False)
            span = f"{merged['timestamp'].min()} -> {merged['timestamp'].max()}" if not merged.empty else "empty"
            print(f"{symbol}_{interval}min.csv: {len(merged)} rows ({span})")


if __name__ == "__main__":
    main()
