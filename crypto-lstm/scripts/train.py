import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.schema import load_config
from ingestion.kraken.metadata import normalize_pair_name
from training.trainer import Trainer


def parse_args():
    default_config = PROJECT_ROOT / "config" / "config.yaml"
    p = argparse.ArgumentParser(description="Train the crypto LSTM forecasting model.")
    p.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to config.yaml (default: %(default)s)",
    )
    p.add_argument(
        "--pair",
        default=None,
        help="Override data.symbol, e.g. XRPEUR (defaults to the symbol in --config)",
    )
    return p.parse_args()


def resolve_config_path(raw_path: Path) -> Path:
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        PROJECT_ROOT / raw_path,
        REPO_ROOT / raw_path,
        Path.cwd() / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / raw_path).resolve()


def apply_pair_override(cfg, pair: str):
    symbol = normalize_pair_name(pair)
    cfg.data.symbol = symbol
    cfg.experiment_name = f"{symbol.lower()}_lstm_{cfg.data.interval_minutes}min_v1"

    if cfg.data.futures_path:
        futures_name = f"{symbol}_futures_{cfg.data.interval_minutes}min.csv"
        futures_path = PROJECT_ROOT / cfg.data.dir / futures_name
        cfg.data.futures_path = str(futures_path) if futures_path.exists() else None


def main():
    args = parse_args()
    config_path = resolve_config_path(args.config)
    cfg = load_config(str(config_path))
    if args.pair:
        apply_pair_override(cfg, args.pair)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
