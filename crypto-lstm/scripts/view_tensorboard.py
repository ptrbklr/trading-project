#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def resolve_log_dir(explicit_dir=None):
    if explicit_dir:
        path = Path(explicit_dir)
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()

    candidates = [
        PROJECT_ROOT / "logs" / "btc_lstm_15min_v1",
        REPO_ROOT / "logs" / "btc_lstm_15min_v1",
        PROJECT_ROOT / "logs",
        REPO_ROOT / "logs",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / "logs" / "btc_lstm_15min_v1").resolve()


def main():
    parser = argparse.ArgumentParser(description="Launch TensorBoard for the configured training run.")
    parser.add_argument("--logdir", type=str, default=None, help="Optional explicit log directory")
    args = parser.parse_args()

    log_dir = resolve_log_dir(args.logdir)
    if not log_dir.exists():
        print(f"TensorBoard log directory not found: {log_dir}")
        print("Run the trainer first to generate logs.")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(log_dir),
    ]
    print(f"Launching TensorBoard for: {log_dir}")
    print("Open the URL shown in the terminal, usually http://localhost:6006")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
