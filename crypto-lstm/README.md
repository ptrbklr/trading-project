# crypto-lstm

Project structure for a cryptocurrency LSTM forecasting workflow.

## Overview

This project trains a PyTorch LSTM model on cryptocurrency candle data to forecast future price movement. The code expects a YAML config file, a candle dataset, and a Python environment with the required ML/runtime dependencies.

## Project layout

- `config/config.yaml` — training and data configuration
- `scripts/train.py` — training entry point
- `data/candles/` — candle CSV files (for example `BTC_15min.csv`)
- `training/` — trainer, callbacks, metrics, and logging
- `models/` — model factory and LSTM implementation
- `losses/` — objective functions
- `artifacts/` — saved checkpoints and metrics

## Setup

From the repository root:

```bash
cd /Volumes/ext-data/Python/Projects/trading-project
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r crypto-lstm/requirements.txt
```

If the system Python is locked by macOS policy, use the venv interpreter directly:

```bash
. .venv/bin/activate
```

## Run training

From the repository root:

```bash
.venv/bin/python crypto-lstm/scripts/train.py --config crypto-lstm/config/config.yaml
```

From inside the project folder:

```bash
cd crypto-lstm
. ../.venv/bin/activate
python scripts/train.py --config config/config.yaml
```

## Notes on config paths

The training script resolves relative config paths against the project root automatically, so you can run it from either location without the `ModuleNotFoundError` and path confusion that often happens when launching scripts directly from a nested directory.

## TensorBoard

The trainer writes metrics to the TensorBoard log directory for the configured experiment name in `config/config.yaml`:

```bash
logs/btc_lstm_15min_v1
```

To launch TensorBoard from the repository root:

```bash
cd /Volumes/ext-data/Python/Projects/trading-project
. .venv/bin/activate
tensorboard --logdir crypto-lstm/logs/btc_lstm_15min_v1
```

A helper script is also included:

```bash
cd /Volumes/ext-data/Python/Projects/trading-project
. .venv/bin/activate
python crypto-lstm/scripts/view_tensorboard.py
```

Then open the URL printed by TensorBoard, usually `http://localhost:6006`.

## Paper trading

The paper trader loads a saved checkpoint and reads live Kraken market data. It does not submit exchange orders, but it does make simulated trades from live prices.

From the repository root, run it with the latest checkpoint under the default experiment directory:

```bash
cd /Volumes/ext-data/Python/Projects/trading-project
.venv/bin/python Trading/test-trading.py --iterations 10 --no-plot
```

To use a specific checkpoint directory, pass `--model-dir`:

```bash
.venv/bin/python Trading/test-trading.py \
	--model-dir crypto-lstm/artifacts/btc_lstm_15min_v1/epoch_15_best \
	--interval 15 \
	--balance 10000 \
	--iterations 10
```

The script writes `paper_trading_results_15min.csv`, `paper_trades_log_15min.csv`, and, unless `--no-plot` is used, `paper_trading_results_15min.png` in the directory where it is launched. Use `--help` to see all options.

## Expected behavior

The script reads the YAML config, loads candle data, builds sequence windows, trains the model across epochs, logs validation metrics, and saves the best checkpoint under `artifacts/`.
