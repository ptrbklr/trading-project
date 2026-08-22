#!/usr/bin/env python3
"""
Paper Trading Script - Trade based on LSTM predictions using real-time Kraken prices
"""

import os
import sys
import pickle
import json
import re as _re
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse
import requests
import time
import signal
import glob
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYPTO_LSTM_ROOT = PROJECT_ROOT / "crypto-lstm"
if str(CRYPTO_LSTM_ROOT) not in sys.path:
    sys.path.insert(0, str(CRYPTO_LSTM_ROOT))

CONFIG_PATH = CRYPTO_LSTM_ROOT / "config" / "config.yaml"

from models.lstm import LSTMModel
from data.sequences import create_sequences
from data.features import add_technical_features
from config.schema import load_config


class KrakenPriceFetcher:
    """Fetch real-time prices and OHLCV data from Kraken API"""
    
    def __init__(self, pair='XXBTZEUR'):
        self.pair = pair
        self.ticker_url = 'https://api.kraken.com/0/public/Ticker'
        self.ohlc_url = 'https://api.kraken.com/0/public/OHLC'
        self.last_price = None
        self.last_update = None
        
    def get_current_price(self):
        try:
            params = {'pair': self.pair}
            response = requests.get(self.ticker_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('error'):
                print(f"⚠️ Kraken API error: {data['error']}")
                return None
            
            if self.pair not in data['result']:
                print(f"⚠️ Pair '{self.pair}' not found")
                return None
            
            ticker = data['result'][self.pair]
            price = float(ticker['c'][0])
            
            self.last_price = price
            self.last_update = datetime.now()
            
            return price
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching price: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def get_current_price_with_retry(self, max_retries=3, delay=1):
        for attempt in range(max_retries):
            price = self.get_current_price()
            if price is not None:
                return price
            
            if attempt < max_retries - 1:
                print(f"   Retry {attempt + 1}/{max_retries} in {delay} seconds...")
                time.sleep(delay)
        
        return None
    
    def get_historical_ohlc(self, interval=15, since=None):
        try:
            params = {
                'pair': self.pair,
                'interval': interval
            }
            if since:
                params['since'] = since
            
            response = requests.get(self.ohlc_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('error'):
                print(f"⚠️ Kraken API error: {data['error']}")
                return None
            
            if self.pair not in data['result']:
                print(f"⚠️ Pair '{self.pair}' not found")
                return None
            
            candles = data['result'][self.pair]
            if not candles:
                print("⚠️ No OHLCV data received")
                return None
            
            df = pd.DataFrame(
                candles,
                columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
            )
            
            df = df[['time', 'open', 'high', 'low', 'close', 'volume', 'count']]
            df = df.astype({
                'time': 'int64',
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64',
                'count': 'int64'
            })
            
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'count': 'Trades'
            })

            df.columns = [column.lower() for column in df.columns]
            
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching OHLCV: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    def get_completed_candle(self, before_time, interval):
        """Return the latest candle that closed before the requested boundary."""
        df = self.get_historical_ohlc(interval=interval)
        if df is None or df.empty:
            return None

        completed = df[df.index < before_time]
        if completed.empty:
            return None
        return completed.iloc[-1]


class PaperTradingSimulator:
    def __init__(self, model_dir=None, initial_balance=10000, fee_rate=0.001,
                 pair='XXBTZEUR', interval_minutes=15, spread_rate=0.0005,
                 slippage_rate=0.0005):
        if model_dir is None:
            model_dir = str(CRYPTO_LSTM_ROOT / "artifacts")
        self.model_dir = model_dir
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fee_rate = fee_rate
        self.spread_rate = spread_rate
        self.slippage_rate = slippage_rate
        self.max_position_pct = 0.25
        self.risk_per_trade_pct = 0.01
        self.stop_loss_pct = 0.05
        self.position = 0
        self.trades = []
        self.portfolio_history = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pair = pair
        self.interval_minutes = interval_minutes
        self.buy_price = 0
        self.buy_cost = 0
        self.buy_fee = 0
        self.buy_time = None
        self.is_running = True
        self.last_prediction = None
        self.hold_count = 0
        
        # Prediction error tracking
        self.prediction_history = []
        self.prediction_errors = []
        self.accuracy_window = 20
        self.confidence_threshold = 0.5
        
        # Adjust max hold intervals based on interval length
        if interval_minutes <= 5:
            self.max_hold_intervals = 12
        elif interval_minutes <= 15:
            self.max_hold_intervals = 4
        else:
            self.max_hold_intervals = 2
        
        # Initialize price fetcher
        self.price_fetcher = KrakenPriceFetcher(pair)
        self.df = None
        self.initialized = False
        
        # Load model and metadata
        self.load_model()
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        print("\n\n🛑 Stopping trading...")
        self.is_running = False
        
    def find_latest_model(self):
        """Find the latest model for the specified interval."""
        model_dir = Path(self.model_dir)

        if model_dir.is_file():
            model_dir = model_dir.parent

        if model_dir.exists():
            direct_metadata = model_dir / 'metadata.json'
            direct_model = model_dir / 'model.pth'
            if direct_metadata.exists() and direct_model.exists():
                return str(model_dir), 'direct'

            epoch_dirs = sorted(
                [p for p in model_dir.rglob('metadata.json') if p.parent.name.startswith('epoch_')],
                key=lambda p: int(re.search(r'epoch_(\d+)_', p.parent.name).group(1)),
                reverse=True,
            )
            if epoch_dirs:
                return str(epoch_dirs[0].parent), 'epoch'

            legacy_files = [
                (model_dir / 'metadata.pkl', model_dir / 'lstm.pth'),
                (model_dir / f'metadata_{self.interval_minutes}min.pkl', model_dir / f'lstm_{self.interval_minutes}min.pth'),
            ]
            for metadata_path, model_path in legacy_files:
                if metadata_path.exists() and model_path.exists():
                    return str(model_dir), 'legacy'

        return None, None
        
    def load_model(self):
        """Load the trained model and its metadata for the specified interval."""
        model_path, path_type = self.find_latest_model()
        
        if model_path is None:
            print(f"❌ No model found for interval {self.interval_minutes}min")
            print(f"   Looking in: {os.path.abspath(self.model_dir)}")
            raise FileNotFoundError(f"No model found for interval {self.interval_minutes}min")
        
        print(f"📂 Using model directory: {model_path} ({path_type})")
        
        model_dir = Path(model_path)
        metadata_path = model_dir / 'metadata.json'
        legacy_metadata_path = model_dir / f'metadata_{self.interval_minutes}min.pkl'
        legacy_metadata_path2 = model_dir / 'metadata.pkl'
        model_path_pth = model_dir / 'model.pth'
        legacy_model_path = model_dir / f'lstm_{self.interval_minutes}min.pth'
        legacy_model_path2 = model_dir / 'lstm.pth'
        scaler_path = model_dir / 'scalers.pt'

        if not metadata_path.exists() and legacy_metadata_path.exists():
            metadata_path = legacy_metadata_path
        if not metadata_path.exists() and legacy_metadata_path2.exists():
            metadata_path = legacy_metadata_path2
        if not model_path_pth.exists() and legacy_model_path.exists():
            model_path_pth = legacy_model_path
        if not model_path_pth.exists() and legacy_model_path2.exists():
            model_path_pth = legacy_model_path2

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        if not model_path_pth.exists():
            raise FileNotFoundError(f"Model not found: {model_path_pth}")

        if scaler_path.exists():
            self.scaler = torch.load(scaler_path, map_location=self.device, weights_only=False)
        elif (model_dir / 'scaler.pkl').exists():
            with open(model_dir / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f"Scaler not found in {model_dir}")

        close_scaler_path = model_dir / 'close_scaler.pkl'
        if close_scaler_path.exists():
            with open(close_scaler_path, 'rb') as f:
                self.close_scaler = pickle.load(f)
        else:
            self.close_scaler = None

        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        except Exception:
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)

        config = load_config(str(CONFIG_PATH))
        self.model_config = config.model
        self.seq_len = config.model.seq_len
        self.features = [
            feature for feature in config.data.col_names.replace(' ', '').split(',')
            if feature not in {'time', 'timestamp'}
        ] if config.data.col_names else ['open', 'high', 'low', 'close', 'volume', 'trades']
        self.has_features = config.data.add_features

        if isinstance(self.metadata, dict) and 'features' in self.metadata:
            self.features = self.metadata['features']
            self.seq_len = self.metadata.get('seq_len', self.seq_len)
            self.has_features = self.metadata.get('has_features', self.has_features)

        if self.has_features and 'ma_20' not in self.features:
            self.features.extend(['ma_20', 'ma_50', 'vol_ma_20'])

        model_state = torch.load(model_path_pth, map_location=self.device, weights_only=False)

        lstm_layer_keys = sorted(k for k in model_state if k.startswith('lstm.weight_ih_l'))
        if lstm_layer_keys:
            input_size = model_state[lstm_layer_keys[0]].shape[1]
            hidden_size = model_state[lstm_layer_keys[0]].shape[0] // 4
            num_layers = len({k.split('l')[-1].split('.')[0] for k in model_state if k.startswith('lstm.weight_ih_l')})
        else:
            input_size = self.model_config.hidden_size
            hidden_size = self.model_config.hidden_size
            num_layers = self.model_config.num_layers

        use_attention = any(k.startswith('attn.') for k in model_state)

        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.model_config.dropout,
            use_attention=use_attention,
        ).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()

        if isinstance(self.metadata, dict):
            self.seq_len = self.metadata.get('seq_len', self.seq_len)
            self.features = self.metadata.get('features', self.features)
            self.has_features = self.metadata.get('has_features', self.has_features)

        self.model_input_size = self.model.lstm.input_size
        self.features = self.features if isinstance(self.features, list) else list(self.features)
        self.has_features = bool(self.has_features)

        if self.model_input_size != len(self.features):
            print(f"   ℹ️ Model expects {self.model_input_size} features, dataset has {len(self.features)}")
        
        print(f"✅ Model loaded successfully!")
        print(f"   Feature count: {len(self.features)}")
        print(f"   Sequence length: {self.seq_len}")
        print(f"   Hidden size: {self.model.lstm.hidden_size}")
        print(f"   Layers: {self.model.lstm.num_layers}")
        print(f"   Has enhanced features: {self.has_features}")
        print(f"   Has close scaler: {self.close_scaler is not None}")
        
        if isinstance(self.metadata, dict) and 'interval_minutes' in self.metadata:
            print(f"   Model Interval: {self.metadata['interval_minutes']} minutes")
            if self.metadata['interval_minutes'] != self.interval_minutes:
                print(f"   ⚠️  WARNING: Model trained on {self.metadata['interval_minutes']}min but trading on {self.interval_minutes}min")
        
        if isinstance(self.metadata, dict) and 'loss_function' in self.metadata:
            print(f"   Loss Function: {self.metadata['loss_function']}")
        
        if isinstance(self.metadata, dict) and 'final_directional_accuracy' in self.metadata:
            print(f"   Validation Directional Accuracy: {self.metadata['final_directional_accuracy']:.2f}%")
        
        if isinstance(self.metadata, dict) and 'seed' in self.metadata:
            print(f"   Seed: {self.metadata['seed']}")
        
        if isinstance(self.metadata, dict) and 'data_file_basename' in self.metadata:
            print(f"   Training Data: {self.metadata['data_file_basename']}")

    def calculate_prediction_accuracy(self):
        if len(self.prediction_history) < 2:
            return None
        
        recent = self.prediction_history[-self.accuracy_window:]
        if len(recent) < 2:
            return None
        
        errors = [entry['error'] for entry in recent]
        actuals = [entry['actual'] for entry in recent]
        
        if not errors or not actuals:
            return None
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        mape = np.mean([abs(e / a) * 100 for e, a in zip(errors, actuals) if a != 0])
        
        directions_correct = sum(
            1 for entry in recent 
            if (entry['predicted'] > entry['price_at_prediction'] and entry['actual'] > entry['price_at_prediction']) or
               (entry['predicted'] < entry['price_at_prediction'] and entry['actual'] < entry['price_at_prediction'])
        )
        directional_accuracy = (directions_correct / len(recent)) * 100 if recent else 0
        bias = mean_error
        
        confidence = 1.0
        if mape > 10:
            confidence *= 0.3
        elif mape > 5:
            confidence *= 0.6
        elif mape > 2:
            confidence *= 0.8
        
        if directional_accuracy > 70:
            confidence *= 1.2
        elif directional_accuracy < 50:
            confidence *= 0.6
        
        if std_error > np.mean(actuals) * 0.02:
            confidence *= 0.7
        
        confidence = max(0, min(1, confidence))
        
        return {
            'mean_error': mean_error,
            'std_error': std_error,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'bias': bias,
            'recent_count': len(recent),
            'confidence': confidence
        }
    
    def should_trust_prediction(self, predicted_return, current_price):
        accuracy_stats = self.calculate_prediction_accuracy()
        
        if accuracy_stats is None:
            return True, 0.5, "Insufficient history"
        
        confidence = accuracy_stats['confidence']
        
        if confidence < self.confidence_threshold:
            return False, confidence, f"Low confidence ({confidence:.2f})"
        
        if accuracy_stats['mape'] > 10.0:
            return False, confidence, f"High error rate ({accuracy_stats['mape']:.2f}% MAPE)"
        
        if accuracy_stats['directional_accuracy'] < 45:
            return False, confidence, f"Poor directional accuracy ({accuracy_stats['directional_accuracy']:.1f}%)"
        
        if abs(predicted_return) < accuracy_stats['std_error'] / current_price:
            return False, confidence, f"Signal too weak"
        
        return True, confidence, "Confidence OK"
    
    def update_prediction_accuracy(self, predicted_close, actual_price, price_at_prediction):
        error = predicted_close - actual_price
        error_pct = (error / actual_price) * 100 if actual_price != 0 else 0
        
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'predicted': predicted_close,
            'actual': actual_price,
            'price_at_prediction': price_at_prediction,
            'error': error,
            'error_pct': error_pct
        })
        self.prediction_errors.append(error)
        
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]
            self.prediction_errors = self.prediction_errors[-500:]
    
    def calibrate_prediction(self, predicted_price):
        accuracy_stats = self.calculate_prediction_accuracy()
        if accuracy_stats and len(self.prediction_history) > 10:
            bias = accuracy_stats['bias']
            if abs(bias / predicted_price) > 0.005:
                corrected_price = predicted_price - bias
                print(f"   🔧 Calibrated prediction: {predicted_price:.2f} -> {corrected_price:.2f} (bias: {bias:+.2f})")
                return corrected_price
        return predicted_price
    
    def calculate_position_size(self, confidence, portfolio_value):
        min_position_pct = 0.2
        risk_limited_pct = self.risk_per_trade_pct / self.stop_loss_pct
        maximum_pct = min(self.max_position_pct, risk_limited_pct)
        return min_position_pct * maximum_pct + (maximum_pct - min_position_pct * maximum_pct) * confidence

    def initialize_historical_data(self, num_candles=1000):
        print(f"\n📊 Fetching initial historical data from Kraken...")
        
        since = int((datetime.now() - timedelta(minutes=num_candles * self.interval_minutes)).timestamp())
        
        df = self.price_fetcher.get_historical_ohlc(
            interval=self.interval_minutes,
            since=since
        )
        
        if df is None or len(df) == 0:
            print("❌ Failed to fetch initial historical data")
            return False
        
        self.df = df
        print(f"✅ Fetched {len(df)} historical candles from {df.index[0]} to {df.index[-1]}")

        if self.has_features:
            feature_df = add_technical_features(self.df)
            print(f"📊 Technical features ready for {len(feature_df)} candles")
        
        self.initialized = True
        return True

    def get_next_candle_time(self):
        now = datetime.now()
        minutes = now.minute
        minutes_to_next = self.interval_minutes - (minutes % self.interval_minutes)
        seconds_to_next = 60 - now.second
        
        if minutes % self.interval_minutes == 0 and now.second == 0:
            minutes_to_next = self.interval_minutes
            seconds_to_next = 0
        
        next_time = now + timedelta(minutes=minutes_to_next, seconds=seconds_to_next)
        next_time = next_time.replace(microsecond=0)
        
        return next_time

    def wait_until_candle_close(self):
        next_time = self.get_next_candle_time()
        wait_seconds = (next_time - datetime.now()).total_seconds()
        
        if wait_seconds > 0:
            print(f"\n   ⏱️ Waiting until candle close: {next_time.strftime('%H:%M:%S')}")
            print(f"   (Waiting {wait_seconds:.0f} seconds)")
            
            update_interval = 5 if self.interval_minutes <= 5 else 30
            
            while wait_seconds > 0 and self.is_running:
                if wait_seconds > update_interval:
                    time.sleep(update_interval)
                else:
                    time.sleep(wait_seconds)
                wait_seconds = (next_time - datetime.now()).total_seconds()
        
        return next_time

    def prepare_data(self):
        if self.df is None or len(self.df) < self.seq_len + 1:
            return None
        
        prediction_df = self.df

        # Build indicators on a temporary dataframe so raw candles stay available.
        missing_features = [f for f in self.features if f not in prediction_df.columns]
        if missing_features:
            print(f"   ⚠️ Missing {len(missing_features)} features")
            if self.has_features:
                print(f"   📊 Attempting to add technical features...")
                prediction_df = add_technical_features(prediction_df)
                missing_features = [f for f in self.features if f not in prediction_df.columns]
                if missing_features:
                    print(f"   ❌ Still missing {len(missing_features)} features")
                    return None
        
        if len(prediction_df) < self.seq_len + 1:
            return None

        values = prediction_df[self.features].values
        values_scaled = self.scaler.feature_scaler.transform(values)
        X, _ = create_sequences(
            values_scaled,
            seq_len=self.seq_len,
            target_idx=self.features.index('close'),
        )
        return X

    def predict_next_candle(self, X):
        X_tensor = torch.from_numpy(X).float().to(self.device)
        
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).cpu().numpy()
        
        # Use close_scaler for accurate inverse transform
        if self.close_scaler is not None:
            pred_inv = self.close_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        else:
            # The training pipeline saves the target scaler inside scalers.pt.
            pred_inv = self.scaler.target_scaler.inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).flatten()
            print("   ℹ️ Inverse-transformed predictions with the saved target scaler")
        
        return pred_inv

    def get_live_price(self, action="trade"):
        print(f"   🔄 Fetching live price for {action}...")
        price = self.price_fetcher.get_current_price_with_retry()
        
        if price is not None:
            print(f"   ✅ Live price: {price:.2f}")
            return price
        else:
            print(f"   ❌ Failed to get live price for {action}")
            return None

    def update_dataframe_with_candle(self, current_price, candle_time, ohlc_candle=None):
        """
        Update the DataFrame with the completed candle.
        """
        # Initialize DataFrame if None
        if self.df is None:
            print("   ⚠️ DataFrame is None, creating new...")
            self.df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'trades'])
        
        if ohlc_candle is not None:
            new_row = ohlc_candle.to_frame().T
            new_row.index = pd.DatetimeIndex([candle_time])
            self.df = pd.concat([self.df, new_row])
            self.df = self.df[~self.df.index.duplicated(keep='last')].sort_index()
            return self.df

        # Check if this candle time already exists
        if candle_time in self.df.index:
            # Update the existing candle
            self.df.loc[candle_time, 'close'] = current_price
            if current_price > self.df.loc[candle_time, 'high']:
                self.df.loc[candle_time, 'high'] = current_price
            if current_price < self.df.loc[candle_time, 'low']:
                self.df.loc[candle_time, 'low'] = current_price
        else:
            # Add a new candle
            if len(self.df) > 0:
                previous_close = self.df.iloc[-1]['close']
            else:
                previous_close = current_price
            
            # Create new row as a DataFrame
            new_row = pd.DataFrame({
                'open': [previous_close],
                'high': [max(previous_close, current_price)],
                'low': [min(previous_close, current_price)],
                'close': [current_price],
                'volume': [0],
                'trades': [0]
            }, index=[candle_time])
            
            # Append to existing DataFrame
            self.df = pd.concat([self.df, new_row])
        
        # Sort by index (time)
        self.df = self.df.sort_index()
        
        return self.df

    def run_paper_trading(self, max_iterations=None):
        print(f"\n🚀 Starting Paper Trading")
        print("=" * 60)
        print(f"   Model Directory: {os.path.abspath(self.model_dir)}")
        print(f"   Initial Balance: {self.initial_balance:.2f} EUR")
        print(f"   Fee Rate: {self.fee_rate*100:.2f}%")
        print(f"   Interval: {self.interval_minutes} minutes")
        print(f"   Max Hold Intervals: {self.max_hold_intervals}")
        print(f"   Confidence Threshold: {self.confidence_threshold}")
        print(f"   Price Source: LIVE Kraken API")
        print(f"   Pair: {self.pair}")
        print("=" * 60)
        
        if not self.initialize_historical_data():
            print("❌ Failed to initialize. Exiting.")
            return
        
        print("\n⏳ Starting paper trading loop...")
        print("   Press Ctrl+C to stop\n")
        
        iteration = 0
        last_prediction = None
        price_at_prediction = None
        
        # Store the current candle time to avoid duplicate processing
        last_candle_time = None
        
        while self.is_running:
            try:
                iteration += 1
                
                # Wait for candle close
                candle_time = self.wait_until_candle_close()
                
                if not self.is_running:
                    break
                
                # Skip if this candle was already processed
                if last_candle_time == candle_time:
                    print(f"   ⏭️ Skipping duplicate candle: {candle_time}")
                    time.sleep(1)
                    continue
                
                last_candle_time = candle_time
                
                current_time = datetime.now()
                
                print(f"\n{'='*60}")
                print(f"📊 Candle Close - {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Iteration {iteration}")
                print(f"   Interval: {self.interval_minutes} min")
                print(f"{'='*60}")
                
                # Get price at candle close
                completed_candle = self.price_fetcher.get_completed_candle(
                    candle_time,
                    interval=self.interval_minutes,
                )
                if completed_candle is None:
                    print("   ⚠️ Failed to fetch completed OHLC candle, waiting for next candle...")
                    continue

                completed_candle_time = completed_candle.name
                if last_candle_time == completed_candle_time:
                    print(f"   ⏭️ Completed candle already processed: {completed_candle_time}")
                    continue
                last_candle_time = completed_candle_time
                current_price = float(completed_candle['close'])
                
                # Update accuracy with previous prediction
                if last_prediction is not None and price_at_prediction is not None:
                    self.update_prediction_accuracy(last_prediction, current_price, price_at_prediction)
                    print(f"   📊 Updated prediction accuracy")
                    
                    accuracy_stats = self.calculate_prediction_accuracy()
                    if accuracy_stats:
                        print(f"   📈 Recent Accuracy: {accuracy_stats['recent_count']} predictions")
                        print(f"       MAPE: {accuracy_stats['mape']:.2f}%")
                        print(f"       Directional Accuracy: {accuracy_stats['directional_accuracy']:.1f}%")
                        print(f"       Confidence: {accuracy_stats['confidence']:.2f}")
                
                # Update DataFrame with the completed candle
                self.update_dataframe_with_candle(
                    current_price,
                    completed_candle_time,
                    ohlc_candle=completed_candle,
                )
                print(f"   📊 Updated candle: {completed_candle_time.strftime('%H:%M')} Close: {current_price:.2f}")
                print(f"   📊 Total candles in memory: {len(self.df)}")
                
                # Check if we have enough data
                if len(self.df) < self.seq_len + 1:
                    print(f"   ⚠️ Not enough data for prediction. Need {self.seq_len + 1} candles, have {len(self.df)}")
                    last_prediction = None
                    price_at_prediction = None
                    continue
                
                # Make prediction
                print("   📈 Making prediction for next candle...")
                X = self.prepare_data()
                if X is None or len(X) == 0:
                    print("   ❌ Failed to prepare data for prediction")
                    continue
                
                predictions = self.predict_next_candle(X)
                predicted_close_raw = predictions[-1]
                
                # Calibrate prediction
                predicted_close = self.calibrate_prediction(predicted_close_raw)
                predicted_return = (predicted_close / current_price) - 1
                self.last_prediction = predicted_close
                
                print(f"   Current price: {current_price:.2f}")
                print(f"   Raw predicted close: {predicted_close_raw:.2f}")
                print(f"   Calibrated predicted close: {predicted_close:.2f}")
                print(f"   Predicted return: {predicted_return*100:+.2f}%")
                
                # Check confidence
                should_trust, confidence, confidence_reason = self.should_trust_prediction(
                    predicted_return, current_price
                )
                
                if not should_trust:
                    print(f"   ⚠️ Low confidence prediction: {confidence_reason}")
                    print(f"   ⏸️ Skipping trade for this interval")
                    last_prediction = predicted_close
                    price_at_prediction = current_price
                    continue
                
                print(f"   ✅ Prediction confidence: {confidence:.2f} - {confidence_reason}")
                
                # Trading logic
                is_prediction_higher = predicted_close > current_price
                
                if self.position == 0:
                    minimum_edge = (2 * self.fee_rate) + self.spread_rate + (2 * self.slippage_rate)
                    if predicted_return > minimum_edge:
                        print(f"\n   🟢 BUY SIGNAL - Predicted return: {predicted_return*100:.2f}%")
                        
                        buy_price = self.get_live_price("BUY")
                        if buy_price is None:
                            print("   ❌ Failed to get live price for BUY, skipping...")
                            continue
                        
                        portfolio_value = self.balance + (self.position * current_price)
                        position_pct = self.calculate_position_size(confidence, portfolio_value)
                        available_balance = self.balance
                        position_value = available_balance * position_pct
                        execution_buy_price = buy_price * (
                            1 + (self.spread_rate / 2) + self.slippage_rate
                        )
                        position_amount = position_value / (execution_buy_price * (1 + self.fee_rate))
                        buy_fee = position_amount * execution_buy_price * self.fee_rate
                        total_buy_cost = (position_amount * execution_buy_price) + buy_fee
                        
                        self.position = position_amount
                        self.buy_price = execution_buy_price
                        self.buy_cost = total_buy_cost
                        self.buy_fee = buy_fee
                        self.buy_time = current_time
                        self.balance = available_balance - total_buy_cost
                        self.hold_count = 0
                        
                        self.trades.append({
                            'timestamp': current_time,
                            'type': 'BUY',
                            'price': execution_buy_price,
                            'reference_price': buy_price,
                            'amount': self.position,
                            'value': position_amount * execution_buy_price,
                            'fee': buy_fee,
                            'total_cost': total_buy_cost,
                            'balance': self.balance,
                            'predicted_return': predicted_return * 100,
                            'predicted_close': predicted_close,
                            'confidence': confidence,
                            'interval': self.interval_minutes
                        })
                        
                        print(f"   ✅ BUY executed at {buy_price:.2f}")
                        print(f"   Amount: {self.position:.6f} BTC")
                        print(f"   Position size: {position_pct*100:.1f}% of balance")
                        print(f"   Value: {position_value:.2f} EUR")
                        print(f"   Confidence: {confidence:.2f}")
                    else:
                        print(f"\n   ⏸️ No trade - Edge {predicted_return*100:.2f}% is below estimated costs {minimum_edge*100:.2f}%")
                
                else:
                    self.hold_count += 1
                    hold_time = (current_time - self.buy_time).total_seconds() / 60
                    current_position_value = self.position * current_price
                    unrealized_pl = current_position_value - (self.position * self.buy_price)
                    unrealized_pl_pct = (unrealized_pl / (self.position * self.buy_price)) * 100
                    
                    print(f"\n   💼 Current Position:")
                    print(f"   Buy price: {self.buy_price:.2f}")
                    print(f"   Current price: {current_price:.2f}")
                    print(f"   Unrealized P/L: {unrealized_pl:+.2f} EUR ({unrealized_pl_pct:+.2f}%)")
                    print(f"   Hold time: {hold_time:.1f} min")
                    print(f"   Hold count: {self.hold_count}/{self.max_hold_intervals}")
                    
                    should_sell = False
                    sell_reason = ""
                    
                    if predicted_return < 0:
                        should_sell = True
                        sell_reason = f"Negative prediction ({predicted_return*100:.2f}%)"
                    elif not is_prediction_higher:
                        should_sell = True
                        sell_reason = f"Prediction ({predicted_close:.2f}) < Current ({current_price:.2f})"
                    elif self.hold_count >= self.max_hold_intervals:
                        should_sell = True
                        sell_reason = f"Max hold time reached ({self.max_hold_intervals} intervals)"
                    elif current_price > self.buy_price * 1.10:
                        should_sell = True
                        sell_reason = f"Take profit - 10% gain"
                    elif current_price < self.buy_price * 0.95:
                        should_sell = True
                        sell_reason = f"Stop loss - 5% loss"
                    
                    if should_sell:
                        print(f"\n   🔴 SELL SIGNAL - {sell_reason}")
                        
                        sell_price = self.get_live_price("SELL")
                        if sell_price is None:
                            print("   ❌ Failed to get live price for SELL, using current price")
                            sell_price = current_price
                        
                        execution_sell_price = sell_price * (
                            1 - (self.spread_rate / 2) - self.slippage_rate
                        )
                        gross_sell_value = self.position * execution_sell_price
                        sell_fee = gross_sell_value * self.fee_rate
                        net_sell_proceeds = gross_sell_value - sell_fee
                        self.balance += net_sell_proceeds

                        profit_loss = net_sell_proceeds - self.buy_cost
                        profit_loss_pct = (profit_loss / self.buy_cost) * 100
                        
                        self.trades.append({
                            'timestamp': current_time,
                            'type': 'SELL',
                            'price': execution_sell_price,
                            'reference_price': sell_price,
                            'amount': self.position,
                            'value': gross_sell_value,
                            'fee': sell_fee,
                            'buy_fee': self.buy_fee,
                            'total_fees': self.buy_fee + sell_fee,
                            'net_proceeds': net_sell_proceeds,
                            'balance': self.balance,
                            'profit_loss': profit_loss,
                            'profit_loss_pct': profit_loss_pct,
                            'predicted_return': predicted_return * 100,
                            'predicted_close': predicted_close,
                            'confidence': confidence,
                            'sell_reason': sell_reason,
                            'hold_time_minutes': hold_time,
                            'interval': self.interval_minutes
                        })
                        
                        print(f"   ✅ SELL executed at {sell_price:.2f}")
                        print(f"   Buy price: {self.buy_price:.2f}")
                        print(f"   P/L: {profit_loss:+.2f} EUR ({profit_loss_pct:+.2f}%)")
                        print(f"   Balance: {self.balance:.2f} EUR")
                        print(f"   Hold time: {hold_time:.1f} min")
                        
                        self.position = 0
                        self.buy_price = 0
                        self.buy_cost = 0
                        self.buy_fee = 0
                        self.buy_time = None
                        self.hold_count = 0
                    else:
                        print(f"\n   ⏳ HOLDING position - Prediction higher than current")
                        print(f"   Predicted close: {predicted_close:.2f} > Current: {current_price:.2f}")
                        print(f"   Hold count: {self.hold_count}/{self.max_hold_intervals}")
                
                # Record portfolio state
                portfolio_value = self.balance + (self.position * current_price)
                self.portfolio_history.append({
                    'timestamp': current_time,
                    'candle_time': completed_candle_time,
                    'balance': self.balance,
                    'position': self.position,
                    'position_value': self.position * current_price,
                    'portfolio_value': portfolio_value,
                    'price': current_price,
                    'signal': 'BUY' if predicted_return > 0 and self.position > 0 else 'HOLD' if self.position > 0 else 'WAIT',
                    'predicted_return': predicted_return,
                    'predicted_close': predicted_close,
                    'confidence': confidence,
                    'is_prediction_higher': is_prediction_higher,
                    'hold_count': self.hold_count,
                    'interval': self.interval_minutes
                })
                
                print(f"\n   💰 Portfolio Value: {portfolio_value:.2f} EUR")
                print(f"   Balance: {self.balance:.2f} EUR")
                print(f"   Position: {self.position:.6f} BTC")
                
                # Store prediction for next iteration
                last_prediction = predicted_close
                price_at_prediction = current_price
                
                if max_iterations and iteration >= max_iterations:
                    print(f"\n✅ Max iterations ({max_iterations}) reached. Stopping.")
                    break
                
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(10)
        
        self.print_summary()

    def print_summary(self):
        print("\n" + "="*60)
        print("📊 TRADING SUMMARY")
        print("="*60)
        
        if self.portfolio_history:
            final_value = self.portfolio_history[-1]['portfolio_value']
            total_return = (final_value - self.initial_balance) / self.initial_balance * 100
        else:
            final_value = self.initial_balance
            total_return = 0
        
        print(f"   Interval: {self.interval_minutes} minutes")
        print(f"   Initial Balance: €{self.initial_balance:,.2f}")
        print(f"   Final Balance: €{self.balance:,.2f}")
        print(f"   Final Portfolio Value: €{final_value:,.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Total Trades: {len(self.trades) // 2}")
        
        if self.prediction_history:
            accuracy_stats = self.calculate_prediction_accuracy()
            if accuracy_stats:
                print(f"\n   Prediction Accuracy Summary:")
                print(f"   Total Predictions: {len(self.prediction_history)}")
                print(f"   MAPE: {accuracy_stats['mape']:.2f}%")
                print(f"   Directional Accuracy: {accuracy_stats['directional_accuracy']:.1f}%")
                print(f"   Confidence: {accuracy_stats['confidence']:.2f}")
        
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        if sell_trades:
            profits = [t['profit_loss'] for t in sell_trades]
            win_rate = sum(1 for p in profits if p > 0) / len(profits) * 100
            avg_profit = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
            avg_loss = np.mean([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 0
            
            print(f"\n   Trade Statistics:")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Avg Profit: €{avg_profit:+.2f}")
            print(f"   Avg Loss: €{avg_loss:+.2f}")
        
        print("="*60)

    def plot_results(self):
        if not self.portfolio_history:
            print("No data to plot")
            return
        
        df_history = pd.DataFrame(self.portfolio_history)
        df_history.set_index('timestamp', inplace=True)
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        ax1 = axes[0]
        ax1.plot(df_history.index, df_history['portfolio_value'], label='Portfolio Value', linewidth=2, color='blue')
        ax1.axhline(y=self.initial_balance, color='gray', linestyle='--', label=f'Initial Balance')
        ax1.set_title(f'Portfolio Value ({self.interval_minutes}-min intervals)')
        ax1.set_ylabel('Value (€)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.plot(df_history.index, df_history['price'], label='Price', color='black', alpha=0.5, linewidth=1)
        ax2.set_title('Price and Trades')
        ax2.set_ylabel('Price (€)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[2]
        if 'confidence' in df_history.columns:
            ax3.plot(df_history.index, df_history['confidence'], label='Confidence', color='purple', linewidth=2)
            ax3.axhline(y=self.confidence_threshold, color='red', linestyle='--', label=f'Min Confidence')
            ax3.set_title('Prediction Confidence')
            ax3.set_ylabel('Confidence')
            ax3.set_ylim(0, 1.1)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'paper_trading_results_{self.interval_minutes}min.png', dpi=150)
        print(f"\n📊 Results plot saved to: paper_trading_results_{self.interval_minutes}min.png")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Paper Trading with LSTM Predictions')
    parser.add_argument('--model-dir', default=str(CRYPTO_LSTM_ROOT / "artifacts" / "btc_lstm_15min_v1"),
                       help='Model directory')
    parser.add_argument('--balance', type=float, default=10000,
                       help='Initial balance')
    parser.add_argument('--fee', type=float, default=0.001,
                       help='Trading fee rate (0.001 = 0.1%%)')
    parser.add_argument('--spread', type=float, default=0.0005,
                       help='Estimated bid-ask spread rate (0.0005 = 0.05%%)')
    parser.add_argument('--slippage', type=float, default=0.0005,
                       help='Estimated slippage rate per execution (0.0005 = 0.05%%)')
    parser.add_argument('--pair', default='XXBTZEUR',
                       help='Kraken trading pair')
    parser.add_argument('--interval', type=int, default=15, choices=[1, 5, 15, 30, 60],
                       help='Trading interval in minutes (1, 5, 15, 30, 60)')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Maximum number of trading iterations')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Minimum confidence threshold for trading (0-1)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting results')
    
    args = parser.parse_args()
    if not os.path.isabs(args.model_dir):
        args.model_dir = str((PROJECT_ROOT / args.model_dir).resolve())

    simulator = PaperTradingSimulator(
        model_dir=args.model_dir,
        initial_balance=args.balance,
        fee_rate=args.fee,
        pair=args.pair,
        interval_minutes=args.interval,
        spread_rate=args.spread,
        slippage_rate=args.slippage,
    )
    
    simulator.confidence_threshold = args.confidence_threshold
    simulator.run_paper_trading(max_iterations=args.iterations)
    
    if not args.no_plot:
        simulator.plot_results()
    
    if simulator.portfolio_history:
        df_history = pd.DataFrame(simulator.portfolio_history)
        df_history.to_csv(f'paper_trading_results_{args.interval}min.csv', index=False)
        print(f"\n💾 Portfolio history saved to: paper_trading_results_{args.interval}min.csv")
    
    if simulator.trades:
        df_trades = pd.DataFrame(simulator.trades)
        df_trades.to_csv(f'paper_trades_log_{args.interval}min.csv', index=False)
        print(f"💾 Trade log saved to: paper_trades_log_{args.interval}min.csv")


if __name__ == '__main__':
    main()