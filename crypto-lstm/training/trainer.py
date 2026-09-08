import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import xgboost as xgb
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error  # ← CRITICAL IMPORT
from data.loader import load_candles, load_multi_pair_candles
from data.features import build_feature_set
from data.scaling import fit_scalers, apply_scalers
from data.sequences import create_sequences
from models.factory import create_model
from losses.factory import create_loss
from losses.directional import DirectionalLoss
from losses.hybrid import HybridLoss
from training.callbacks import create_scheduler, EarlyStopping, ModelCheckpoint, TensorBoardLogger
from training.metrics import compute_metrics

def diagnostic_after_validation(pred_batches, target_batches, last_close_batches, scalers, deadband=0.0):
    preds = np.concatenate(pred_batches, axis=0).flatten()
    targets = np.concatenate(target_batches, axis=0).flatten()
    last_close = np.concatenate(last_close_batches, axis=0).flatten()
    preds_full = preds.reshape(-1, 1)
    targets_full = targets.reshape(-1, 1)
    last_close_full = last_close.reshape(-1, 1)
    preds_inv = scalers.target_scaler.inverse_transform(preds_full)[:, 0]
    targets_inv = scalers.target_scaler.inverse_transform(targets_full)[:, 0]
    last_close_inv = scalers.target_scaler.inverse_transform(last_close_full)[:, 0]
    print("Shapes:")
    print(f"  preds: {preds.shape}, targets: {targets.shape}, last_close: {last_close.shape}")
    print("Sample values (first 10):")
    print(f"  preds (scaled): {preds[:10]}")
    print(f"  targets (scaled): {targets[:10]}")
    print(f"  last_close (scaled): {last_close[:10]}")
    print(f"  preds (inv scaled): {preds_inv[:10]}")
    print(f"  targets (inv scaled): {targets_inv[:10]}")
    print(f"  last_close (inv scaled): {last_close_inv[:10]}")
    pred_delta = preds_inv - last_close_inv
    true_delta = targets_inv - last_close_inv
    print("Deltas (first 10):")
    print(f"  pred_delta: {pred_delta[:10]}")
    print(f"  true_delta: {true_delta[:10]}")
    mask = np.abs(true_delta) >= deadband
    print(f"Samples above deadband ({deadband}): {np.sum(mask)} / {len(mask)}")
    pred_sign = np.sign(pred_delta[mask])
    true_sign = np.sign(true_delta[mask])
    dir_acc = 100.0 * (pred_sign == true_sign).mean() if np.any(mask) else float('nan')
    print(f"Directional accuracy on masked samples: {dir_acc:.2f}%")
    print(f"Sign distribution in predictions: {np.unique(pred_sign, return_counts=True)}")
    print(f"Sign distribution in targets: {np.unique(true_sign, return_counts=True)}")
    return dir_acc


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.xgb_model = None  # Add this line
        self.xgb_predictions = None  # Store XGBoost predictions
        self.model = None  # ADD THIS LINE - store PyTorch model

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    def _prepare_data(self):
        symbols = getattr(self.cfg.data, 'symbols', None)
        target_symbol = getattr(self.cfg.data, 'target_symbol', None)

        if symbols:
            df = load_multi_pair_candles(self.cfg.data)
            prefixes = [s.lower() for s in symbols]
            if target_symbol and target_symbol.upper() not in [s.upper() for s in symbols]:
                prefixes.append(target_symbol.lower())
            if self.cfg.data.add_features:
                df = build_feature_set(df, prefixes=prefixes, target_prefix=target_prefix)
            target_prefix = (target_symbol or symbols[0]).lower()
        else:
            df = load_candles(self.cfg.data)
            if self.cfg.data.add_features:
                df = build_feature_set(df, prefixes=(""), target_prefix="")
            target_prefix = None

        self.predict_returns = getattr(self.cfg.data, 'predict_returns', False)
        return_col = f'{target_prefix}_return_vol_norm' if target_prefix else 'return_vol_norm'
        log_return_col = f'{target_prefix}_log_return' if target_prefix else 'log_return'
        close_col = f'{target_prefix}_close' if target_prefix else 'close'
        if self.predict_returns and return_col in df.columns:
            target_col = return_col
        elif self.predict_returns and log_return_col in df.columns:
            target_col = log_return_col
        else:
            target_col = close_col
        target_idx = df.columns.get_loc(target_col) if target_col in df.columns else 4

        values = df.values

        scalers = fit_scalers(values, target_idx)
        values_scaled = apply_scalers(values, scalers)

        X, y = create_sequences(values_scaled, self.cfg.model.seq_len, target_idx)
        X, y = X.astype(np.float32), y.astype(np.float32)

        split = int(len(X) * self.cfg.data.train_split)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        self.train_loader = DataLoader(train_ds, batch_size=self.cfg.training.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.cfg.training.batch_size, shuffle=False)

        self.df = df
        self.scalers = scalers
        self.target_idx = target_idx

        if self.predict_returns:
            # scaled representation of a raw return of exactly 0, used as the directional baseline
            fs = scalers.feature_scaler
            self.zero_baseline_scaled = float(-fs.mean_[target_idx] / fs.scale_[target_idx])
        else:
            self.zero_baseline_scaled = None

    def run(self):
        seed = getattr(self.cfg.training, 'seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self._prepare_data()

        # Train XGBoost ensemble if configured
        use_xgboost = getattr(self.cfg.training, 'use_xgboost', False)
        if use_xgboost:
            self.train_xgboost_ensemble()

        model = create_model(self.cfg.model, input_size=self.df.shape[1]).to(self.device)
        self.model = model  # STORE THE MODEL HERE
        criterion = create_loss(self.cfg.loss)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.training.lr)
        scheduler = create_scheduler(optimizer, self.cfg.training.scheduler)

        early_stopping = EarlyStopping(self.cfg.training.early_stopping)
        checkpoint = ModelCheckpoint(self.cfg)
        logger = TensorBoardLogger(self.cfg)

        for epoch in range(1, self.cfg.training.epochs + 1):
            train_loss = self._train_epoch(model, criterion, optimizer)
            val_loss, metrics = self._validate_epoch(model, criterion)

            if scheduler is not None:
                scheduler.step(val_loss)

            logger.log(epoch, train_loss, val_loss, metrics, optimizer)
            checkpoint.maybe_save(epoch, model, self.scalers, metrics, val_loss)

            print(
                f"Epoch {epoch}/{self.cfg.training.epochs} "
                f"| train_loss {train_loss:.6f} "
                f"| val_loss {val_loss:.6f} "
                f"| dir_acc {metrics['directional_accuracy']:.2f}% "
                f"| lr {optimizer.param_groups[0]['lr']:.6f}"
            )

            if early_stopping.step(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        print("\n💾 Saving trained models...")
        self.save_models()

    def _apply_loss(self, criterion, pred, yb, last_close):
        if isinstance(criterion, (DirectionalLoss, HybridLoss)):
            return criterion(pred, yb, last_close)
        return criterion(pred, yb)

    def _reference_baseline(self, xb, yb):
        if self.predict_returns:
            return torch.full_like(yb, self.zero_baseline_scaled)
        return xb[:, -1, self.target_idx].unsqueeze(1)

    def _forward_and_loss(self, model, criterion, xb, yb, last_close):
        pred = model(xb)
        if isinstance(pred, tuple):
            reg_pred, cls_logit = pred
        else:
            reg_pred, cls_logit = pred, None

        loss = self._apply_loss(criterion, reg_pred, yb, last_close)
        cls_weight = getattr(self.cfg.loss, 'classification_weight', 0.0)
        if cls_logit is not None and cls_weight > 0:
            deadband = getattr(self.cfg.loss, 'dir_deadband', 0.0)
            true_delta = (yb - last_close).squeeze(1)
            # 3-way direction target: 0=down, 1=flat, 2=up
            cls_target = torch.ones_like(true_delta, dtype=torch.long)
            cls_target[true_delta > deadband] = 2
            cls_target[true_delta < -deadband] = 0
            cls_loss = torch.nn.functional.cross_entropy(cls_logit, cls_target)
            loss = loss + cls_weight * cls_loss
        return reg_pred, loss

    def _train_epoch(self, model, criterion, optimizer):
        model.train()
        losses = []
        for xb, yb in self.train_loader:
            xb = xb.to(self.device).float()
            yb = yb.to(self.device).float().unsqueeze(1)
            last_close = self._reference_baseline(xb, yb)
            optimizer.zero_grad()
            _, loss = self._forward_and_loss(model, criterion, xb, yb, last_close)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.grad_clip)
            optimizer.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)

    def _validate_epoch(self, model, criterion):
        model.eval()
        losses = []
        preds, targets, last_closes = [], [], []
        xgb_preds = []  # Store XGBoost predictions

        # Get ensemble weight from config
        ensemble_weight = getattr(self.cfg.training, 'ensemble_weight', 0.3)  # Default 30% XGBoost

        with torch.no_grad():
            for idx, (xb, yb) in enumerate(self.val_loader):
                xb = xb.to(self.device).float()
                yb = yb.to(self.device).float().unsqueeze(1)
                last_close = self._reference_baseline(xb, yb)

                # Get PyTorch prediction
                reg_pred, loss = self._forward_and_loss(model, criterion, xb, yb, last_close)
                losses.append(loss.item())

                # Get XGBoost prediction if available
                if self.xgb_model is not None:
                    xb_flat = xb.cpu().numpy().reshape(xb.shape[0], -1)
                    xgb_batch_pred = self.xgb_model.predict(xb_flat)
                    xgb_preds.append(xgb_batch_pred.reshape(-1, 1))

                    # Combine predictions (weighted average)
                    reg_pred_np = reg_pred.cpu().numpy()
                    combined_pred = (1 - ensemble_weight) * reg_pred_np + ensemble_weight * xgb_batch_pred.reshape(-1, 1)
                    reg_pred = torch.from_numpy(combined_pred).to(self.device)

                preds.append(reg_pred.cpu().numpy())
                targets.append(yb.cpu().numpy())
                last_closes.append(last_close.cpu().numpy())

        val_loss = sum(losses) / len(losses)

        # Compute metrics
        deadband = getattr(self.cfg.loss, 'dir_deadband', 0.0)
        metrics = compute_metrics(preds, targets, last_closes, self.scalers, self.target_idx, deadband=deadband)

        # Add XGBoost metrics if available
        if self.xgb_model is not None and xgb_preds:
            xgb_metrics = self._compute_xgboost_metrics(xgb_preds, targets, last_closes)
            metrics.update({f'xgb_{k}': v for k, v in xgb_metrics.items()})

        return val_loss, metrics

    def _compute_xgboost_metrics(self, xgb_preds, targets, last_closes):
        """Compute metrics for XGBoost predictions separately"""
        preds = np.concatenate(xgb_preds, axis=0).flatten()
        targets = np.concatenate(targets, axis=0).flatten()
        last_close = np.concatenate(last_closes, axis=0).flatten()

        # Inverse transform
        preds_full = preds.reshape(-1, 1)
        targets_full = targets.reshape(-1, 1)
        last_close_full = last_close.reshape(-1, 1)

        preds_inv = self.scalers.target_scaler.inverse_transform(preds_full)[:, 0]
        targets_inv = self.scalers.target_scaler.inverse_transform(targets_full)[:, 0]
        last_close_inv = self.scalers.target_scaler.inverse_transform(last_close_full)[:, 0]

        # Compute metrics
        mae = mean_absolute_error(targets_inv, preds_inv)
        rmse = np.sqrt(mean_squared_error(targets_inv, preds_inv))

        # Directional accuracy
        pred_delta = preds_inv - last_close_inv
        true_delta = targets_inv - last_close_inv
        deadband = getattr(self.cfg.loss, 'dir_deadband', 0.0)
        mask = np.abs(true_delta) >= deadband
        pred_sign = np.sign(pred_delta[mask])
        true_sign = np.sign(true_delta[mask])
        dir_acc = 100.0 * (pred_sign == true_sign).mean() if np.any(mask) else float('nan')

        return {
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': dir_acc
            }

    def train_xgboost_ensemble(self):
        """Train XGBoost model and use it as an ensemble with PyTorch"""
        print("🚀 Training XGBoost ensemble model...")
        sys.stdout.flush()  # Force print to appear immediatel
        # Get XGBoost config
        if hasattr(self.cfg.training, 'extra') and 'xgboost' in self.cfg.training.extra:
            xgb_config = self.cfg.training.extra['xgboost']
        else:
            xgb_config = getattr(self.cfg, 'xgboost', {})  

        # Prepare data for XGBoost
        X_train_flat = []
        y_train_flat = []

        # Get training data
        for xb, yb in self.train_loader:
            X_train_flat.append(xb.numpy().reshape(xb.shape[0], -1))
            y_train_flat.append(yb.numpy().flatten())
    
        X_train_flat = np.concatenate(X_train_flat, axis=0)
        y_train_flat = np.concatenate(y_train_flat, axis=0)
        # Get validation data
    
        X_val_flat = []
        y_val_flat = []
        for xb, yb in self.val_loader:
            X_val_flat.append(xb.numpy().reshape(xb.shape[0], -1))
            y_val_flat.append(yb.numpy().flatten())
        X_val_flat = np.concatenate(X_val_flat, axis=0) if X_val_flat else None
        y_val_flat = np.concatenate(y_val_flat, axis=0) if y_val_flat else None
        # Get XGBoost config from cfg
        xgb_config = getattr(self.cfg, 'xgboost', {})
        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=min(xgb_config.get('n_estimators', 50), 100),  # Reduce estimators
            max_depth=min(xgb_config.get('max_depth', 6), 4),  # Reduce depth
            learning_rate=xgb_config.get('learning_rate', 0.1),
            subsample=xgb_config.get('subsample', 0.5),
            colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
            objective=xgb_config.get('objective', 'reg:squarederror'),
            early_stopping_rounds=xgb_config.get('early_stopping_rounds', 20),
            random_state=xgb_config.get('random_state', 42),
            n_jobs=1,  # Use single thread to avoid memory issues
            tree_method='hist',  # Use histogram-based algorithm (more memory efficient)
        )
        if X_val_flat is not None and y_val_flat is not None:
            self.xgb_model.fit(
                X_train_flat, y_train_flat,
                eval_set=[(X_train_flat, y_train_flat), (X_val_flat, y_val_flat)],
                verbose=xgb_config.get('verbose', False)
            )
        else:
            self.xgb_model.fit(X_train_flat, y_train_flat)
        print(f"✅ XGBoost training complete. Best score: {self.xgb_model.best_score if hasattr(self.xgb_model, 'best_score') else 'N/A'}")
        # Get XGBoost predictions on validation set
        if X_val_flat is not None:
            self.xgb_predictions = self.xgb_model.predict(X_val_flat)
        return self.xgb_model

    def diagnose_xgboost(self):

        """Diagnostic tool to analyze XGBoost performance"""
        if self.xgb_model is None:
            print("❌ XGBoost model not trained yet. Call train_xgboost_ensemble() first.")
            return

        print("\n🔍 XGBoost Diagnostic Report")
        print("=" * 50)

        # Feature importance
        if hasattr(self.xgb_model, 'feature_importances_'):
            importance = self.xgb_model.feature_importances_

            # Get feature names - ensure we have the right number
            n_features = importance.shape[0]
            feature_names = []
            df_columns = self.df.columns

            # Calculate how many features per timestep
            seq_len = self.cfg.model.seq_len
            n_original_features = len(df_columns)

            # For flattened data, each timestep has all features
            features_per_timestep = n_original_features

            for i in range(n_features):
                # Determine which timestep and which feature
                timestep = i // features_per_timestep
                feature_idx = i % features_per_timestep

                if feature_idx < len(df_columns):
                    feature_name = df_columns[feature_idx]
                    if timestep < seq_len:
                        feature_names.append(f"t{timestep}_{feature_name}")
                    else:
                        feature_names.append(f"feature_{i}")
                else:
                    feature_names.append(f"feature_{i}")

            # Top 10 features - ensure we don't go out of bounds
            sorted_idx = np.argsort(importance)[::-1]
            print("\n📊 Top 10 Most Important Features:")
            for i in range(min(10, len(sorted_idx))):
                idx = sorted_idx[i]
                if idx < len(feature_names):
                    feature_name = feature_names[idx]
                else:
                    feature_name = f"feature_{idx}"
                print(f"  {i+1}. {feature_name}: {importance[idx]:.4f}")

        # Model parameters
        print(f"\n📈 Model Performance:")
        print(f"  Best score: {self.xgb_model.best_score if hasattr(self.xgb_model, 'best_score') else 'N/A'}")
        print(f"  Best iteration: {self.xgb_model.best_iteration if hasattr(self.xgb_model, 'best_iteration') else 'N/A'}")

        # Get number of trees
        try:
            booster = self.xgb_model.get_booster()
            num_trees = len(booster.get_dump())
            print(f"  Number of trees: {num_trees}")
        except:
            print(f"  Number of trees: {len(self.xgb_model.get_booster().get_dump())}")

        # If we have predictions stored
        if hasattr(self, 'xgb_predictions') and self.xgb_predictions is not None:
            print(f"  Validation predictions (first 5): {self.xgb_predictions[:5]}...")

        print("=" * 50 + "\n")

    def save_models(self, artifacts_path="artifacts/"):

        """
        Save both PyTorch and XGBoost models along with scalers.

        Args:
            artifacts_path: Directory to save the models (default: "../artifacts/")
        """
        import os
        import pickle
        from pathlib import Path

        # Create artifacts directory
        artifacts_path = Path(artifacts_path)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving models to: {artifacts_path}")

        # Save PyTorch model
        if self.model is not None:
            pytorch_path = artifacts_path / 'pytorch_model.pth'
            torch.save(self.model.state_dict(), pytorch_path)
            print(f"✅ PyTorch model saved to {pytorch_path}")
        else:
            print("⚠️ No PyTorch model to save")

        # Save XGBoost model
        if self.xgb_model is not None:
            xgb_path = artifacts_path / 'xgboost_model.json'
            self.xgb_model.save_model(str(xgb_path))
            print(f"✅ XGBoost model saved to {xgb_path}")
        else:
            print("⚠️ No XGBoost model to save")

        # Save scalers
        if hasattr(self, 'scalers') and self.scalers is not None:
            scaler_path = artifacts_path / 'scalers.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers, f)
            print(f"✅ Scalers saved to {scaler_path}")
        else:
            print("⚠️ No scalers to save")

        # Save config
        if hasattr(self, 'cfg'):
            config_path = artifacts_path / 'config.yaml'
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(self.cfg, f)
            print(f"✅ Config saved to {config_path}")

        # Save training metadata
        metadata = {
            'target_idx': self.target_idx if hasattr(self, 'target_idx') else None,
            'predict_returns': self.predict_returns if hasattr(self, 'predict_returns') else None,
            'seq_len': self.cfg.model.seq_len if hasattr(self.cfg, 'model') else None,
            'features': self.df.shape[1] if hasattr(self, 'df') else None,
        }

        metadata_path = artifacts_path / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Metadata saved to {metadata_path}")

        print("\n🎉 All models and artifacts saved successfully!")

    def load_models(self, artifacts_path="artifacts/"):
        """
        Load saved PyTorch and XGBoost models along with scalers.
        
        Args:
            artifacts_path: Directory containing the saved models (default: "artifacts/")
        
        Returns:
            bool: True if models were loaded successfully, False otherwise
        """
        import os
        import pickle
        from pathlib import Path
        
        artifacts_path = Path(artifacts_path)
        
        if not artifacts_path.exists():
            print(f"❌ Artifacts directory not found: {artifacts_path}")
            return False
        
        print(f"📂 Loading models from: {artifacts_path}")
        loaded = False
        
        # Load PyTorch model
        pytorch_path = artifacts_path / 'pytorch_model.pth'
        if pytorch_path.exists() and self.model is not None:
            self.model.load_state_dict(torch.load(pytorch_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ PyTorch model loaded from {pytorch_path}")
            loaded = True
        else:
            print(f"⚠️ PyTorch model not found at {pytorch_path}")
        
        # Load XGBoost model
        xgb_path = artifacts_path / 'xgboost_model.json'
        if xgb_path.exists():
            if self.xgb_model is not None:
                self.xgb_model.load_model(str(xgb_path))
                print(f"✅ XGBoost model loaded from {xgb_path}")
                loaded = True
        else:
            print(f"⚠️ XGBoost model not found at {xgb_path}")
        
        # Load scalers
        scaler_path = artifacts_path / 'scalers.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scalers = pickle.load(f)
            print(f"✅ Scalers loaded from {scaler_path}")
            loaded = True
        else:
            print(f"⚠️ Scalers not found at {scaler_path}")
        
        # Load metadata
        metadata_path = artifacts_path / 'metadata.pkl'
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"✅ Metadata loaded: {metadata}")
        else:
            print(f"⚠️ Metadata not found at {metadata_path}")
        
        return loaded