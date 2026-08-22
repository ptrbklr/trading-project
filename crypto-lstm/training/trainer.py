import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from data.loader import load_candles
from data.features import add_technical_features
from data.scaling import fit_scalers, apply_scalers
from data.sequences import create_sequences
from models.factory import create_model
from losses.factory import create_loss
from losses.directional import DirectionalLoss
from losses.hybrid import HybridLoss
from training.callbacks import create_scheduler, EarlyStopping, ModelCheckpoint, TensorBoardLogger
from training.metrics import compute_metrics


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _prepare_data(self):
        df = load_candles(self.cfg.data)
        if self.cfg.data.add_features:
            df = add_technical_features(df)

        self.predict_returns = getattr(self.cfg.data, 'predict_returns', False)
        if self.predict_returns and 'return_vol_norm' in df.columns:
            target_col = 'return_vol_norm'
        elif self.predict_returns and 'log_return' in df.columns:
            target_col = 'log_return'
        else:
            target_col = 'close'
        target_idx = df.columns.get_loc(target_col) if target_col in df.columns else 4

        values = df.values

        scalers = fit_scalers(values, target_idx)
        values_scaled = apply_scalers(values, scalers)

        X, y = create_sequences(values_scaled, self.cfg.model.seq_len, target_idx)

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

        model = create_model(self.cfg.model, input_size=self.df.shape[1]).to(self.device)
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
            cls_target = (yb > last_close).float()
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(cls_logit, cls_target)
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
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb = xb.to(self.device).float()
                yb = yb.to(self.device).float().unsqueeze(1)
                last_close = self._reference_baseline(xb, yb)
                reg_pred, loss = self._forward_and_loss(model, criterion, xb, yb, last_close)
                losses.append(loss.item())
                preds.append(reg_pred.cpu().numpy())
                targets.append(yb.cpu().numpy())
                last_closes.append(last_close.cpu().numpy())
        val_loss = sum(losses) / len(losses)
        deadband = getattr(self.cfg.loss, 'dir_deadband', 0.0)
        metrics = compute_metrics(preds, targets, last_closes, self.scalers, self.target_idx, deadband=deadband)
        return val_loss, metrics

    