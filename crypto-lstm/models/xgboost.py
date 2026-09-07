import numpy as np
import xgboost as xgb
from training.trainer import Trainer

class XGBoostTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        
    def prepare_data(self):
        # Reuse your existing data preparation
        trainer = Trainer(self.cfg)
        trainer._prepare_data()
        
        # Extract flattened sequences
        X_train_flat = trainer.train_loader.dataset.tensors[0].numpy()
        X_train_flat = X_train_flat.reshape(X_train_flat.shape[0], -1)
        y_train_flat = trainer.train_loader.dataset.tensors[1].numpy().flatten()
        
        X_val_flat = trainer.val_loader.dataset.tensors[0].numpy()
        X_val_flat = X_val_flat.reshape(X_val_flat.shape[0], -1)
        y_val_flat = trainer.val_loader.dataset.tensors[1].numpy().flatten()
        
        return X_train_flat, y_train_flat, X_val_flat, y_val_flat, trainer
    
    def train(self):
        X_train, y_train, X_val, y_val, trainer = self.prepare_data()
        
        # Create DMatrix for faster training
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': ['rmse', 'mae']
        }
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=50
        )
        
        # Predict and compute metrics
        y_pred = self.model.predict(dval)
        # ... compute metrics similar to your existing code
    
    def predict(self, X):
        dtest = xgb.DMatrix(X.reshape(X.shape[0], -1))
        return self.model.predict(dtest)