# test_predictions.py
import numpy as np
import torch
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.schema import load_config
from training.trainer import Trainer

def main():
    # Load the trained model - USE ABSOLUTE PATH
    config_path = PROJECT_ROOT / 'config' / 'config.yaml'
    print(f"📂 Loading config from: {config_path}")
    
    if not config_path.exists():
        print(f"❌ Config file not found at: {config_path}")
        print("Please make sure config.yaml exists in the project root.")
        return
    
    cfg = load_config(str(config_path))
    trainer = Trainer(cfg)
        
    # Get some test data (using validation loader as example)
    print("📊 Getting test data...")
    X_test = []
    y_test = []
    for xb, yb in trainer.val_loader:
        X_test.append(xb.numpy())
        y_test.append(yb.numpy())
        if len(X_test) >= 1:  # Just one batch for testing
            break
    
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    
    print(f"✅ Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Get predictions from both models
    print("🔮 Generating predictions...")
    
    # PyTorch prediction
    pytorch_pred = trainer.predict_pytorch(X_test)
    
    # XGBoost prediction
    xgb_pred = trainer.predict_xgboost(X_test)
    
    # Ensemble prediction
    ensemble_pred = trainer.predict_ensemble(X_test)
    
    # Inverse transform to get actual prices
    pytorch_pred_inv = trainer.scalers.target_scaler.inverse_transform(pytorch_pred)
    xgb_pred_inv = trainer.scalers.target_scaler.inverse_transform(xgb_pred)
    ensemble_pred_inv = trainer.scalers.target_scaler.inverse_transform(ensemble_pred)
    y_test_inv = trainer.scalers.target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    print("\n📊 Sample Predictions (in USD):")
    print("=" * 70)
    print(f"{'Index':<6} {'Actual':<12} {'PyTorch':<12} {'XGBoost':<12} {'Ensemble':<12}")
    print("-" * 70)
    for i in range(min(10, len(y_test_inv))):
        print(f"{i:<6} ${y_test_inv[i][0]:<11.2f} ${pytorch_pred_inv[i][0]:<11.2f} ${xgb_pred_inv[i][0]:<11.2f} ${ensemble_pred_inv[i][0]:<11.2f}")
    
    # Calculate errors
    print("\n📊 Error Analysis:")
    print("=" * 70)
    print(f"{'Model':<12} {'MAE (USD)':<12} {'RMSE (USD)':<12}")
    print("-" * 70)
    
    pt_mae = np.mean(np.abs(y_test_inv - pytorch_pred_inv))
    xgb_mae = np.mean(np.abs(y_test_inv - xgb_pred_inv))
    ens_mae = np.mean(np.abs(y_test_inv - ensemble_pred_inv))
    
    pt_rmse = np.sqrt(np.mean((y_test_inv - pytorch_pred_inv)**2))
    xgb_rmse = np.sqrt(np.mean((y_test_inv - xgb_pred_inv)**2))
    ens_rmse = np.sqrt(np.mean((y_test_inv - ensemble_pred_inv)**2))
    
    print(f"{'PyTorch':<12} ${pt_mae:<11.2f} ${pt_rmse:<11.2f}")
    print(f"{'XGBoost':<12} ${xgb_mae:<11.2f} ${xgb_rmse:<11.2f}")
    print(f"{'Ensemble':<12} ${ens_mae:<11.2f} ${ens_rmse:<11.2f}")
    
    # Directional accuracy
    print("\n📈 Directional Accuracy:")
    print("=" * 70)
    
    def directional_accuracy(pred, actual):
        """Calculate directional accuracy (percentage of correct price direction predictions)"""
        pred_delta = pred[1:] - pred[:-1]
        actual_delta = actual[1:] - actual[:-1]
        correct = np.sign(pred_delta) == np.sign(actual_delta)
        return np.mean(correct) * 100
    
    pt_dir_acc = directional_accuracy(pytorch_pred_inv.flatten(), y_test_inv.flatten())
    xgb_dir_acc = directional_accuracy(xgb_pred_inv.flatten(), y_test_inv.flatten())
    ens_dir_acc = directional_accuracy(ensemble_pred_inv.flatten(), y_test_inv.flatten())
    
    print(f"PyTorch Directional Accuracy:  {pt_dir_acc:.2f}%")
    print(f"XGBoost Directional Accuracy:  {xgb_dir_acc:.2f}%")
    print(f"Ensemble Directional Accuracy: {ens_dir_acc:.2f}%")
    
    # Which model performs best?
    print("\n🏆 Best Model Analysis:")
    print("=" * 70)
    
    best_counts = {'PyTorch': 0, 'XGBoost': 0, 'Ensemble': 0}
    for i in range(len(y_test_inv)):
        errors = [
            abs(y_test_inv[i][0] - pytorch_pred_inv[i][0]),
            abs(y_test_inv[i][0] - xgb_pred_inv[i][0]),
            abs(y_test_inv[i][0] - ensemble_pred_inv[i][0])
        ]
        best_idx = np.argmin(errors)
        best_labels = ['PyTorch', 'XGBoost', 'Ensemble']
        best_counts[best_labels[best_idx]] += 1
    
    for model, count in best_counts.items():
        print(f"{model}: {count} times ({count/len(y_test_inv)*100:.1f}%)")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()