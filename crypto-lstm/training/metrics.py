import numpy as np

def directional_accuracy(preds, targets, last_close, deadband=0.0):
    preds = preds.flatten()
    targets = targets.flatten()
    last_close = last_close.flatten()
    true_move = targets - last_close
    mask = np.abs(true_move) >= deadband
    if not np.any(mask):
        return float('nan')
    dp = np.sign(preds[mask] - last_close[mask])
    dt = np.sign(true_move[mask])
    return (dp == dt).mean() * 100.0

def compute_metrics(pred_batches, target_batches, last_close_batches, scalers, target_idx, deadband=0.0):
    preds = np.concatenate(pred_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    last_close = np.concatenate(last_close_batches, axis=0)

    # inverse scale target
    preds_full = np.zeros((len(preds), 1))
    targets_full = np.zeros((len(targets), 1))
    last_close_full = np.zeros((len(last_close), 1))
    preds_full[:, 0] = preds.flatten()
    targets_full[:, 0] = targets.flatten()
    last_close_full[:, 0] = last_close.flatten()

    preds_inv = scalers.target_scaler.inverse_transform(preds_full)[:, 0]
    targets_inv = scalers.target_scaler.inverse_transform(targets_full)[:, 0]
    last_close_inv = scalers.target_scaler.inverse_transform(last_close_full)[:, 0]

    mae = np.mean(np.abs(preds_inv - targets_inv))
    mse = np.mean((preds_inv - targets_inv) ** 2)
    dir_acc = directional_accuracy(preds_inv, targets_inv, last_close_inv, deadband=deadband)

    return {
        'mae': mae,
        'mse': mse,
        'directional_accuracy': dir_acc,
    }
