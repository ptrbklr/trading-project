from torch import nn
from .hybrid import HybridLoss
from .directional import DirectionalLoss
from .base import LogCoshLoss
from .directional import TradingLoss

def create_loss(cfg_loss):
    if cfg_loss.type == 'mse':
        return nn.MSELoss()
    if cfg_loss.type == 'mae':
        return nn.L1Loss()
    if cfg_loss.type == 'directional':
        return DirectionalLoss(alpha=cfg_loss.directional_alpha)
    if cfg_loss.type == 'hybrid':
        return HybridLoss(
            mse_weight=cfg_loss.mse_weight,
            mae_weight=cfg_loss.mae_weight,
            directional_weight=cfg_loss.directional_weight,
            directional_alpha=cfg_loss.directional_alpha,
        )
    if cfg_loss.type == 'logcosh':
        return LogCoshLoss()
    elif loss_cfg.name == "trading":
        return TradingLoss(
            alpha_dir=loss_cfg.alpha_dir,
            alpha_mag=loss_cfg.alpha_mag,
            alpha_sharpe=loss_cfg.alpha_sharpe
        )    
    raise ValueError(f"Unknown loss type: {cfg_loss.type}")
