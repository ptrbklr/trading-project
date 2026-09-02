from torch import nn
from .directional import DirectionalLoss

class HybridLoss(nn.Module):
    def __init__(self, mse_weight, mae_weight, directional_weight, directional_alpha, dir_deadband=0.0):
        super().__init__()
        self.mse_w = mse_weight
        self.mae_w = mae_weight
        self.dir_w = directional_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.dir = DirectionalLoss(alpha=directional_alpha, deadband=dir_deadband)

    def forward(self, y_pred, y_true, last_close):
        return (
            self.mse_w * self.mse(y_pred, y_true) +
            self.mae_w * self.mae(y_pred, y_true) +
            self.dir_w * self.dir(y_pred, y_true, last_close)
        )
