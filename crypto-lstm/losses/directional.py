import torch
from torch import nn

class DirectionalLoss(nn.Module):
    def __init__(self, alpha: float, deadband: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.deadband = deadband
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true, last_close):
        # Base regression loss
        base = self.mse(y_pred, y_true)

        # Price deltas relative to last close
        pred_delta = y_pred - last_close
        true_delta = y_true - last_close

        # ignore near-zero true moves so noise doesn't dominate the directional penalty
        mask = (true_delta.abs() > self.deadband).float()

        # Differentiable directional penalty:
        # penalizes negative alignment (wrong direction)
        # but gives zero penalty when aligned
        direction_penalty = (torch.nn.functional.softplus(
            - pred_delta * true_delta) * mask).mean()


        return base + self.alpha * direction_penalty


class TradingLoss(nn.Module):
    def __init__(self,
                 alpha_dir=0.1,     # direction penalty
                 alpha_mag=0.05,    # magnitude realism penalty
                 alpha_sharpe=0.1): # risk-adjusted reward penalty
        super().__init__()
        self.alpha_dir = alpha_dir
        self.alpha_mag = alpha_mag
        self.alpha_sharpe = alpha_sharpe
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true, last_close):
        # -------------------------
        # 1. Base regression loss
        # -------------------------
        base = self.mse(y_pred, y_true)

        # -------------------------
        # 2. Directional loss (smooth)
        # -------------------------
        pred_delta = y_pred - last_close
        true_delta = y_true - last_close

        # penalizes wrong direction proportionally
        direction_penalty = torch.relu(- pred_delta * true_delta).mean()

        # -------------------------
        # 3. Magnitude realism loss
        # -------------------------
        # penalizes predictions that deviate too far from true magnitude
        magnitude_penalty = torch.abs(pred_delta - true_delta).mean()

        # -------------------------
        # 4. Sharpe-style stability loss
        # -------------------------
        # treat predicted returns as a "strategy"
        returns = pred_delta

        mean_ret = returns.mean()
        std_ret = returns.std() + 1e-6

        sharpe_ratio = mean_ret / std_ret

        # we *maximize* Sharpe, so we penalize negative Sharpe
        sharpe_penalty = -sharpe_ratio

        # -------------------------
        # Final combined loss
        # -------------------------
        loss = (
            base
            + self.alpha_dir * direction_penalty
            + self.alpha_mag * magnitude_penalty
            + self.alpha_sharpe * sharpe_penalty
        )

        return loss

