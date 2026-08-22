from torch import nn

class LogCoshLoss(nn.Module):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return (diff + nn.functional.softplus(-2 * diff) - nn.functional.softplus(-diff)).mean()
