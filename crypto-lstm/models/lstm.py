import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, use_attention=False, use_classification_head=False):
        super().__init__()
        self.use_attention = use_attention
        self.use_classification_head = use_classification_head
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        if use_attention:
            self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)
        if use_classification_head:
            self.cls_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq, hidden)
        if self.use_attention:
            weights = torch.softmax(self.attn(out), dim=1)  # (batch, seq, 1)
            context = (out * weights).sum(dim=1)            # (batch, hidden)
        else:
            context = out[:, -1, :]                         # last timestep
        reg_out = self.fc(context)
        if self.use_classification_head:
            return reg_out, self.cls_head(context)
        return reg_out
