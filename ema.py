from copy import deepcopy

import torch
from torch import nn


class ModelEMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.α = decay
        self.num_updates = 0
        self.model = deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.eval()

    def update(self, model: nn.Module):
        α = min(self.α, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.copy_(α * ema_p.data + (1 - α) * p.data)
            for ema_b, b in zip(self.model.buffers(), model.buffers()):
                ema_b.copy_(α * ema_b.data + (1 - α) * b.data)
        self.num_updates += 1

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
