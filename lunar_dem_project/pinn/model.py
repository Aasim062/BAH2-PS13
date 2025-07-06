import torch.nn as nn
import torch

class PINNNet(nn.Module):
    def __init__(self):
        super(PINNNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),  # [brightness, sun_angle]
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)  # Output = elevation
        )

    def forward(self, x):
        return self.model(x)
