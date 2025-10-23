import torch
import torch.nn as nn

class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),    # input layer
            nn.Tanh(),          # allows nonlinearity in both directions
            nn.Linear(8, 1),    # output layer
            nn.Sigmoid()        # squashes to [0, 1] for probability
        )

    def forward(self, x):
        return self.net(x)
