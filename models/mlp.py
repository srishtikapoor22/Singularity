from typing import List
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,input_dim: int,hidden_dims: List[int],output_dim: int):
        super().__init__()
        layers: List[nn.Module]=[]
        prev=input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev,h))
            layers.append(nn.ReLU(inplace=True))
            prev=h

        layers.append(nn.Linear(prev,output_dim))
        self.net=nn.Sequential(*layers)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        if x.dim()>2:
            x=x.view(x.size(0),-1)
        return self.net(x)