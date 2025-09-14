import torch
import pytest
from models.mlp import MLP

def test_mlp_shapebatch():
    model=MLP(input_dim=2,hidden_dims=[4,4],output_dim=1)
    x=torch.randn(8,2)
    y=model(x)
    assert y.shape==(8,1)

def test_mlp_shapebatch_single():
    model=MLP(input_dim=2,hidden_dims=[4,4],output_dim=1)
    x=torch.randn(1,2)
    y=model(x)
    assert y.shape==(1,1)

def test_mlp_shape_no_batch():
    model=MLP(input_dim=2,hidden_dims=[4,4],output_dim=1)
    x=torch.randn(4,3)
    with pytest.raises(RuntimeError):
        _=model(x)