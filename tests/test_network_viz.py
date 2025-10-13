import torch
from models.mlp import MLP
from utils.activations import ActivationRecorder
from Singularity.viz.network_viz import draw_network_with_activations

def test_draw_returns_fig():
    model = MLP(2, [4], 1)
    rec = ActivationRecorder()
    rec.register(model.net, ["0", "2"])
    X = torch.randn(2,2)
    fig, ax = draw_network_with_activations(model, rec, X, layer_sizes=[2,4,1], sample_idx=1)
    assert fig is not None
