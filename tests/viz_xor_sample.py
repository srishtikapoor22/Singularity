import torch
from models.mlp import MLP
from utils.activations import ActivationRecorder
from viz.network_viz import draw_network_with_activations


model = MLP(input_dim=2, hidden_dims=[8], output_dim=1)


X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=torch.float32)

rec = ActivationRecorder()

rec.register(model.net, ["0", "2"])

draw_network_with_activations(model, rec, X,[2,8,1],1)
