# scripts/visualize_xor_sample.py
import torch
from models.mlp import MLP
from utils.activations import ActivationRecorder
from viz.network_viz import draw_network_with_activations

# Build model + sample input
model = MLP(input_dim=2, hidden_dims=[8], output_dim=1)
# If you have trained weights:
# model.load_state_dict(torch.load("models/xor_mlp.pth"))

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=torch.float32)

rec = ActivationRecorder()
# Tell the recorder which linear module names to hook:
rec.register(model.net, ["0", "2"])

# Visualize sample 1 (index 1 => input (0,1))
draw_network_with_activations(model, rec, X,[2,8,1],1)
