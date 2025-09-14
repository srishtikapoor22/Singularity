import torch
from train.mnist_train import train_mnist
from models.mlp import MLP

def test_mnist_training(tmp_path):
    save_path = tmp_path / "mnist.pt"
    train_mnist(save_path=str(save_path), epochs=1)  # tiny test
    model = MLP(input_dim=784, hidden_dims=[128, 64], output_dim=10)
    model.load_state_dict(torch.load(save_path))
