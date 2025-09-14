import torch
from models.mlp import MLP

def test_forward_pass():
    model = MLP(input_dim=2, hidden_dims=[4], output_dim=1)
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]],dtype=torch.float32)

    with torch.no_grad():
        out=model(X)
        probs=torch.sigmoid(out)

    print("Logits:", out.numpy().round(3))
    print("Probs:", probs.numpy().round(3))

if __name__=="__main__":
    test_forward_pass()