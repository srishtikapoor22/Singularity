import torch
from models.mlp import MLP
from train.xor_train import train_xor

def test_xor_trains_predicts():
    model_path="models/xor.pth"
    train_xor(save_path=model_path, epochs=500, lr=0.05)

    model=MLP(input_dim=2,hidden_dims=[8],output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]],dtype=torch.float32)
    with torch.no_grad():
        probs=torch.sigmoid(model(X))
    preds=(probs>0.5).int().squeeze(1).tolist()
    assert preds==[0,1,1,0],f"expected XOR outputs,got {preds}"