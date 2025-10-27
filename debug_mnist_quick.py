import torch
from models.mlp import MLP
from data.mnist import get_mnist_loaders

model=MLP(28*28, [128,64], 10)
model.load_state_dict(torch.load("models/mnist.pt",map_location="cpu"))
model.eval()

_,test_loader=get_mnist_loaders(batch_size=256)
correct=0
total=0

with torch.no_grad():
    for X,y in test_loader:
        X=X.view(X.size(0),-1)
        out=model(X)
        preds=out.argmax(dim=1)
        correct+=(preds==y).sum().item()
        total+=X.size(0)
    print("Test accuracy (saved model):", correct/total)

    X,y = next(iter(test_loader))
    X0 = X[0].view(1,-1)
    with torch.no_grad():
        logits = model(X0)
        import torch.nn.functional as F
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    print("true label:", y[0].item(), "pred argmax:", int(probs.argmax()), "probs top5:", probs.argsort()[-5:][::-1], probs.max())