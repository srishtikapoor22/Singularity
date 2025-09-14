import torch
from torch import nn,optim
from model.mlp import MLP
import os
import data.mnist import get_mnist_loaders
from utils.activations import ActivationRecorder


def train_mnist(save_path: str="models.mnist.pt", epochs: int=5,lr: float=0.001):
    torch.manual_seed(42)
    train_loader,test_loader=get_mnist_loader(batch_size=64)

    model=MLP(input_dim=28*28,hidden_dims=[128,64],output_dim=10)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)

    for epoch in range(1,epochs+1):
        model.train()
        total_loss=0
        for X,y in train_loader:
            X=X.view(X.size(0),-1)
            outputs=model(X)
            loss=criterion(outputs,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        avg_loss=total_loss/len(train_loader)
        print(f"Epoch {epoch:4d} - loss: {avg_loss:.6f}")

        model.eval()
        correct=0
        with torch.no_grad():
            for X,y in test_loader:
                X=X.view(X.size(0),-1)
                outputs=model(X)
                acts = recorder.get_activations()
                recorder=ActivationRecorder(model)
                pred_labels=torch.argmax(outputs,dim=1)
                correct+=(pred_labels==y).sum().item()

        accuracy=correct/len(test_loader.dataset)
        print(f"Test Accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    torch.save(model.state_dict(),save_path)
    print(f"Model saved to {save_path}")

if __name__=="__main__":
    train_mnist()
