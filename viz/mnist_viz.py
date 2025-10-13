import torch
import matplotlib.pyplot as plt
from models.mlp import MLP
from utils.activations import ActivationRecorder
from data.mnist import get_mnist_loaders

def viz_mnistfwd(model_path="models/mnist.pt"):
    #to visualize hoe a nn processes a single handwritten digit
    #load test data
    train_loader,test_loader=get_mnist_loaders(batch_size=1)
    #load the trained model
    model=MLP(28*28,[128,64],10)
    #loading trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    #setup recorder
    rec=ActivationRecorder()
    rec.register(model,['net.0','net.2'])

    #get one test image
    images,labels=next(iter(test_loader))
    x=images.view(1,-1)
    y=labels.item()

    #fwd pass
    with torch.no_grad():
        output=model(x)
    activations=rec.get_activations()

    #viz
    print(f"Predicted Label: {torch.argmax(output).item()} | True Label: {y}")
    for layer_name,act in activations.items():
        act_flat=act.flatten().numpy()
        plt.figure(figsize=(6,3))
        plt.bar(range(len(act_flat)),act_flat)
        plt.title(f"Activations from {layer_name}")
        plt.xlabel("Neurons Index")
        plt.ylabel("Activation Value")
        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    viz_mnistfwd()