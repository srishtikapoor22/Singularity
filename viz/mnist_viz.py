import torch
import matplotlib.pyplot as plt
from models.mlp import MLP
from utils.activations import ActivationRecorder
from data.mnist import get_mnist_loaders
import os

def viz_mnistfwd(model_path="models/mnist.pt"):
    #to visualize hoe a nn processes a single handwritten digit
    #load test data
    _,test_loader=get_mnist_loaders(batch_size=1)
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
        pred_label=torch.argmax(output,dim=1).item()
    activations=rec.get_activations()

    #visualisation steup
    os.makedirs("assets",exist_ok=True)
    fig,ax=plt.subplots(1,len(activations),figsize=(10,4))

    if len(activations)==1:
        ax=[axs]

    #plot activations per layer
    for i,(layer_name,act) in enumerate(activations.items()):
        act_flat=act.flatten().numpy()
        ax[i].imshow(act_flat[np.newaxis,:],aspect='auto',cmap='plasma')
        ax[i].set_title(layer_name)
        ax[i].axis('off')

    plt.suptitle(f"True Label: {y} - Predicted Label: {pred_label}")
    out_path="assets/mnist_fwd.png"
    plt.savefig(out_path,bbox_inches='tight',facecolor='black')
    plt.close()

    print(f"✅ Forward pass visualization saved to {out_path}")
    print(f"predicted label: {pred_label}, true label: {y}")

    return out_path
if __name__=="__main__":
    viz_mnistfwd()