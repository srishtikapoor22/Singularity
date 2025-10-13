import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from models.mlp import MLP 
from data.mnist import get_mnist_loaders
from utils.activations import ActivationRecorder

def viz_tsne(model_path="models/mnist.pt"):
    #loading the trained data model

    #load architecture
    model=MLP(28*28,[128,64],10)
    #load saved parameters
    model.load_state_dict(torch.load(model_path))
    model.eval()


    #load test dataset
    train_loader,test_loader=get_mnist_loaders(batch_size=256)
    #setup activation recorder
    rec=ActivationRecorder(model)

    all_act=[]
    all_labels=[]

    #collect activations
    with torch.no_grad():
        for X,y in test_loader:
            X=X.view(X.size(0),-1)
            #fwd pass through the network
            _=model(X)
            acts=rec.get_activations()
            all_act.append(acts)
            all_labels.append(y)

        #combine all batches into single tensor
        activations=torch.cat(all_axt,dim=0)
        labels=torch.cat(all_labels,dim=0)

        print("Run t-SNE")
        tsne=TSNE(n_components=2,random_state=42,perplexity=30)
        reduced=tsne.fit_transform(activations.numpy())

        #plot
        plt.figure(figsize=(10,8))
        scatter=plt.scatter(reduced[:,0],reduced[:,1],c=labels.numpy(),cmap="tab10",alpha=0.7,s=10)
        plt.colorbar(scatter,ticks=range(10))
        plt.title("t-SNE of MNIST Hidden Layer Activations")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.tight_layout()
        plt.savefig("viz/tsne_mnist.png")
        plt.show()

    if __name__=="__main__:
        viz_tsne()