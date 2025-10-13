import torch
import plotly.graph_objects as go
import numpy as np
from models.mlp import MLP
from utils.activations import ActivationRecorder
from data.mnist import get_mnist_loaders

#3d viz function
def mnist_3d_viz(images, labels, model):
    #load model and data
    model.eval()

    #load one test image
    
    x=images[0].view(1,-1)
    y=labels[0].item()

    #capture activations
    rec=ActivationRecorder()
    rec.register(model,['net.0','net.2'])
    with torch.no_grad():
        output=model(x)
    activations=rec.get_activations()

    #calculate 3d coords
    layer_sizes=[784,128,64,10]
    z_spacing=np.linspace(0,len(layer_sizes)-1,len(layer_sizes))

    coords=[]
    for layer_idx,size in enumerate(layer_sizes):
        x_coords=np.linspace(-1,1,size)
        y_coords=np.random.uniform(-1,1,size)
        z_coords=np.ones(size)*z_spacing[layer_idx]
        coords.append((x_coords,y_coords,z_coords))

    #create fig
    fig=go.Figure()

    #color coded activations
    for layer_idx,(x_c,y_c,z_c) in enumerate(coords):
        act_values=None
        if layer_idx==1:
            if 'net.0' in activations:
                act_values = activations['net.0'].flatten().cpu().numpy()
            else:
                act_values = np.zeros_like(x_c)
        elif layer_idx==2:
            if 'net.2' in activations:
                act_values = activations['net.2'].flatten().cpu().numpy()
            else:
                act_values = np.zeros_like(x_c)
        else:
            act_values = np.zeros_like(x_c)

        fig.add_trace(go.Scatter3d(
            x=x_c, y=y_c, z=z_c,
            mode='markers',
            marker=dict(size=3, color=act_values, colorscale='Viridis', opacity=0.8),
            name=f"Layer {layer_idx+1}"
        ))
    fig.update_layout(
        title=f"🧠 3D MNIST Forward Pass (True Label: {y} | Predicted: {torch.argmax(output).item()})",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Layer Depth",
            bgcolor="black"
        ),
        paper_bgcolor="black",
        font=dict(color="white")
    )

    return fig