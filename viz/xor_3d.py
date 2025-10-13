import torch
import numpy as np
import plotly.graph_objects as go
from utils.activations import ActivationRecorder

def xor_3d_viz(model, input_tensor):
    """
    Creates an interactive 3D visualization of activations
    across the XOR neural network layers for a given input.
    """
    model.eval()
    
    # Register activation recorder
    recorder = ActivationRecorder()
    recorder.register(model, ['net.0', 'net.2'])  # assuming net.0 and net.2 are linear layers

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    activations = recorder.get_activations()

    # Build neuron layer coordinates
    layer_sizes = [2, 8, 1]  # input, hidden, output
    z_positions = np.arange(len(layer_sizes))

    coords = []
    for i, size in enumerate(layer_sizes):
        x_coords = np.linspace(-1, 1, size)
        y_coords = np.random.uniform(-1, 1, size)
        z_coords = np.ones(size) * z_positions[i]
        coords.append((x_coords, y_coords, z_coords))

    # Build figure
    fig = go.Figure()

    # Map activations to each layer
    for i, (x_c, y_c, z_c) in enumerate(coords):
        if i == 1:  # hidden layer
            act_values = activations.get('net.0', torch.zeros(size)).flatten().numpy()
        elif i == 2:  # output layer
            act_values = activations.get('net.2', torch.zeros(size)).flatten().numpy()
        else:  # input layer
            act_values = input_tensor.flatten().numpy()

        fig.add_trace(go.Scatter3d(
            x=x_c, y=y_c, z=z_c,
            mode='markers',
            marker=dict(
                size=6,
                color=act_values,
                colorscale='Viridis',
                opacity=0.9
            ),
            name=f"Layer {i+1}"
        ))

        # Connection lines
        if i < len(coords) - 1:
            for xi, yi, zi in zip(x_c, y_c, z_c):
                for xj, yj, zj in zip(*coords[i + 1]):
                    fig.add_trace(go.Scatter3d(
                        x=[xi, xj], y=[yi, yj], z=[zi, zj],
                        mode='lines',
                        line=dict(color='white', width=1),
                        opacity=0.2,
                        showlegend=False
                    ))

    # Layout styling
    fig.update_layout(
        title=f"🧠 XOR 3D Forward Pass | Input: {input_tensor.tolist()} | Output: {torch.sigmoid(output).item():.4f}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Layer Depth",
            bgcolor="black"
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
    )

    return fig
