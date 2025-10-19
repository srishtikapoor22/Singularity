# viz/mnist_3d.py
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from models.mlp import MLP
from data.mnist import get_mnist_loaders
from utils.activations import ActivationRecorder


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize(x):
    """Normalize numpy array to 0–1 safely."""
    x = np.array(x, dtype=float)
    if x.size == 0:
        return x
    mn, mx = x.min(), x.max()
    if abs(mx - mn) < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def layer_coords(layer_sizes, layer_gap=10, neuron_gap=1.5):
    """
    Generate clean, linear coordinates for neurons in each layer.
    Layers aligned along the X-axis, neurons evenly spaced vertically.
    """
    coords = []
    for i, size in enumerate(layer_sizes):
        x = i * layer_gap  # layer spacing along x-axis
        y_positions = torch.linspace(-size/2, size/2, steps=size) * neuron_gap
        layer = [(x, float(y), 0.0) for y in y_positions]
        coords.append(layer)
    return coords



def subsample_activations(acts, max_neurons=150):
    """Reduce neuron count for visualization clarity."""
    if acts.shape[1] > max_neurons:
        idx = np.linspace(0, acts.shape[1] - 1, max_neurons).astype(int)
        acts = acts[:, idx]
    return acts


# ------------------------------------------------------------
# Main 3D visualization
# ------------------------------------------------------------

def mnist_3d_viz(images=None, labels=None, model=None, sample_idx=0, max_neurons_per_layer=40):
    """
    Clean, explainable MNIST forward-pass visualization:
    Input -> Hidden1 -> Hidden2 -> Output (linear layout, adaptive spacing, glowing activations)
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import plotly.graph_objects as go
    from utils.activations import ActivationRecorder

    # --- Sanity checks
    if images is None or model is None:
        print("Error: Please provide model and images for visualization.")
        return

    X_sample = images[sample_idx:sample_idx + 1]
    y_true = labels[sample_idx].item()

    # --- Capture activations
    recorder = ActivationRecorder()
    recorder.register(model.net)
    model.eval()

    with torch.no_grad():
        output = model(X_sample)
        probs = F.softmax(output, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_label].item() * 100

    activations = recorder.get_activations()
    acts_by_layer = [X_sample.flatten().numpy()]
    for v in activations.values():
        acts_by_layer.append(v[0].detach().cpu().numpy())

    # --- Subsample layers for clarity
    def subsample(arr, max_n):
        if len(arr) > max_n:
            idx = np.linspace(0, len(arr) - 1, max_n, dtype=int)
            return arr[idx]
        return arr

    acts_by_layer = [subsample(a, max_neurons_per_layer) for a in acts_by_layer]

    # --- Build coordinates
    layer_sizes = [len(a) for a in acts_by_layer]
    x_spacing = 4
    y_spacing = 1.5
    coords = []
    for li, n in enumerate(layer_sizes):
        x = li * x_spacing
        ys = np.linspace(-(n - 1) / 2.0 * y_spacing, (n - 1) / 2.0 * y_spacing, n)
        zs = np.zeros(n) + (li % 2) * 0.8  # subtle depth offset
        coords.append([(x, float(y), float(z)) for y, z in zip(ys, zs)])

    # --- Normalizer
    def normalize(x):
        x = np.array(x, dtype=float)
        mn, mx = np.min(x), np.max(x)
        if abs(mx - mn) < 1e-8:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    # --- Initialize plot
    fig = go.Figure()

    # --- Draw only strongest connections
    for li in range(len(coords) - 1):
        src_act = normalize(acts_by_layer[li])
        dst_act = normalize(acts_by_layer[li + 1])
        src = coords[li]
        dst = coords[li + 1]

        # only connect top active neurons
        top_src_idx = np.argsort(src_act)[-5:]
        top_dst_idx = np.argsort(dst_act)[-5:]

        for i in top_src_idx:
            for j in top_dst_idx:
                strength = (src_act[i] + dst_act[j]) / 2
                if strength < 0.25:
                    continue
                fig.add_trace(go.Scatter3d(
                    x=[src[i][0], dst[j][0]],
                    y=[src[i][1], dst[j][1]],
                    z=[src[i][2], dst[j][2]],
                    mode="lines",
                    line=dict(color=f"rgba(255,255,100,{0.3 + 0.5*strength})", width=2),
                    showlegend=False
                ))

    # --- Plot neurons with glow based on activation
    for li, (acts, layer_coord) in enumerate(zip(acts_by_layer, coords)):
        norm_acts = normalize(acts)
        xs = [p[0] for p in layer_coord]
        ys = [p[1] for p in layer_coord]
        zs = [p[2] for p in layer_coord]

        if li == len(coords) - 1:
            # Output layer: highlight only predicted neuron
            colors = ["white" if i == pred_label else "gray" for i in range(len(ys))]
            sizes = [22 if i == pred_label else 10 for i in range(len(ys))]
        else:
            colors = [f"rgba(255,150,50,{0.3 + 0.7*a})" for a in norm_acts]
            sizes = 6 + 20 * norm_acts

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=sizes, color=colors, line=dict(color="white", width=0.5)),
            hoverinfo="text",
            text=[f"Layer {li} | Neuron {i}<br>Activation: {acts[i]:.3f}"
                  for i in range(len(acts))],
            showlegend=False
        ))

    # --- Add clear labels
    titles = ["Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output Layer"]
    for i, title in enumerate(titles[:len(coords)]):
        lx = coords[i][0][0]
        fig.add_trace(go.Scatter3d(
            x=[lx],
            y=[max([p[1] for p in coords[i]]) + 5],
            z=[0],
            mode="text",
            text=[title],
            textfont=dict(color="white", size=16),
            showlegend=False
        ))

    # --- Layout
    fig.update_layout(
        title=dict(
            text=f"MNIST Forward Pass → Predicted: {pred_label} (Conf {confidence:.1f}%)<br>"
                 f"<span style='font-size:14px;'>True Label: {y_true}</span>",
            font=dict(color="white", size=20)
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="black",
        ),
        paper_bgcolor="black",
        margin=dict(l=0, r=0, t=80, b=0)
    )

    fig.show()
    return fig
