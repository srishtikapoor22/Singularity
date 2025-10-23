# viz/xor_3d.py
import torch
import plotly.graph_objects as go
import numpy as np
from utils.activations import ActivationRecorder

def xor_3d_viz(model: torch.nn.Module, input_tensor: torch.Tensor):
    """
    Build a 3D interactive visualization for XOR MLP forward pass.
    Layers: Input (2), Hidden (n), Output (1)
    Shows activations, connections, and hover info for interpretability.
    """
    recorder = ActivationRecorder()
    recorder.register(model)

    model.eval()
    with torch.no_grad():
        raw_output = model.net[-2](torch.tanh(model.net[0](input_tensor)))  # manually compute last Linear
        prob = torch.sigmoid(raw_output).item()
        pred_value = int(prob > 0.5)

        


        print(f"Predicted XOR output: {pred_value}")

    # Fetch activations
    activations = recorder.get_activations()
    if not activations:
        raise RuntimeError("No activations recorded — check recorder and model.")

    layer_names = list(activations.keys())
    layer_titles = ["Input Layer", "Hidden Layer", "Output Layer"]

    # Include input explicitly
    input_vals = input_tensor.cpu().numpy().flatten()
    activations = {"input": torch.tensor(input_vals).unsqueeze(0), **activations}

    coords = {}
    x_spacing = 3.0

    # Prepare coordinates
    for idx, lname in enumerate(activations.keys()):
        tensor = activations[lname]
        if tensor.ndim == 2:
            n_neurons = tensor.shape[1]
            vals = tensor[0].cpu().numpy()
        else:
            n_neurons = tensor.numel()
            vals = tensor.flatten().cpu().numpy()

        x = np.full(n_neurons, idx * x_spacing)
        y = np.linspace(-1, 1, n_neurons)
        z = np.zeros(n_neurons)

        # If this is the final output layer, ensure it has 2 neurons for XOR (0 and 1)
        if idx == len(activations.keys()) - 1:
            n_neurons = 2
            vals = np.array([1 - vals[0], vals[0]])  # represent output_0 and output_1 probabilities
            x = np.full(n_neurons, idx * x_spacing)
            y = np.linspace(-0.5, 0.5, n_neurons)
            z = np.zeros(n_neurons)

        coords[lname] = {"x": x, "y": y, "z": z, "vals": vals}


    fig = go.Figure()

    # Draw neurons
    for idx, lname in enumerate(coords.keys()):
        c = coords[lname]
        vals = c["vals"]
        vmin, vmax = float(vals.min()), float(vals.max())
        norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)

        # Layer-specific colors
        if idx == 0:
            colors = [f"rgba(50,150,255,{0.4 + 0.6*a})" for a in norm]
            layer_label = "Input Layer"
        elif idx == len(coords) - 1:
            colors = [f"rgba(255,255,150,{0.5 + 0.5*a})" for a in norm]
            layer_label = "Output Layer"
        else:
            colors = [f"rgba(255,180,50,{0.3 + 0.7*a})" for a in norm]
            layer_label = "Hidden Layer"

        # Neuron sizes
        # Detect if this is the final output layer by name instead of idx
        if lname == list(coords.keys())[-1]:
            # both output neurons same base size
            sizes = [20, 20]
            # highlight the predicted one (white), dim the other (gray)
            colors = [
                "rgba(255,255,255,0.95)" if i == pred_value else "rgba(100,100,100,0.5)"
                for i in range(len(vals))
            ]
            layer_label = "Output Layer"
        else:
            sizes = 10 + 25 * norm


        fig.add_trace(go.Scatter3d(
            x=c["x"],
            y=c["y"],
            z=c["z"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(color="white", width=0.8)
            ),
            hoverinfo="text",
            text=[f"{layer_label}<br>Neuron {i}<br>Activation: {vals[i]:.3f}" for i in range(len(vals))],
            showlegend=False
        ))

    # Draw connections
    keys = list(coords.keys())
    for i in range(len(keys) - 1):
        l1, l2 = keys[i], keys[i + 1]
        a1, a2 = coords[l1], coords[l2]
        for j in range(len(a1["x"])):
            # Determine which neurons to connect to
            if l2 == list(coords.keys())[-1]:
                # Output layer → only connect to the predicted neuron
                k_indices = [pred_value]
            else:
                # Otherwise, connect to all neurons normally
                k_indices = range(len(a2["x"]))

            for k in k_indices:
                layer_idx = i

                # Skip invalid layer index (since we added a pseudo 2-neuron output layer for visualization)
                if layer_idx >= len(model.net):
                    continue

                layer = model.net[layer_idx]
                if hasattr(layer, "weight"):
                    weights = layer.weight.detach().cpu().numpy()
                else:
                    weights = None

                if weights is not None and k < weights.shape[0] and j < weights.shape[1]:
                    w = abs(weights[k, j])
                    strength = w / (np.max(weights) + 1e-8)
                else:
                    # fallback: use activation magnitude if no weights found
                    strength = (abs(a1["vals"][j]) + abs(a2["vals"][k])) / 2

                if strength < 0.05:
                    continue

                fig.add_trace(go.Scatter3d(
                    x=[a1["x"][j], a2["x"][k]],
                    y=[a1["y"][j], a2["y"][k]],
                    z=[a1["z"][j], a2["z"][k]],
                    mode="lines",
                    line=dict(color=f"rgba(255,255,0,{0.2 + 0.8*strength})", width=1 + 3*strength),
                    showlegend=False
                ))

    # --- Add explicit labels for input and output neurons ---
    # Input labels
    input_coords = list(coords.values())[0]
    for i, val in enumerate(input_coords["vals"]):
        fig.add_trace(go.Scatter3d(
        x=[input_coords["x"][i]],
        y=[input_coords["y"][i] + 0.3],
        z=[input_coords["z"][i]],
        mode="text",
        text=[f"x{i+1} = {val:.0f}"],
        textfont=dict(color="deepskyblue", size=12),
        showlegend=False
    ))


    # Output labels
    output_coords = list(coords.values())[-1]
    for i, val in enumerate(output_coords["vals"]):
        label = f"Output {i} ({i})"
        color = "yellow" if i == int(pred_value) else "gray"
        fig.add_trace(go.Scatter3d(
        x=[output_coords["x"][i] + 0.4],
        y=[output_coords["y"][i]],
        z=[output_coords["z"][i]],
        mode="text",
        text=[label],
        textfont=dict(color=color, size=12),
        showlegend=False
    ))



    # Annotations for clarity
    fig.add_annotation(text=f"Input: {input_vals.tolist()}",
                       x=0, y=1.2, xref="paper", yref="paper",
                       showarrow=False, font=dict(color="white", size=14))
    fig.add_annotation(text=f"Predicted Output: {pred_value}",
                       x=1, y=1.2, xref="paper", yref="paper",
                       showarrow=False, font=dict(color="yellow", size=16))

    # Final layout
    fig.update_layout(
        title="XOR Neural Network Visualization",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig
