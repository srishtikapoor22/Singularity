import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from models.mlp import MLP
from data.mnist import get_mnist_loaders
from utils.activations import ActivationRecorder
import torch.nn as nn


def normalize(x):
    x = np.array(x, dtype=float)
    if x.size == 0:
        return x
    mn, mx = x.min(), x.max()
    if abs(mx - mn) < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def layer_coords(layer_sizes, layer_gap=10, neuron_gap=1.5):
    coords = []
    for i, size in enumerate(layer_sizes):
        x = i * layer_gap
        y_positions = torch.linspace(-size/2, size/2, steps=size) * neuron_gap
        layer = [(x, float(y), 0.0) for y in y_positions]
        coords.append(layer)
    return coords



def subsample_activations(acts, max_neurons=150):
    if acts.shape[1] > max_neurons:
        idx = np.linspace(0, acts.shape[1] - 1, max_neurons).astype(int)
        acts = acts[:, idx]
    return acts



def mnist_3d_viz(images=None, labels=None, model=None, sample_idx=0, max_neurons_per_layer=40,animate=False):
    import torch
    import torch.nn.functional as F
    import numpy as np
    import plotly.graph_objects as go
    from utils.activations import ActivationRecorder

    if isinstance(labels, (int, float)):
        y_true = int(labels)
        sample_idx = 0
    else:
        sample_idx = 0
        y_true = labels[sample_idx].item()


    if images is None or model is None:
        print("Error: Please provide model and images for visualization.")
        return

    X_sample = images[sample_idx:sample_idx + 1]
    y_true = labels[sample_idx].item()


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

    def subsample(arr, max_n):
        if len(arr) > max_n:
            idx = np.linspace(0, len(arr) - 1, max_n, dtype=int)
            return arr[idx]
        return arr

    acts_by_layer = [subsample(a, max_neurons_per_layer) for a in acts_by_layer]

    weights = []
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.detach().cpu().numpy())

    layer_sizes = [len(a) for a in acts_by_layer]
    x_spacing = 5
    y_spacing = 1.5
    coords = []
    for li, n in enumerate(layer_sizes):
        x = li * x_spacing
        ys = np.linspace(-(n - 1) / 2.0 * y_spacing, (n - 1) / 2.0 * y_spacing, n)
        zs = np.zeros(n)
        coords.append([(x, float(y), float(z)) for y, z in zip(ys, zs)])


    def normalize(x):
        x = np.array(x, dtype=float)
        mn, mx = np.min(x), np.max(x)
        if abs(mx - mn) < 1e-8:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    fig = go.Figure()
    


    CONNECTION_THRESHOLD = 0.18
    for li in range(len(coords) - 1):
        src_act = normalize(acts_by_layer[li])
        dst_act = normalize(acts_by_layer[li + 1])
        src = coords[li]
        dst = coords[li + 1]

        top_src_idx = np.argsort(src_act)[-5:]
        top_dst_idx = np.argsort(dst_act)[-5:]

        for i in top_src_idx:
            for j in top_dst_idx:
                if weights is not None and li < len(weights):
                    W = weights[li]
                    if j < W.shape[0] and i < W.shape[1]:
                        w_val = float(W[j, i])
                    else:
                        w_val = 0.0
                    strength = abs(w_val) * (0.5 + 0.5 * (src_act[i] + dst_act[j]) / 2.0)
                else:
                    strength = (src_act[i] + dst_act[j]) / 2.0


                line_width = 0.8 + 3.5 * min(1.0, strength)
                alpha = 0.15 + 0.75 * min(1.0, strength)
                line_color = f"rgba({int(60 + 195*strength)}, {int(100 + 100*(1-strength))}, {int(200 + 55*strength)}, {alpha})"

                fig.add_trace(go.Scatter3d(
                    x=[src[i][0], dst[j][0]],
                    y=[src[i][1], dst[j][1]],
                    z=[src[i][2], dst[j][2]],
                    mode="lines",
                    line=dict(color=line_color, width=line_width),
                    hoverinfo='none',
                    showlegend=False
                ))


    layer_traces = []
    for li, (acts, layer_coord) in enumerate(zip(acts_by_layer, coords)):
        norm_acts = normalize(acts)
        xs = [p[0] for p in layer_coord]
        ys = [p[1] for p in layer_coord]
        zs = [p[2] for p in layer_coord]

        if li == len(coords) - 1:
            n_out = max(len(ys), 10)
            colors = [
                "rgba(102,252,241,1.0)" if i == pred_label else "rgba(100,100,150,0.22)"
                for i in range(n_out)
            ]
            sizes = [28 if i == pred_label else 6 for i in range(n_out)]

            if len(ys) < n_out:
                colors = colors[:len(ys)]
                sizes = sizes[:len(ys)]

        else:
            sizes = (4 + 16 * (norm_acts ** 1.2)).tolist()
            colors = [
                f"rgba({50 + int(180*a)}, {100 + int(80*(1-a))}, {200 + int(55*a)}, {0.38 + 0.62*a})"
                for a in norm_acts
            ]

        def highlight_layer_colors(norm_acts, layer_index, colors, sizes):
            if layer_index == len(coords) - 1:
                new_colors = []
                new_sizes = []
                for i, c in enumerate(colors):
                    if i == pred_label:
                        new_colors.append("rgba(255,230,120,0.98)")
                        new_sizes.append(max(sizes[i], 20))
                    else:
                        new_colors.append("rgba(80,80,100,0.25)")
                        new_sizes.append(max(6, sizes[i]*0.35))
                return new_colors, new_sizes
            importance = norm_acts
            new_colors = []
            new_sizes = []
            for i, a in enumerate(importance):
                if a > 0.6:
                    new_colors.append(colors[i])
                    new_sizes.append(sizes[i])
                else:
                    new_colors.append(f"rgba(30,30,40,0.18)")
                    new_sizes.append(max(6, sizes[i]*0.45))
            return new_colors, new_sizes

        colors, sizes = highlight_layer_colors(norm_acts, li, colors, sizes)
        init_opacity = 0.15 if animate else 0.35


        trace = go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(color="white", width=0.5),
                opacity=init_opacity
            ),
            hoverinfo="text",
            text=[f"Layer {li} | Neuron {i}<br>Activation: {acts[i]:.3f}" for i in range(len(acts))],
            showlegend=False
        )

        fig.add_trace(trace)


        if 'layer_traces' not in locals():
            layer_traces = []
        layer_traces.append(len(fig.data) - 1)

    layer_titles = [
    f"Input Layer (28×28)",
    f"Hidden Layer 1 — Edge detectors :",
    f"Hidden Layer 2 — Curve patterns :",
    f"Output Layer — Digit classes :"
    ]
    layer_desc = [
        "Raw pixel intensities",
        "Detects strokes & edges",
        "Combines edges into shapes",
        "Activates for digit class"
    ]
    for i, title in enumerate(layer_titles[:len(coords)]):
        lx = coords[i][0][0]
        max_y = max(p[1] for p in coords[i])
        
        fig.add_trace(go.Scatter3d(
            x=[lx],
            y=[max_y + 10],
            z=[0],
            mode="text",
            text=[f"{title}\t{layer_desc[i]}"],
            textfont=dict(color="rgba(200,200,255,0.95)", size=17, family="Orbitron, monospace"),
            showlegend=False
            ))
        
    if len(coords) > 0:
        x_in = coords[0][0][0]
        fig.add_trace(go.Scatter3d(
            x=[x_in] * len(acts_by_layer[0]),
            y=[p[1] for p in coords[0]],
            z=[0] * len(acts_by_layer[0]),
            mode="text",
            text=None,
            textfont=dict(color="#00FFFF", size=10),
            showlegend=False
        ))

    if len(coords) > 0:
        x_out = coords[-1][0][0]
        fig.add_trace(go.Scatter3d(
            x=[x_out] * 10,
            y=[p[1] for p in coords[-1][:10]],
            z=[0] * 10,
            mode="text",
            text=[str(i) for i in range(10)],
            textfont=dict(color="#66FCF1", size=14),
            showlegend=False
        ))

    frames = []
    if animate and len(layer_traces) > 0:
        base_traces = list(fig.data)
        total_layers = len(layer_traces)
        for step in range(total_layers):
            frame_traces = []
            for idx, base in enumerate(base_traces):
                if idx in layer_traces:
                    layer_idx = layer_traces.index(idx)
                    opacity_val = 1.0 if layer_idx <= step else 0.08
                    mk = base.marker.to_plotly_json() if hasattr(base, "marker") and base.marker is not None else {}
                    mk["opacity"] = opacity_val
                    frame_traces.append(go.Scatter3d(
                        x=base.x, y=base.y, z=base.z,
                        mode=base.mode or "markers",
                        marker=mk,
                        text=base.text if hasattr(base, "text") else None,
                        hoverinfo=base.hoverinfo if hasattr(base, "hoverinfo") else None,
                        showlegend=False
                    ))
                else:
                    mk = base.marker.to_plotly_json() if hasattr(base, "marker") and base.marker is not None else {}
                    frame_traces.append(go.Scatter3d(
                        x=base.x, y=base.y, z=base.z,
                        mode=base.mode or "markers",
                        marker=mk,
                        text=base.text if hasattr(base, "text") else None,
                        hoverinfo=base.hoverinfo if hasattr(base, "hoverinfo") else None,
                        showlegend=False
                    ))
            frames.append(go.Frame(data=frame_traces, name=f"frame_{step}"))
        fig.frames = frames
    else:
        fig.frames = []





    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {
                    "frame": {"duration": 700, "redraw": True},
                    "fromcurrent": True,
                    "mode": "immediate"
                }],



                    "label": "Forward Pass",
                    "method": "animate"
                },
                {
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate"
                    }],
                    "label": "Stop",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 60},
            "showactive": True,
            "type": "buttons",
            "x": 0.45,
            "xanchor": "right",
            "y": 1.15,
            "yanchor": "top",
            "bgcolor": "rgba(20,20,40,0.5)",
            "bordercolor": "#444"
        }]
    )
    fig.update_layout(
    title=dict(
        text=f"<b>MNIST Forward Pass → Predicted: {pred_label}"
             f"<span style='font-size:14px; color:#bbb;'>True Label: {y_true}</span>",
        font=dict(color="#C7D7FF", size=22, family="Orbitron, monospace"),
        x=0.5,
    ),
    scene=dict(
        xaxis=dict(visible=False, showgrid=False),
        yaxis=dict(visible=False, showgrid=False),
        zaxis=dict(visible=False, showgrid=False),
        bgcolor="#020611"
    ),
    paper_bgcolor="#000000",
    margin=dict(l=0, r=0, t=90, b=0)
    )

    
    fig.show()
    return fig
