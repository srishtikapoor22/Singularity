import torch
import plotly.graph_objects as go
import numpy as np
from utils.activations import ActivationRecorder

def xor_3d_viz(model: torch.nn.Module, input_tensor: torch.Tensor,animate=False):
    recorder = ActivationRecorder()
    recorder.register(model)

    model.eval()
    with torch.no_grad():
        raw_output = model.net[-2](torch.tanh(model.net[0](input_tensor)))
        prob = torch.sigmoid(raw_output).item()
        pred_value = int(prob > 0.5)

        


        print(f"Predicted XOR output: {pred_value}")

    activations = recorder.get_activations()
    if not activations:
        raise RuntimeError("No activations recorded — check recorder and model.")

    layer_names = list(activations.keys())
    layer_titles = ["Input Layer", "Hidden Layer", "Output Layer"]

    input_vals = input_tensor.cpu().numpy().flatten()
    activations = {"input": torch.tensor(input_vals).unsqueeze(0), **activations}

    coords = {}
    x_spacing = 3.0


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

        if idx == len(activations.keys()) - 1:
            n_neurons = 2
            vals = np.array([1 - vals[0], vals[0]])
            x = np.full(n_neurons, idx * x_spacing)
            y = np.linspace(-0.5, 0.5, n_neurons)
            z = np.zeros(n_neurons)

        coords[lname] = {"x": x, "y": y, "z": z, "vals": vals}


    fig = go.Figure()
    frames = []
    layer_traces = []
    init_opacity = 0.05 if animate else 1.0


    # Draw neurons
    for idx, lname in enumerate(coords.keys()):
        c = coords[lname]
        vals = c["vals"]
        vmin, vmax = float(vals.min()), float(vals.max())
        norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)

        if idx == 0:
            colors = [f"rgba(102,252,241,{0.4 + 0.6*a})" for a in norm]
            layer_label = "Input Layer"
        elif idx == len(coords) - 1:
            colors = [f"rgba(102,252,241,{0.6 + 0.4*a})" for a in norm]
            layer_label = "Output Layer"
        else:
            colors = [f"rgba(69,162,158,{0.3 + 0.7*a})" for a in norm]
            layer_label = "Hidden Layer"



        if lname == list(coords.keys())[-1]:
            sizes = [20, 20]
            colors = [
                "rgba(255,255,255,0.95)" if i == pred_value else "rgba(100,100,100,0.5)"
                for i in range(len(vals))
            ]
            layer_label = "Output Layer"
        else:
            sizes = 10 + 25 * norm

        init_opacity = 0.2 if animate else 1.0
        xs, ys, zs = c["x"], c["y"], c["z"]

        trace = go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=init_opacity, line=dict(width=0.5, color="white")),
            hoverinfo="text",
            text=[f"{layer_label}<br>Neuron {i}<br>Value: {vals[i]:.3f}" for i in range(len(vals))],
            showlegend=False
        )

        fig.add_trace(trace)
        layer_traces.append(trace)

        if idx == len(coords) - 1:
            for j, (x, y, z) in enumerate(zip(xs, ys, zs)):
                label_color = "#00FFFF" if j == pred_value else "#888888"
                fig.add_trace(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode="text",
                    text=[f"<b>{j}</b>"],
                    textfont=dict(color="#C5C6C7", size=14),
                    hoverinfo="none",
                    showlegend=False
                ))



    # Draw connections
    keys = list(coords.keys())
    for i in range(len(keys) - 1):
        l1, l2 = keys[i], keys[i + 1]
        a1, a2 = coords[l1], coords[l2]
        for j in range(len(a1["x"])):

            if l2 == list(coords.keys())[-1]:
                k_indices = [pred_value]
            else:
                k_indices = range(len(a2["x"]))

            for k in k_indices:
                layer_idx = i
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
                    strength = (abs(a1["vals"][j]) + abs(a2["vals"][k])) / 2

                if strength < 0.05:
                    continue

                fig.add_trace(go.Scatter3d(
                    x=[a1["x"][j], a2["x"][k]],
                    y=[a1["y"][j], a2["y"][k]],
                    z=[a1["z"][j], a2["z"][k]],
                    mode="lines",
                    line=dict(color=f"rgba(255,255,0,{0.2 + 0.8*strength})", width=0.8),
                    showlegend=False
                ))


    fig.add_annotation(text=f"Input: {input_vals.tolist()}",
                       x=0, y=1.2, xref="paper", yref="paper",
                       showarrow=False, font=dict(color="white", size=14))
    fig.add_annotation(text=f"Predicted Output: {pred_value}",
                       x=1, y=1.2, xref="paper", yref="paper",
                       showarrow=False, font=dict(color="yellow", size=16))

    layer_titles = ["Input Layer : ", "Hidden Layer : ", "Output Layer : "]
    layer_desc = [
        "Takes 2 input values (X1, X2)",
        "Applies learned weights + activation",
        "Outputs XOR prediction"
    ]

    for i, (layer_name, c) in enumerate(coords.items()):
        x_mean = np.mean(c["x"])
        y_mean = np.mean(c["y"])
        z_mean = np.mean(c["z"]) + 0.8
        fig.add_trace(go.Scatter3d(
            x=[x_mean], y=[y_mean], z=[z_mean],
            mode="text",
            text=[f"{layer_titles[i]}\n{layer_desc[i]}"],
            textfont=dict(color="#aab6ff", size=11),
            hoverinfo="none",
            showlegend=False
        ))

    frames = []
    num_neuron_traces = len(layer_traces)  # number of neuron-only traces (first traces in fig.data)

    for li in range(num_neuron_traces):
        frame_data = []
        for tj, trace in enumerate(fig.data):
            trace_dict = trace.to_plotly_json()
            # only modify the neuron traces (we assume first num_neuron_traces are neurons)
            if tj < num_neuron_traces and "marker" in trace_dict:
                # light up up to current layer
                if tj <= li:
                    trace_dict["marker"]["opacity"] = 1.0
                    # enlarge marker size for a "glow" pulse (handles scalar or per-point sizes)
                    sz = trace_dict["marker"].get("size", 6)
                    # if scalar -> scale, if list -> scale each
                    if isinstance(sz, (int, float)):
                        trace_dict["marker"]["size"] = sz * 1.5
                    else:
                        trace_dict["marker"]["size"] = [s * 1.5 for s in sz]
                else:
                    trace_dict["marker"]["opacity"] = 0.05
            # leave other traces (lines/text) unchanged
            frame_data.append(trace_dict)
        frames.append(go.Frame(data=frame_data, name=f"layer_{li}"))

    fig.frames = frames


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

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Forward Pass",
                        method="animate",
                        args=[[f"layer_{i}" for i in range(len(frames))],
                            {"frame": {"duration": 700, "redraw": True},
                            "transition": {"duration": 300, "easing": "linear"},
                            "fromcurrent": True, "mode": "immediate"}]),

                    dict(label="Pause",
                        method="animate",
                        args=[[None], {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}])
                ],
                x=0.15, y=0.98,
                xanchor="left", yanchor="top",
                bgcolor="rgba(0,0,0,0)",
                bordercolor="#66FCF1"
            )
        ]
    )

    btn_args = fig.layout.updatemenus[0].buttons[0].args[1]

    if "transition" not in btn_args:
        btn_args["transition"] = {}

    btn_args["frame"]["redraw"] = True
    btn_args["transition"]["easing"] = "linear"
    btn_args["transition"]["duration"] = 300

    fig.layout.sliders = [{
        "currentvalue": {"prefix": "Layer: "},
        "pad": {"t": 30},
    }]

    if "scene" in fig.layout and hasattr(fig.layout.scene, "camera"):
        camera = fig.layout.scene.camera
    else:
        camera = dict(eye=dict(x=1.8, y=1.8, z=1.8))


    for fr in fig.frames:
        fr.layout = go.Layout(scene_camera=camera)

    return fig
