import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.activations import ActivationRecorder
from models.mlp import MLP
import numpy as np
import torch
import math
from typing import List, Tuple

def _normalize(arr: np.ndarray, eps:float=1e-8)->np.ndarray:
    amin=np.min(arr)
    amax=np.max(arr)
    return (arr-amin)/(amax-amin+eps)

def layout_layer(layer_sizes, x_spacing=5, y_spacing=2):
    coords=[]
    for layer_idx, n in enumerate(layer_sizes):
        if isinstance(n, torch.Tensor):
            n = n.detach().cpu().view(-1).tolist()
            if len(n)==1:
                n=n[0]
            else:
                raise ValueError("layer size tensor must be scalar")    
                  # extract scalar value from tensor
        else:
            n = int(n)

        xs = np.full(n, layer_idx * x_spacing)
        ys = np.linspace(-(n-1)/2.0*y_spacing, (n-1)/2.0*y_spacing, n)
        zs = np.zeros(n)
        coords.append(list(zip(xs, ys, zs)))
    return coords


def _get_linear_weight(model_net: torch.nn.Sequential, layer_idx: int):
    try:
        model=model_net[linear_idx]
        if isinstance(model, torch.nn.Linear):
            return model.weight.detach().cpu().numpy()
    except Exception:
        return None
    return None

def draw_network_with_activations(model: torch.nn.Module, recorder,X,layer_sizes: List[int],sample_idx: int=0,neuron_base_size: float=200.0,size_scale:float=600.0,cmap: str="plasma"):
    print("Debug:", layer_sizes)
    coords=layout_layer(layer_sizes)
    linear_names=[str(2*i) for i in range(len(layer_sizes)-1)]
    try:
        recorder.remove()
    except Exception:
        pass
    recorder.register(model.net, linear_names)
    _=model(X)
    act_by_layer=[]
    try:
        input_act=X[sample_idx].detach().cpu().numpy().flatten()
    except Exception:
        input_act=np.zeros(layer_sizes[0])
    act_by_layer.append(input_act)

    for i in range(len(layer_sizes)-1):
        layer_key = str(2 * i)
        if layer_key in recorder.data:
            arr = recorder.data[layer_key][sample_idx].detach().cpu().numpy().flatten()
            act_by_layer.append(arr)
        else:
            act_by_layer.append(np.zeros(layer_sizes[i+1]))
    flat=np.concatenate([a.flatten() for a in act_by_layer if a.size>0])
    norm_flat=_normalize(flat)
    splits=np.cumsum([a.size for a in act_by_layer[:-1]])
    per_layer_norm=[]
    for a in act_by_layer:
        if a.size==0:
            per_layer_norm.append(np.zeros_like(a))
        else:
            per_layer_norm.append(_normalize(a))


    fig=plt.figure(figsize=(9,7))
    ax=fig.add_subplot(111,projection='3d')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    all_vals = np.concatenate([a.flatten() for a in act_by_layer if a.size > 0]) if len(act_by_layer) > 0 else np.array([0.0])
    global_min, global_max = float(np.min(all_vals)), float(np.max(all_vals))

    for li,layer in enumerate(coords):
        acts = per_layer_norm[li] if li < len(per_layer_norm) else np.zeros(len(layer))
        for ni, (x, y, z) in enumerate(layer):
            val = float(acts[ni]) if ni < len(acts) else 0.0
            size = neuron_base_size + size_scale * val
            # color map: map val -> color
            cmap_map = plt.get_cmap(cmap)
            color_rgba = cmap_map(val)
            ax.scatter([x], [y], [z], s=size, c=[color_rgba], edgecolors="white", linewidths=0.6)

    line_alpha_base = 0.25
    for l in range(len(coords) - 1):
        weight = _get_linear_weight(model.net, 2 * l)
        src_coords = coords[l]
        dst_coords = coords[l + 1]
        for i_src, (x1, y1, z1) in enumerate(src_coords):
            for j_dst, (x2, y2, z2) in enumerate(dst_coords):
                w = 0.0
                if weight is not None:
                    
                    try:
                        w = float(weight[j_dst, i_src])
                    except Exception:
                        w = 0.0
                color = (0.2, 0.6, 1.0) if w >= 0 else (1.0, 0.2, 0.2)
                alpha = line_alpha_base * min(1.0, abs(w) / (np.abs(weight).max() + 1e-8)) if weight is not None else line_alpha_base * 0.4
                ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, alpha=alpha, linewidth=1.0)

    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(all_vals)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label("Normalized activation", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    
    ax.set_title("Network snapshot (activations) — sample idx: {}".format(sample_idx), color="white")

    plt.show()
    return fig, ax