import numpy as np
import plotly.graph_objects as go
import torch
from typing import List

#calculate neuron position

def coords_for_layers(layer_sizes,x_spacing,y_spacing):
    coords=[]
    for layer_idx,n in enumerate(layer_sizes):
        x=layer_idx*x_spacing
        ys = np.linspace(- (n - 1) / 2.0 * y_spacing, (n - 1) / 2.0 * y_spacing, n)
        coords.append([(x, float(y), 0.0) for y in ys])
    return coords

def plotly_network_activations(layer_sizes,acts_by_layer,weights_matrices=None, title='Singularity'):
    coords=coords_for_layers(layer_sizes,x_spacing,y_spacing=1.5)
    fig=go.Figure(layout=dict(
        scene=dict=(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='black'
            ),
            paper_bgcolor='black',
            plot_bgcolor='black',
            title=dict(text=title, font=dict(color='white'))
    ))

    for li, coords in enumerate(coords):
        xc=[p[0] for p in coords]
        yc=[p[1] for p in coords]
        zc=[p[2] for p in coords]
        #get activation values
        acts=acts_by_layer[li]
        #normalize to 0-1
        norm = (acts - acts.min()) / ( (acts.max() - acts.min()) + 1e-8 )
        sizes=8+30*norm
        colors=norm
        sc=go.Scatter3d(
            x=xc, y=yc, z=zc,
            mode="markers",
            marker=dict(
                size=sizes.tolist(),
                color=colors.tolist,
                colorscale="Viridis",
                cmin=0,cmax=1,
                opacity=0.9,
                line=dict(color='white',width=0.5)
            ),
            hoverinfo='text',
            text=[f"Layer {li} | idx {i} | act {acts[i]:.3f}" for i in range(len(acts))],
                        )
        fig.add_trace(sc)

        #draw connections
        for layer_idx in range(len(coords)-1):
            source_layer=coords[layer_idx]
            destination_layer=coords[layer_idx+1]
            for source_idx,(x1,y1,z1) in enumerate(source_layer):
                for destination_idx,(x2,y2,z2) in enumerate(destination_layer):
                    fig.add_trace(go.Scatter3d(
                        x=[x1,x2], y=[y1,y2], z=[z1,z2],
                        mode="lines",
                        line=dict(color='rgba(150,150,150,0.15)',width=1),
                        hoverinfo='none',
                        showlegend=False
                    ))
    return fig