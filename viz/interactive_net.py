import numpy as np
import torch
import plotly.graph_objects as go
from viz.network_viz import layout_layer,_normalize

#build network
def build_network_fig(layer_sizes,activations):
    coords=layout_layer(layer_sizes)
    fig=go.Figure()

    #drawing connection lines
    for layer_index in range(len(coords)-1):
        for (x1,y1,z1) in coords[layer_index]:
            for (x2,y2,z2) in coords[layer_index+1]:
                fig.add_trace(go.Scatter3d(
                    x=[x1,x2],
                    y=[y1,y2],
                    z=[z1,z2],
                    mode='lines',
                    line=dict(color='rgba(255,255,255,0.15)', width=1),
                    hoverinfo='none',
                    showlegend=False
                ))
    #drawing neurons
    for li, layer_coords in enumerate(coords):
        xc=[p[0] for p in layer_coords]
        yc=[p[1] for p in layer_coords]
        zc=[p[2] for p in layer_coords]

        if activations is not None and li<len(activations):
            acts=_normalize(activations[li])
            colors=[f'rgba({int(a*255)}, {int(50)}, {int(255 - a*255)}, 1)' for a in acts]
            sizes=[10 + a*15 for a in acts]
        else:
            colors = ['rgba(100,100,255,0.6)'] * len(xs)
            sizes = [10] * len(xs)
        
        fig.add_trace(go.Scatter3d(
            x=xs,y=ys,z=zs,
            mode='markers',
            marker=dict(size=sizes, color=colors, opacity=0.9, line=dict(width=1, color='white')),
            name=f"Layer {li+1}"
        ))

    fig.update_layout(scene=dict(
        xaxis=dict(showbackground=False,visible=False),
        yaxis=dict(showbackground=False,visible=False),
        zaxis=dict(showbackground=False,visible=False),
    ),
    margin=dict(l=0,r=0,t=0,b=0),
    paper_bgcolor='black',
    plot_bgcolor='black',
    )
    return fig

#running a fwd pass to capture activations
def forward_pass_act(model,X_sample,layer_sizes):
    acts_by_layer=[]
    current=X_sample
    acts_by_layer.append(current.detach().cpu().numpy().flatten())

    idx=0
    for layer in model.net:
        current=layer(current)
        if hasattr(layer, 'weight'):
            acts_by_layer.append(current.detach().cpu().numpy().flatten())
    return acts_by_layer