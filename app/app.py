import streamlit as st
import plotly.graph_objects as go
import torch
import numpy as np
from Singularity.models.mlp import MLP
from Singulrity.utils.activations import ActivationRecorder

st.set_page_config(page_title="Singularity",layout="wide")
def load_model(path: str=models/xor.pt):
    model=MLP(input_dim=2,hidden_dims=[8],output_dim=1)
    state=torch.load(path,map_location=cpu)
    model.load_state_dict(state)
    model.eval()
    return model

def grid_layout(n:int, layer_idx: int, spacing: float=1.5):
    cols=int(np.ceil(np.sqrt(n)))
    rows=int(np.ceil(n/cols))
    xcoords,yccoords=[],[]
    for idx in range(n):
        r=idx//cols
        c=idx%cols
        xcoords.append(c-(cols-1)/2.0)
        ycoords.append(r-(rows-1)/2.0)
    zcoords=[layer_idx*spacing]*n
    return np.array(xcoords),np.array(ycoords),np.array(zcoords)

def make_connection_segments(x1,y1,z1,x2,y2,z2):
    xs,ys,zs=[],[],[]
    for i in range(len(x1)):
        xs+=[float(x1[i]),float(x2[i]),None]
        ys+=[float(y1[i]),float(y2[i]),None]
        zs+=[float(z1[i]),float(z2[i]),None]
    return xs,ys,zs

def main():
    st.title("Singularity")
    a=st.sidebar.selectbox("A",[0,1],index=0)
    b=st.sidebar.selectbox("B",[0,1],index=0)
    run=st.sidebar.button("Run")

    model=load_model()
    recorder=ActivationRecorder()
    recorder.register(model,["net.0","net.2"])

    if run:
        input=torch.tensor([[float(a),float(b)]], dtype=torch.float32)
        recorder.clear()
        with torch.no_grad():
            output=model(input)
        
        layers=[]
        layer_names=sorted(recorder.data.keys())
        for idx,name in enumerate(layer_names):
            t=recorder.data[name]
            vec=t.squeeze(0).numpy().reshape(-1)
            layers.append((name,vec))

        
        all_x,all_y,all_z,all_color,text=[],[],[],[],[]
        for li, (name, vec) in enumerate(layers):
            x_coords, y_coords, z_coords = grid_layout(len(vec), li, spacing=1.8)
            all_x.append(x_coords)
            all_y.append(y_coords)
            all_z.append(z_coords)
            all_color.append(vec)
            text.append([f"{name}:{i}" for i in range(len(vec))])

        fig=go.Figure()
        for i in range(len(layers)):
            fig.add_trace(go.Scatter3d(
                x=all_x[i], y=all_y[i], z=all_z[i],
                mode="markers",
                marker=dict(
                    size=8 + (np.abs(all_color[i]) * 12).tolist(),  # scale size by activation
                    color=all_color[i],
                    colorbar=dict(title="activation"),
                    showscale=(i==0)  # show colorbar once
                ),
                text=text[i],
                name=layers[i][0]
            ))
        for i in range(len(layers)-1):
            Xs, Ys, Zs = make_connection_segments(all_x[i], all_y[i], all_z[i],
                                                  all_x[i+1], all_y[i+1], all_z[i+1])
            fig.add_trace(go.Scatter3d(x=Xs, y=Ys, z=Zs, mode="lines",
                                       line=dict(width=2, color='lightblue'),
                                       hoverinfo='none', showlegend=False))

        fig.update_layout(scene=dict(bgcolor="black"),
                          paper_bgcolor="black",
                          plot_bgcolor="black",
                          scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))
        st.plotly_chart(fig, use_container_width=True)

        # prediction display
        prob = torch.sigmoid(logits).item()
        st.metric("Predicted probability (1)", f"{prob:.3f}")

if __name__ == "__main__":
    main()

