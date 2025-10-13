import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["animation.ffmpeg_path"] = r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import torch
from viz.network_viz import layout_layer,_normalize
from models.mlp import MLP
from utils.activations import ActivationRecorder
import os
def animate_fwd(model,recorder,X,layer_sizes,sample_idx=0,frames_per_stage=1,out="xor_fwd.mp4"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    layer_keys=[str(2*i) for i in range(len(layer_sizes)-1)]
    try:
        recorder.remove()
    except Exception:
        pass
    recorder.register(model.net,layer_keys)
    _=model(X)

    acts_by_layer=[]
    acts_by_layer.append(X[sample_idx].detach().cpu().numpy().flatten())
    for i in range(len(layer_sizes)-1):
        key=str(2*i)
        arr=recorder.data.get(key)
        if arr is None:
            acts_by_layer.append(np.zeros(layer_sizes[i+1]))
        else:
            acts_by_layer.append(arr[sample_idx].detach().cpu().numpy().flatten())


    normed=[_normalize(a) for a in acts_by_layer]

    coords=layout_layer(layer_sizes)
    fig=plt.figure(figsize=(9,7))
    ax=fig.add_subplot(111,projection="3d")
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    for l in range(len(coords)-1):
        for (x1,y1,z1) in coords[l]:
            for (x2,y2,z2) in coords[l+1]:
                ax.plot([x1,x2],[y1,y2],[z1,z2],color='white',alpha=0.12)

    scatter_artists=[]
    for li,layer in enumerate(coords):
        xs=[p[0] for p in layer]
        ys=[p[1] for p in layer]
        zs=[p[2] for p in layer]
        sc=ax.scatter(xs,ys,zs,s=50,c='black')
        scatter_artists.append(sc)

    ax.set_axis_off()
    ax.set_title("Forward pass" , color="white")

    n_stages=len(layer_sizes)
    total_frames=n_stages*frames_per_stage

    def update(frame):
        stage=frame//frames_per_stage
        t=(frame%frames_per_stage)/frames_per_stage

        for li,sc in enumerate(scatter_artists):
            acts=normed[li]
            sizes=[]
            colors=[]
            for ni,val in enumerate(acts):
                target=val*1.0
                if li<stage:
                    s=100+300*target
                    c=plt.get_cmap("plasma")(target)
                elif li==stage:
                    s=50+(100+300*target-50)*t
                    c=plt.get_cmap("plasma")(target*t)
                else:
                    s=25
                    c=(0.05,0.05,0.05,1.0)
                sizes.append(s)
                colors.append(c)

            sc._sizes=sizes
            sc.set_facecolor(colors)
        return scatter_artists
    ani=animation.FuncAnimation(fig,update,frames=total_frames,interval=150,blit=False)
    print("Saving to:", out)

    ani.save(out,writer="ffmpeg",fps=10)
    plt.close(fig)
    return out

if __name__ == "__main__":
    model = MLP(2, [8], 1)
    X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    recorder = ActivationRecorder()

    out_file = animate_fwd(
        model, recorder, X, [2,8,1], 3,
        out="C:/Users/Srishti/Singularity/assets/xor_fwd.mp4"
    )

    print("Animation saved to:", out_file)
    import os

out_file = "C:/Users/Srishti/Singularity/assets/xor_fwd.mp4"
print("Does file exist?", os.path.exists(out_file))
if os.path.exists(out_file):
    print("File size:", os.path.getsize(out_file), "bytes")
