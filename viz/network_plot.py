import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_networks(layer_sizes):
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111,projection='3d')
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    coords=[]
    for layer_idx, num_neurons in enumerate(layer_sizes):
        layer_coords=[]
        for neurons_idx in range(num_neurons):
            x=0
            y=neurons_idx-num_neurons/2
            z=layer_idx*5

            ax.scatter(x,y,z,s=200,edgecolors='w',linewidths=1,alpha=0.9)
            layer_coords.append((x,y,z))
        coords.append(layer_coords)

    for l in range(len(coords)-1):
        for(x1,y1,z1) in coords[l]:
            for(x2,y2,z2) in coords[l+1]:
                ax.plot([x1,x1],[y1,y2],[z1,z2], c='gray',alpha=0.3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/3d_layout.png", dpi=200)
    plt.close()

if __name__=="__main__":
    draw_networks([2,8,1])