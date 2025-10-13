import numpy as np

def layout_layers(layer_sizes,x_spacing=3.0,y_spacing=0.6):
    coords=[]
    #iterate through each layer
    for layer_idx,n in enumerate(layer_sizes):
        xs=layer_idx*x_spacing

        if n>1:
            ys=np.linspace(-(n-1)/2.0*y_spacing,(n-1)/2.0*y_spacing,n)

        else:
            ys=np.array([0.0])
        zs=np.zeros(n)
        layer_coords=[(x,float(y),float(z)) for y,z in zip(ys,zs)]
        coords.append(layer_coords)
    return coords

def normalize_activations(arr):
    a=_np.array(arr,dtype=float)
    if a.size==0:
        return a
    mn,mx=a.min(),a.max()
    if mx-mn<1e-8:
        return np.zeros_like(a)*0.5
    return (a-mn)/(mx-mn)