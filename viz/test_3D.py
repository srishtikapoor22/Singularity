import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Example: 3 layers, neurons at different Z positions
layers = [2, 4, 1]
for i, n in enumerate(layers):
    xs = np.linspace(-1, 1, n)
    ys = np.zeros(n)
    zs = np.ones(n) * i
    ax.scatter(xs, ys, zs, s=200, c='cyan', edgecolors='k')

ax.set_title("Test 3D Neural Net Layout")
plt.show()
