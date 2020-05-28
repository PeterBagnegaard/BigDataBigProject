# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Create_Graph import Create_Graph    

def Plot_Graph(data, charge_scale=1, t_scale=1/1000):
    # Define true values
    E_true, t_true = data.y[0][0], data.y[0][1]
    px, py, pz = float(data.y[0][2]), float(data.y[0][3]), float(data.y[0][4])
    vx, vy, vz = float(data.y[0][2]), float(data.y[0][3]), float(data.y[0][4])
    
    # Define measured values
    x, y, z, s, t = data.x[:, 0], data.x[:, 1], data.x[:, 2], charge_scale * data.x[:, 3] / data.x[:, 3].min(), data.x[:, 4]
    dt = (t[1:] - t[:-1]) * t_scale
    
    # Plot figure
    fig = plt.figure()
    plt.suptitle('True energy: %.2f' % E_true)
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(2, len(t)):
        ax.clear()
        Simulated_data = ax.scatter(x[:i], y[:i], z[:i], 'o', c='black', s=s[:i], label='Simulated data')
        ax.scatter(px, py, pz, '*', s=100, c='red', label='True position')
        ax.quiver3D(px, py, pz, vx, vy, vz, color='red', label='True direction')
        ax.legend()
        ax.plot(x[:i], y[:i], z[:i], ':k')
        ax.set_title("Time: %.0f of %.0f, true time: %.0f" % (t[i], t[-1], t_true))
        
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(z.min(), z.max())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()
        plt.pause(dt[i-1])

