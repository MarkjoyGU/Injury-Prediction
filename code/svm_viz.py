import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def circles():
    x  = np.random.uniform(0, 2*math.pi ,50)
    plt.scatter(np.cos(x), np.sin(x), color='r', alpha=0.7)
    plt.scatter(3*np.cos(x), 3*np.sin(x), alpha=0.7)

    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Circle within a Circle cannot be separated by a line')
    plt.show()


def hyperspace():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Add points

    x  = np.random.uniform(0, 2*math.pi ,50)
    ax.scatter(np.cos(x), np.sin(x), 3, color='r', alpha=0.7)
    ax.scatter(3*np.cos(x), 3*np.sin(x), 1, alpha=0.7, color='b')

    # Add plane
    # normal = np.array([1, 1, 2])
    # xx, yy = np.meshgrid(range(10), range(10))
    # d = np.array([0,0,0]).dot(normal)
    # z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    # ax.plot_surface(xx, yy, z, alpha=0.4, color='g')

    plt.show()

if __name__ == '__main__':
    hyperspace()
