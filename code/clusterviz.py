import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles

if __name__ == '__main__':
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)

    X, y = make_gaussian_quantiles(n_features=2, n_classes=1)
    ax.scatter(X[:, 0], X[:, 1], marker='o', c='k', alpha=0.6)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_0$')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.title("Cluster Analysis")
    plt.savefig('/Users/benjamin/Desktop/cluster.png')

    kx = np.random.uniform(-3, 3, 5)
    ky = np.random.uniform(-3, 3, 5)
    plt.scatter(kx,ky, c='rbgyc', s=50)
    plt.savefig('/Users/benjamin/Desktop/kpoints.png')
    plt.show()
