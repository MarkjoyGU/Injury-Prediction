import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_energy, DATA_DIR

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def interpolation_viz(degree_max=6,):
    """
    Shows interpolation of polynomial degree with Ridge.
    """
    def f(x):
        """ function to approximate by polynomial interpolation"""
        return x * np.sin(x)

    # generate points used to plot
    x_plot = np.linspace(0, 10, 100)

    # generate points and keep a subset of them
    x = np.linspace(0, 10, 100)
    rng = np.random.RandomState(0)
    rng.shuffle(x)
    x = np.sort(x[:20])
    y = f(x)

    # create matrix versions of these arrays
    X = x[:, np.newaxis]
    X_plot = x_plot[:, np.newaxis]

    plt.plot(x_plot, f(x_plot), label="ground truth")
    plt.scatter(x, y, label="training points")

    for degree in xrange(degree_max):
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X, y)
        y_plot = model.predict(X_plot)
        plt.plot(x_plot, y_plot, label="degree %d" % degree)

    plt.legend(loc='lower left')

    plt.show()


def nba_viz(degree=None):
    """
    Regression of NBA Data set with Ridge
    """
    df = pd.read_csv(os.path.join(DATA_DIR, 'nba_players.csv'))

    plt.scatter(df['PER'], df['SALARY'], label="tranining points",
                alpha=.6, color='#70B7BA')

    score = None
    if degree is not None:
        x = np.linspace(0, 35, 100)
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(df['PER'].reshape((len(df['PER']), 1)), df['SALARY'])
        y_plot = model.predict(x.reshape((len(x), 1)))
        plt.plot(x, y_plot, label="ridge degree %d" % degree,
                 color="#F1433F", linewidth=2, alpha=.7)

        score = model.score(df['PER'].reshape((len(df['PER']), 1)),
                            df['SALARY'])

    plt.ylim(0, df['SALARY'].max() + 100000)
    plt.xlim(0, df['PER'].max() + 5)
    plt.ylabel('salary')
    plt.xlabel('player efficiency rating')

    if score is not None:
        plt.title('NBA 2013 PER to Salary; Score: %0.3f' % score)
    else:
        plt.title('NBA 2013 PER to Salary Correlation')
    plt.legend(loc='lower right')

    # plt.show()
    plt.savefig("/Users/benjamin/Desktop/nba_regression_degree_%i.png" % degree)

if __name__ == '__main__':
    nba_viz(3)
