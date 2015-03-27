import time
import random
import numpy as np
import matplotlib.pyplot as plt

from utils import load_energy
from sklearn import linear_model
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import mean_squared_error  # , r2_score


def visualize_error_ridge_alpha(n_alphas=200, n_folds=12):
    dataset = load_energy()
    alphas  = np.logspace(-10, -2, n_alphas)
    model   = linear_model.Ridge(fit_intercept=False)
    seed    = random.randint(1, 10000)
    X       = dataset.data
    y       = dataset.target('Y1')

    errors  = np.zeros(shape=(n_alphas, n_folds))
    for idx, alpha in enumerate(alphas):
        model.set_params(alpha=alpha)
        splits = ShuffleSplit(len(y), n_iter=n_folds, test_size=0.2,
                              random_state=seed)

        for jdx, (train, test) in enumerate(splits):
            X_train = X[train]
            y_train = y[train]
            X_test  = X[test]
            y_test  = y[test]

            model.fit(X_train, y_train)
            error = mean_squared_error(y_test, model.predict(X_test))

            errors[idx, jdx] = error

    print errors
    print errors.shape
    print alphas
    print alphas.shape

    plt.figure()
    plt.plot(alphas, errors, ':')

    plt.show()


def visualize_error_lasso_alpha():
    dataset = load_energy()

    start = time.time()
    model = linear_model.LassoCV(cv=20)
    model.fit(dataset.data, dataset.target('Y1'))
    delta = time.time() - start

    m_log_alphas = -np.log10(model.alphas_)

    plt.figure()
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                label='alpha: CV estimate')

    plt.legend()

    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: coordinate descent '
              '(train time: %.2fs)' % delta)
    plt.axis('tight')

    plt.show()

if __name__ == '__main__':
    visualize_error_ridge_alpha()
