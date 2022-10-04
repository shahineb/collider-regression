import numpy as np
from sklearn.linear_model import LinearRegression
from math import floor


def make_transition_matrices(d_X1, d_X2, d_Y, seed):
    # Define transition matrices
    X2_X1 = np.ones((d_X1, d_X2)) / (d_X1 * d_X2)
    Y_X1 = np.ones((d_X1, d_Y)) / (d_X1 * d_Y)
    return X2_X1, Y_X1


def generate_data(n, d_X1, d_X2, d_Y, X2_X1, Y_X1, seed=None):
    # Generate samples
    if seed:
        np.random.seed(seed)
    Y = np.random.randn(n, d_Y)
    X2 = np.random.randn(n, d_X2)
    X1 = (Y_X1 @ Y.T + X2_X1 @ X2.T).T + np.random.randn(n, d_X1)
    X = np.concatenate((X1, X2), axis=1)
    return X, Y


def lin_most_gain(reg1, d_X1, d_X2, d_Y, X2_X1):
    n = 5000
    X2 = np.random.randn(n, d_X2)
    Y = np.zeros((n, d_Y))
    X1 = (X2_X1 @ X2.T).T
    X = np.concatenate((X1, X2), axis=1)
    Y_pred1 = reg1.predict(X)
    mse1 = np.mean((Y - Y_pred1)**2)
    return mse1


def mse(reg1, reg2, d_X1, d_X2, d_Y, X2_X1, Y_X1):
    n = 5000
    X, Y = generate_data(n, d_X1, d_X2, d_Y, X2_X1, Y_X1)
    Y_pred1 = reg1.predict(X)
    Y_pred2 = reg1.predict(X) - reg2.predict(X[:, d_X1:])
    mse1 = np.mean((Y - Y_pred1)**2)
    mse2 = np.mean((Y - Y_pred2)**2)
    return mse1, mse2


def run(n, d_X1, d_X2, d_Y, semi_prop, seed, recycle_samples=True):
    # Generate dataset
    m = floor(n * (1 + semi_prop))
    X2_X1, Y_X1 = make_transition_matrices(d_X1, d_X2, d_Y, seed)
    X, Y = generate_data(m, d_X1, d_X2, d_Y, X2_X1, Y_X1, seed)
    X_train, Y_train = X[:n], Y[:n]
    X_semi_train = X[:] if recycle_samples else X[n:]

    # Fit models
    reg1 = LinearRegression().fit(X_train, Y_train)
    reg2 = LinearRegression().fit(X_semi_train[:, d_X1:], reg1.predict(X_semi_train))

    # Compute MSEs
    mse1, mse2 = mse(reg1, reg2, d_X1, d_X2, d_Y, X2_X1, Y_X1)
    mse3 = lin_most_gain(reg1, d_X1, d_X2, d_Y, X2_X1)

    # Make output dict
    output = {'linreg': mse1.item(),
              'collider': mse2.item(),
              'difference': mse1.item() - mse2.item(),
              'most_gain': mse3.item()}
    return output
