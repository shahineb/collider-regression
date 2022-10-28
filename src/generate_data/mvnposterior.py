import torch
from gpytorch import kernels
from scipy.spatial.transform import Rotation as R
import numpy as np


def build_data_generator(d_X1, d_X2, d_Y, noise):
    # Initialise rotation matrix
    R_mat = (R.from_rotvec(np.pi / 3 * np.array([1, 1, 1]))).as_matrix()
    R_mat = torch.as_tensor(R_mat).float()
    d_Z = d_X2 + d_Y

    # Initialise covariance matrices
    rbf = kernels.RBFKernel()
    rbf.lengthscale = 1.
    with torch.no_grad():
        torch.random.manual_seed(10)
        K = rbf(torch.randn(d_Z + d_X1).flip(dims=(0,))).evaluate()
    KX1X1 = K[:d_X1, :d_X1]
    KX1Z = K[:d_X1, d_X1:]
    KZZ = K[d_X1:, d_X1:]
    LZZ = torch.linalg.cholesky(KZZ)
    Sigma_post = KX1X1 - KX1Z @ torch.cholesky_solve(KX1Z.t(), LZZ)
    L_post = torch.linalg.cholesky(Sigma_post)

    # Define utility that generates n samples
    def generate_data(n):
        Y = 5 * torch.randn(n, d_Y)
        X2 = torch.randn(n, d_X2)
        XY = torch.cat([Y, X2], dim=1)
        Z = (XY @ R_mat.T)
        mu_post = KX1Z @ torch.cholesky_solve(Z.T, LZZ)
        X1 = mu_post + L_post @ torch.randn(d_X1, n)
        X1 = X1.T + noise * torch.randn(n, d_X1)
        X = torch.cat([X1, X2], dim=1)

        X = X / torch.tensor([0.8371, 7.6675, 2.3280, 0.9988, 0.9991])
        return X, Y

    # Define utility that generates samples for most gain evaluation
    def generate_most_gain_data(n, most_gain_sample):
        Y = 5 * torch.randn(most_gain_sample, d_Y, n)                         # (most_gain_sample, d_Y, n)
        X2 = torch.randn(d_X2, n)                                             # (d_X2, n)
        X2 = X2.unsqueeze(0).repeat(most_gain_sample, 1, 1)                   # (most_gain_sample, d_X2, n)
        XY = torch.cat([Y, X2], dim=1)                                        # (most_gain_sample, d_Z, n)
        Z = (R_mat @ XY)                                                      # (most_gain_sample, d_Z, n)
        mu_post = KX1Z @ torch.cholesky_solve(Z, LZZ)                         # (most_gain_sample, d_X1, n)
        X1 = mu_post + L_post @ torch.randn(most_gain_sample, d_X1, n)        # (most_gain_sample, d_X1, n)
        X1 = X1 + noise * torch.randn(most_gain_sample, d_X1, n)              # (most_gain_sample, d_X1, n)
        X = torch.cat([X1, X2], dim=1).permute(2, 1, 0)                       # (n, d_X1 + d_X2, most_gain_sample)

        X = X / torch.tensor([0.8371, 7.6675, 2.3280, 0.9988, 0.9991]).view(1, -1, 1)
        return X, None

    # Define utility that wraps up above
    def generate(n, seed=None, most_gain=False, most_gain_samples=0):
        if seed:
            torch.random.manual_seed(seed)
        if most_gain:
            X, Y = generate_most_gain_data(n, most_gain_samples)
        else:
            X, Y = generate_data(n)
        return X, Y

    # Return utility for usage
    return generate
