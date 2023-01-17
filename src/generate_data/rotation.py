import torch
from scipy.spatial.transform import Rotation as R
import numpy as np


def build_data_generator(d_X1, d_X2, d_Y, noise, **kwargs):
    # Initialise transition matrices
    R_mat = (R.from_rotvec(np.pi / 3 * np.array([1, 1, 1]))).as_matrix()
    R_mat = torch.as_tensor(R_mat).float()

    # Define utility that generates n samples
    def generate_data(n):
        Y = torch.randn(n, d_Y)
        X2 = torch.randn(n, d_X2)
        XY = torch.cat([Y, X2], dim=1)
        X1 = (XY @ R_mat.T) + noise * torch.randn(n, d_X1)
        X = torch.cat([X1, X2], dim=1)
        return X, Y

    # Define utility that generates samples for most gain evaluation
    def generate_most_gain_data(n, most_gain_sample):
        Y = torch.randn(n, d_Y, most_gain_sample)
        X2 = torch.randn(n, d_X2)
        X2 = X2.unsqueeze(2).repeat(1, 1, most_gain_sample)
        XY = torch.cat([Y, X2], dim=1)
        X1 = (R_mat @ XY.permute(2, 1, 0)).permute(2, 1, 0) + noise * torch.randn(n, d_X1, most_gain_sample)
        X = torch.cat([X1, X2], dim=1)
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
