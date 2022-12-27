import torch
from math import pi


def build_data_generator(d_X1, d_X2, noise, d_X2_max=8, **kwargs):
    # Initialise covariance matrix
    torch.random.manual_seed(2000)
    M = torch.randn(4, d_X1 + d_X2_max + 1)
    M = M.div(M.norm(dim=0))
    idx = torch.cat([torch.arange(d_X1 + d_X2), torch.tensor([M.size(1) - 1])])
    M = M[:, idx]

    MY = M[:, -1:]
    M2 = M[:, d_X1:d_X1 + d_X2]
    proj = (M2.T @ MY).T * MY.repeat(1, d_X2)
    M[:, d_X1:d_X1 + d_X2] = M2 - proj

    Sigma = M.T @ M + 0.01 * torch.eye(d_X1 + d_X2 + 1)
    rootinvDiagSigma = torch.diag(1 / torch.sqrt(Sigma.diag()))
    Sigma = rootinvDiagSigma @ Sigma @ rootinvDiagSigma
    Sigma[Sigma.abs() < torch.finfo(torch.float32).eps] = 0

    # Initialise mvn distribution
    mvn = torch.distributions.MultivariateNormal(loc=torch.zeros(d_X1 + d_X2 + 1),
                                                 covariance_matrix=Sigma)

    # Define utility that generates n samples
    def generate_data(n):
        Z = mvn.sample((n,))
        X1, X2, Y = Z[:, :d_X1], Z[:, d_X1:d_X1 + d_X2], Z[:, -1:]
        X1 = X1 + 0.1 * torch.cos(2 * pi * X1**2)
        X2 = X2 + 0.1 * torch.sin(2 * pi * X2**2)
        X1 = X1.add(noise * torch.randn_like(X1))
        X = torch.cat([X1, X2], dim=1)
        return X, Y

    # Define utility that generates samples for most gain evaluation
    def generate_most_gain_data(n, most_gain_sample):
        raise NotImplementedError

    # Estimate means and standard deviation for standardisation
    torch.random.manual_seed(2000)
    X, Y = generate_data(n=100000)
    mu_X, sigma_X = X.mean(dim=0), X.std(dim=0)
    mu_Y, sigma_Y = Y.mean(), Y.std()

    def standardize(X, Y):
        X = (X - mu_X) / sigma_X
        Y = (Y - mu_Y) / sigma_Y
        return X, Y

    # Define utility that wraps up above
    def generate(n, seed=None, most_gain=False, most_gain_samples=0):
        if seed:
            torch.random.manual_seed(seed)
        if most_gain:
            raise NotImplementedError
        else:
            X, Y = generate_data(n)
            X, Y = standardize(X, Y)
        return X, Y

    # Return utility for usage
    return generate
