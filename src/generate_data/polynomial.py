import torch


def build_data_generator(d_X1, d_X2, d_Y, noise, **kwargs):
    # Initialise transition matrices
    X2_X1 = torch.ones((d_X1, d_X2)) / (d_X1 * d_X2)
    Y_X1 = torch.ones((d_X1, d_Y)) / (d_X1 * d_Y)
    X2_X1_poly = torch.ones((d_X1, d_X2)) / (d_X1 * d_X2)
    Y_X1_poly = torch.ones((d_X1, d_Y)) / (d_X1 * d_Y)
    X_1_mix = torch.ones((d_X1, d_X1)) / (d_X1 * d_X1)

    # Define utility that generates n samples
    def generate_data(n):
        Y = torch.randn(n, d_Y)
        X2 = torch.randn(n, d_X2)
        X1 = (Y_X1 @ Y.T + X2_X1 @ X2.T + X2_X1_poly @ (X2**2).T + Y_X1_poly @ (Y**2).T ).T
        X1.add_(noise * torch.randn(n, d_X1))
        X = torch.cat((X1, X2), dim=1)
        return X, Y

    # Define utility that generates samples for most gain evaluation
    def generate_most_gain_data(n, most_gain_sample):
        Y = torch.randn(n, d_X1, most_gain_sample)
        X2 = torch.randn(n, d_X2)
        X2 = X2.unsqueeze(2).repeat(1, 1, most_gain_sample)
        X1 = (Y_X1 @ Y.T + X2_X1 @ X2.T + X2_X1_poly @ (X2**2).T  + Y_X1_poly @ (Y**2).T).T
        X1 =  X1.add_(noise * torch.randn(n, d_X1,most_gain_sample))  # to finish
        X = torch.cat([X1, X2], dim=1)
        Y = torch.zeros(n, d_Y)
        return X, Y

    # Define utility that wraps up above
    def generate(n, seed=None, most_gain=False, most_gain_samples=100):
        if seed:
            torch.random.manual_seed(seed)
        if most_gain:
            X, Y = generate_most_gain_data(n, most_gain_samples)
        else:
            X, Y = generate_data(n)
        return X, Y

    # Return utility for usage
    return generate
