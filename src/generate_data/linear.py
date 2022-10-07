import torch


def build_data_generator(d_X1, d_X2, d_Y):
    # Initialise transition matrices
    X2_X1 = torch.ones((d_X1, d_X2)) / (d_X1 * d_X2)
    Y_X1 = torch.ones((d_X1, d_Y)) / (d_X1 * d_Y)

    # Define utility that generates n samples
    def generate_data(n):
        Y = torch.randn(n, d_Y)
        X2 = torch.randn(n, d_X2)
        X1 = (Y_X1 @ Y.T + X2_X1 @ X2.T).T + torch.randn(n, d_X1)
        X = torch.cat([X1, X2], dim=1)
        return X, Y

    # Define utility that generates samples for most gain evaluation
    def generate_most_gain_data(n):
        Y = torch.zeros(n, d_Y)
        X2 = torch.randn(n, d_X2)
        X1 = (X2_X1 @ X2.T).T
        X = torch.cat([X1, X2], dim=1)
        return X, Y

    # Define utility that wraps up above
    def generate(n, seed=None, most_gain=False,most_gain_sample=0):
        if seed:
            torch.random.manual_seed(seed)
        if most_gain:
            X, Y = generate_most_gain_data(n)
        else:
            X, Y = generate_data(n)
        return X, Y

    # Return utility for usage
    return generate
