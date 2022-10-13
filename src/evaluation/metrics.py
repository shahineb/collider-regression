from scipy.stats import pearsonr


def spearman_correlation(x, y):
    """Computes Spearman Correlation between x and y
    Args:
        x (torch.Tensor)
        y (torch.Tensor)
    Returns:
        type: torch.Tensor
    """
    x_std = (x - x.mean()) / x.std()
    y_std = (y - y.mean()) / y.std()
    corr = float(pearsonr(x_std.numpy(), y_std.numpy())[0])
    return corr
