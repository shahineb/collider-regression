import torch
import torch.nn as nn


class KRR(nn.Module):
    """Exact Kernel Ridge Regression

    Args:
        kernel (gpytorch.kernels.Kernel)
        λ (float): regularisation weight
    """

    def __init__(self, kernel, λ):
        super().__init__()
        self.kernel = kernel
        self.λ = λ

    def fit(self, X, y):
        """Fits kernel ridge regression model
        ```
            α = (K(X,X) + nλI)^{-1} @ y
            f(x) = K(x,X) @ α
        ```

        Args:
            X (torch.tensor): (n,) or (n, d)
            y (torch.tensor): (n,)
        """
        n = len(y)
        K = self.kernel(X, X)
        Kλ = K.add_diag(self.λ * n * torch.ones(n)).evaluate()
        chol = torch.linalg.cholesky(Kλ)
        α = torch.cholesky_solve(y.view(-1, 1), chol).squeeze()
        self.register_buffer('X', X)
        self.register_buffer('α', α)

    def forward(self, x):
        output = self.kernel(x, self.X) @ self.α
        return output
