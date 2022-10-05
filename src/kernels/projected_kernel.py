from gpytorch import kernels


class ProjectedKernel(kernels.Kernel):
    """Kernel of the projected RKHS

    Args:
        k (gpytorch.kernels.Kernel): base kernel acting on X
        l (gpytorch.kernels.Kernel): secondary kernel used to estimate CME
        X (torch.tensor): (n,) or (n, d) tensor used for CME estimation
        K (torch.tensor): precomputed k(X, X) tensor
        Lλ_rootinv (torch.tensor): precomputed (l(X,X) + λ)^{-1/2} tensor
    """

    def __init__(self, k, l, X, K, Lλ_inv):
        super().__init__()
        self.k = k
        self.l = l
        self.K = K
        self.Lλ_inv = Lλ_inv
        self.Lλ_inv_KW_Lλ_inv = self.Lλ_inv @ self.K @ self.Lλ_inv
        self.register_buffer('X', X)

    def forward(self, x1, x2, **kwargs):
        # Compute kernel matrices
        K_x1_to_x2 = self.k(x1, x2).evaluate()
        L_X_to_x1 = self.l(self.X, x1).evaluate()
        L_X_to_x2 = self.l(self.X, x2).evaluate()
        K_X_to_x1 = self.k(self.X, x1).evaluate()
        K_X_to_x2 = self.k(self.X, x2).evaluate()

        # Combine terms together
        output = K_x1_to_x2
        output -= L_X_to_x1.t() @ self.Lλ_inv @ K_X_to_x2
        output -= K_X_to_x1.t() @ self.Lλ_inv @ L_X_to_x2
        output += L_X_to_x1.t() @ self.Lλ_inv_KW_Lλ_inv @ L_X_to_x2
        return output


class ResidualKernel(kernels.Kernel):
    """Kernel of the residual RKHS

    Args:
        k (gpytorch.kernels.Kernel): base kernel acting on X
        l (gpytorch.kernels.Kernel): secondary kernel used to estimate CME
        X (torch.tensor): (n,) or (n, d) tensor used for CME estimation
        K (torch.tensor): precomputed k(X, X) tensor
        Lλ_rootinv (torch.tensor): precomputed (l(X,X) + λ)^{-1/2} tensor
    """

    def __init__(self, k, l, X, K, Lλ_inv):
        super().__init__()
        self.k = k
        self.l = l
        self.K = K
        self.Lλ_inv = Lλ_inv
        self.Lλ_inv_KW_Lλ_inv = self.Lλ_inv.t() @ self.K @ self.Lλ_inv
        self.register_buffer('X', X)

    def forward(self, x1, x2, **kwargs):
        # Compute kernel matrices
        L_X_to_x1 = self.l(self.X, x1).evaluate()
        L_X_to_x2 = self.l(self.X, x2).evaluate()

        # Combine terms together
        output = L_X_to_x1.t() @ self.Lλ_inv_KW_Lλ_inv @ L_X_to_x2
        return output
