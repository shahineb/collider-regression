from gpytorch import kernels


class ProjectedKernel(kernels.Kernel):
    """Kernel of the projected RKHS

        TODO : redo docs

    Args:
        k (gpytorch.kernels.Kernel): base kernel acting on X
        l (gpytorch.kernels.Kernel): secondary kernel used to estimate CME
        X (torch.tensor): (n,) or (n, d) tensor used for CME estimation
        K (torch.tensor): precomputed k(X, X) tensor
        Lλ_rootinv (torch.tensor): precomputed (l(X,X) + λ)^{-1/2} tensor
    """

    def __init__(self, r_plus, l, X, Rplus, Lλ_inv):
        super().__init__()
        self.r_plus = r_plus
        self.l = l
        self.Rplus = Rplus
        self.Lλ_inv = Lλ_inv
        self.Lλ_inv_R_Lλ_inv = self.Lλ_inv @ self.Rplus @ self.Lλ_inv
        self.register_buffer('X', X)

    def forward(self, x1, x2, **kwargs):
        # Compute kernel matrices
        R_x1_to_x2 = self.r_plus(x1, x2).evaluate()
        L_X_to_x1 = self.l(self.X, x1).evaluate()
        L_X_to_x2 = self.l(self.X, x2).evaluate()
        L_x1_to_x2 = self.l(x1, x2).evaluate()
        R_X_to_x1 = self.r_plus(self.X, x1).evaluate()
        R_X_to_x2 = self.r_plus(self.X, x2).evaluate()

        # Combine terms together
        output = R_x1_to_x2
        output -= L_X_to_x1.t() @ self.Lλ_inv @ R_X_to_x2
        output -= R_X_to_x1.t() @ self.Lλ_inv @ L_X_to_x2
        output += L_X_to_x1.t() @ self.Lλ_inv_R_Lλ_inv @ L_X_to_x2
        output = output.mul(L_x1_to_x2)
        return output
