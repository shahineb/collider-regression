import torch
from gpytorch import kernels


class ConstantKernel(kernels.Kernel):
    """Constant kernel

    Args:
        variance (torch.tensor): (1,)
    """

    def __init__(self, variance=1.):
        super().__init__()
        raw_variance = -torch.log(torch.exp(torch.tensor(variance)) - 1)
        self.raw_variance = torch.nn.Parameter(raw_variance)

    def forward(self, x1, x2, **kwargs):
        return self.variance * torch.ones(len(x1), len(x2))

    @property
    def variance(self):
        return torch.log(1 + torch.exp(-self.raw_variance))
