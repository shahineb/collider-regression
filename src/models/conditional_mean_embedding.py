import torch
import torch.nn as nn

class CME_mat(nn.Module):
    """Implementation of CME which takes in the required matricies and
    kernel as input """

    def __init__(self, k, l, X, Y, Lλ_inv):
        super().__init__()
        self.k = k
        self.l = l
        self.Lλ_inv = Lλ_inv
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)

    def asses_fit(self,X,Y):
        term1 = self.l(Y,Y)
        term2 = -2* self.k(X,self.X) @ self.Lλ_inv @  self.l(self.Y,Y)
        term3 = self.k(X,self.X) @ self.Lλ_inv @ self.l(self.X, X)
        tot= 1/(Y.shape[0]) * torch.trace(term1 + term2 +term3) 
        return tot
