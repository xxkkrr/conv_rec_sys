import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.gpu = config.use_gpu
        self.n = config.FM_n
        self.k = config.FM_k
        self.V = nn.Parameter(torch.randn(self.n, self.k),requires_grad=True)
        self.lin = nn.Linear(self.n, 1)

        if self.gpu:
            self.V = self.V.cuda()
            self.lin = self.lin.cuda()
        
    def forward(self, x):
        # x: M*n 
        # M: training objects, n: features
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        
        return out.squeeze(-1)