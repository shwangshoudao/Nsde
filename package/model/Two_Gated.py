import torch
import torch.nn as nn
import numpy as np


class two_gate(nn.Module):
    def __init__(self, dim, vNetWidth):
        """初始化

        Args:
            dim (int): 输入维度
            vNetWidth (int): 隐藏层维度
        """
        super(two_gate, self).__init__()
        self.dim = dim
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()        
        self.m_h = self.layer_1(dim, vNetWidth)
        self.t_h = self.layer_2(dim, vNetWidth)

    
    def layer_2(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn,nOut,bias=True), self.sigmoid)   
        return layer   
    def layer_1(self,  nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn,nOut,bias=True), self.softplus)   
        return layer
    
    def forward(self, S):
        """前向
        Args:
            S (n*2 array): input 第一列为到期时间 第二列为行权价格

        Returns:
            n*1 array: option price
        """
        a = self.m_h(S[:,0].view(-1,1))
        b = self.t_h(S[:,1].view(-1,1))
        output = (a*b).sum(axis=1)
        return output