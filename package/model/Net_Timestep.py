import torch
import torch.nn as nn
import numpy as np

class Net_Timestep(nn.Module):
    def __init__(self, n_dim, n_out, n_layers, vNetWidth, activation = "relu"):
        super(Net_Timestep, self).__init__()
        self.n_dim = n_dim
        self.n_out = n_out
        
        if activation!="relu" and activation!="tanh":
            raise ValueError("unknown activation function {}".format(activation))
        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        self.i_h = self.hiddenLayerT0(n_dim, vNetWidth)
        self.h_h = nn.ModuleList([self.hiddenLayerT1(vNetWidth, vNetWidth) for l in range(n_layers-1)])
        self.h_o = self.outputLayer(vNetWidth, n_out)
        
    def hiddenLayerT0(self,  nIn, n_out):
        layer = nn.Sequential(#nn.BatchNorm1d(nIn, momentum=0.1),
                              nn.Linear(nIn,n_out,bias=True),
                              #nn.BatchNorm1d(n_out, momentum=0.1),   
                              self.activation)   
        return layer
    
    def hiddenLayerT1(self, nIn, n_out):
        layer = nn.Sequential(nn.Linear(nIn,n_out,bias=True),
                              #nn.BatchNorm1d(n_out, momentum=0.1),  
                              self.activation)   
        return layer
    def outputLayer(self, nIn, n_out):
        layer = nn.Sequential(nn.Linear(nIn, n_out,bias=True))
        return layer
    
    def forward(self, S):
        h = self.i_h(S)
        for l in range(len(self.h_h)):
            h = self.h_h[l](h)
        output = self.h_o(h)
        return output