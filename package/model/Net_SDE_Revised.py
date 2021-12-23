import torch
import torch.nn as nn
import numpy as np
from model import Net_Timestep

class Net_SDE_Revised(nn.Module):
    
    def __init__(self, asset_info,n_dim,n_layers, vNetWidth, MC_samples,device):
        """SDE_PRO的模型

        Args:
            asset_info (list): asset info  0. S0 1. V0 2. rate 无风险利率
            n_dim (int): 输入维度
            n_layers (int): nn网络有几层
            vNetWidth (int): nn网络宽度
            MC_samples (int): 仿真样本数
            device (str): 决定是cpu还是gpu
        """
        
        super(Net_SDE_Revised, self).__init__()
        self.dim = n_dim
        self.device = device
        
        self.S0 = asset_info[0]*torch.ones(1,1)
        self.V0 = asset_info[1]*torch.ones(1,1)
        self.rate = asset_info[2]*torch.ones(1,1)
        self.MC_samples = MC_samples
        
        #Input to each coefficient (NN) will be (t,S_t,V_t)
        self.driftS = Net_Timestep(n_dim=n_dim, n_out=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusion =Net_Timestep(n_dim=n_dim, n_out=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.driftV = Net_Timestep(n_dim=n_dim, n_out=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusionV = Net_Timestep(n_dim=n_dim, n_out=1, n_layers=n_layers, vNetWidth=vNetWidth)
    
        
        
    def forward(self,x): 
        S_old = torch.repeat_interleave(self.S0, 2*self.MC_samples, dim=0).to(device=self.device)
        V_old = torch.repeat_interleave(self.V0, 2*self.MC_samples, dim=0).to(device=self.device)  
        Sample_path = torch.Tensor().to(device=self.device)
        Sample_path = torch.cat([Sample_path,S_old],1)
        rate = torch.repeat_interleave(self.rate, 2*self.MC_samples, dim=0).to(device=self.device) 
        zeros = torch.repeat_interleave(torch.zeros(1,1), 2*self.MC_samples, dim=0).to(device=self.device) 
        # use fixed step size
        timegrid = torch.linspace(0,int(x[:,0].max())/360,int(x[:,0].max()+1))
        h = (timegrid[1]-timegrid[0]).to(device=self.device)
        n_steps = len(timegrid)-1
        
        z = torch.randn(self.MC_samples,n_steps)
        z1 = torch.randn(self.MC_samples,n_steps)
        z1 = -0.5*z + np.sqrt(0.75)*z1
        z = torch.cat([z,-z],axis=0)
        z1 = torch.cat([z1,-z1],axis=0)



        # Solve for S_t, V_t (Euler)
        for i in range(1, len(timegrid)):
            dW = (torch.sqrt(h) * z[:,i-1]).reshape(2*self.MC_samples,1).to(device=self.device)
            dW1 = (torch.sqrt(h) * z1[:,i-1]).reshape(2*self.MC_samples,1).to(device=self.device)
            current_time = (torch.ones(1, 1)*timegrid[i-1]).to(device=self.device)
            input_time  = torch.repeat_interleave(current_time, 2*self.MC_samples,dim=0).to(device=self.device)
            inputNN = torch.cat([input_time.reshape(2*self.MC_samples,1),S_old.float(), V_old.float(),rate.float()],1).to(device=self.device)
            S_old = (S_old + self.driftS(inputNN)*h + self.diffusion(inputNN)*dW).float()
            S_old = torch.cat([S_old,zeros],1)
            S_old = torch.max(S_old,1,keepdim=True)[0]
            Sample_path = torch.cat([Sample_path,S_old],1)
            V_old = V_old + self.driftV(inputNN)*h +self.diffusionV(inputNN)*dW1
            V_old = torch.cat([V_old,zeros],1)
            V_old = torch.max(V_old,1,keepdim=True)[0]
        
        price_all = torch.Tensor()
        for i in range(len(x)):
            strike = x[i,1]*torch.ones(1,1)
            strike_extend = torch.repeat_interleave(strike, 2*self.MC_samples, dim=0).to(device=self.device)
            S_now = Sample_path[:,int(x[i,0])]
            price = torch.cat([S_now-strike_extend,zeros],1)
            price = torch.max(price, 1, keepdim=True)[0]*torch.exp(-self.rate*x[i,0]/360) 
            price = torch.nanmean(price.mean()).view(1,1)
            price_all = torch.cat([price_all,price],0)
    
        return price_all
        