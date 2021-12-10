import torch
import torch.nn as nn
import numpy as np
from model import Net_Timestep

class Net_SDE_Revised(nn.Module):
    
    def __init__(self, dim,n_layers, vNetWidth, S0,S01,V0,rate,MC_samples,device):
        
        super(Net_SDE_Revised, self).__init__()
        self.dim = dim
        self.device = device
        self.S0 = S0
        self.S01 = S01
        self.rate = rate
        self.V0 = V0
        self.MC_samples = MC_samples
        
        #Input to each coefficient (NN) will be (t,S_t,V_t)
        self.driftS = Net_Timestep(dim=dim+3, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.driftS1 = Net_Timestep(dim=dim+3, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusion =Net_Timestep(dim=dim+3, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.driftV = Net_Timestep(dim=dim+3, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusionV = Net_Timestep(dim=dim+3, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusionV1 = Net_Timestep(dim=dim+3, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.poisson_change = Net_Timestep(dim=1,nOut=1 ,n_layers=1,vNetWidth=1)
        self.jump_size_mean = Net_Timestep(dim=1,nOut=1 ,n_layers=1,vNetWidth=1)
        self.jump_size_vol = Net_Timestep(dim=1,nOut=1 ,n_layers=1,vNetWidth=1)
        
        
    def forward(self,x): 
        S_old = torch.repeat_interleave(self.S0, 2*self.MC_samples, dim=0).to(device=self.device)
        S_old_new = torch.repeat_interleave(self.S01, 2*self.MC_samples, dim=0).to(device=self.device)
        V_old = torch.repeat_interleave(self.V0, 2*self.MC_samples, dim=0).to(device=self.device)  
        V_old_new = torch.repeat_interleave(self.V0, 2*self.MC_samples, dim=0).to(device=self.device)
        rate = torch.repeat_interleave(self.rate, 2*self.MC_samples, dim=0).to(device=self.device)  
        zeros = torch.repeat_interleave(torch.zeros(1,1), 2*self.MC_samples, dim=0).to(device=self.device)
        Sample_path = torch.Tensor().to(device=self.device)
        Sample_path = torch.cat([Sample_path,S_old],1)
        Sample_path1 = torch.Tensor().to(device=self.device)
        Sample_path1 = torch.cat([Sample_path1,S_old_new],1)
        # use fixed step size
        timegrid = torch.linspace(0,int(x[:,0].max()),int(x[:,0].max()+1))
        h = (timegrid[1]-timegrid[0]).to(device=self.device)
        n_steps = len(timegrid)-1
        
        z = torch.randn(self.MC_samples,n_steps)
        z1 = torch.randn(self.MC_samples,n_steps)
        z1 = -0.5*z + np.sqrt(0.75)*z1
        z = torch.cat([z,-z],axis=0)
        z1 = torch.cat([z1,-z1],axis=0)
        small_zeros = torch.zeros(1,1)
        poisson_rate_0 = 0.001*torch.ones(1,1)
        mean_0 = -0.388*torch.ones(1,1)
        vol_0 = 4.33*torch.ones(1,1)
        poisson_rate = self.poisson_change(poisson_rate_0)
        poisson_rate = torch.cat([poisson_rate,small_zeros+0.0001],1)
        poisson_rate = torch.max(poisson_rate,1,keepdim=True)[0]
        poisson_rate = poisson_rate*torch.ones(2*self.MC_samples, n_steps)
        mean = self.jump_size_mean(mean_0)
        vol = self.jump_size_vol(vol_0)
        vol_cat = torch.cat([vol,small_zeros+0.0001],1)
        vol = torch.max(vol_cat,1,keepdim=True)[0]
        jump_size = torch.normal(float(mean[0,0].detach().numpy()),float(vol[0,0].detach().numpy()),size=(2*self.MC_samples, n_steps))
        poisson = torch.poisson(poisson_rate)
        """poisson_rate_old1 = 0.001*torch.ones(1,1)
        mean_old1 = -0.388*torch.ones(1,1)
        vol_old1 = 4.33*torch.ones(1,1)"""
        z_new = torch.randn(self.MC_samples,30)
        z1_new = torch.randn(self.MC_samples,30)
        z1_new = -0.5*z_new + np.sqrt(0.75)*z1_new
        z_new = torch.cat([z_new,-z_new],axis=0)
        z1_new = torch.cat([z1_new,-z1_new],axis=0)

        #include underlying asset information
        """for j in range(1,31):
            poisson1 = poisson_rate_old1[0,0]*torch.ones(2*self.MC_samples,1)
            poisson1 = torch.poisson(poisson1)
            jump_size1 = torch.normal(float(mean_old1[0,0].detach().numpy()), float(vol_old1[0,0].detach().numpy()), size=(2*self.MC_samples, 1))
            poisson_rate_old1 = self.poisson_change(poisson_rate_old1)
            poisson_rate_old1 = torch.cat([poisson_rate_old1,small_zeros],1)
            poisson_rate_old1 = torch.max(poisson_rate_old1,1,keepdim=True)[0]
            mean_old1 = self.jump_size_mean(mean_old1)
            vol_old1 = self.jump_size_vol(vol_old1)
            vol_old_cat1 = torch.cat([vol_old1,small_zeros+0.0001],1)
            vol_old1 = torch.max(vol_old_cat1,1,keepdim=True)[0]
            dW = (torch.sqrt(h) * z_new[:,j-1]).reshape(2*self.MC_samples,1).to(device=device)
            dW1 = (torch.sqrt(h) * z1_new[:,j-1]).reshape(2*self.MC_samples,1).to(device=device)
            jump = (jump_size1*poisson1).reshape(2*self.MC_samples,1).to(device=device)
            current_time = (torch.ones(1, 1)*j).to(device=device)
            input_time  = torch.repeat_interleave(current_time, 2*self.MC_samples,dim=0).to(device=device)
            inputNN = torch.cat([input_time.reshape(2*self.MC_samples,1),S_old.float(), V_old.float(),rate.float()],1).to(device=device)
            

            S_old_new = (S_old_new + self.driftS1(inputNN)*h + self.diffusion(inputNN)*dW+jump).float()
            S_old_new = torch.cat([S_old_new,zeros],1)
            S_old_new = torch.max(S_old_new,1,keepdim=True)[0]
            Sample_path1 = torch.cat([Sample_path1,S_old_new],1)
            V_old_new = V_old_new + self.driftV(inputNN)*h +self.diffusionV(inputNN)*dW + self.diffusionV1(inputNN)*dW1

        underlying = Sample_path1[:,0:30].mean(dim=0).view(30,1)"""

        # Solve for S_t, V_t (Euler)
        for i in range(1, len(timegrid)):
            """poisson = poisson_rate_old[0,0]*torch.ones(2*self.MC_samples,1)
            poisson = torch.poisson(poisson)
            jump_size = torch.normal(float(mean_old[0,0].detach().numpy()), float(vol_old[0,0].detach().numpy()), size=(2*self.MC_samples, 1))
            poisson_rate_old = self.poisson_change(poisson_rate_0)
            poisson_rate_old = torch.cat([poisson_rate_old,small_zeros+0.0001],1)
            poisson_rate_old = torch.max(poisson_rate_old,1,keepdim=True)[0]
            mean_old = self.jump_size_mean(mean_0)
            vol_old = self.jump_size_vol(vol_0)
            vol_old_cat = torch.cat([vol_old,small_zeros+0.0001],1)
            vol_old = torch.max(vol_old_cat,1,keepdim=True)[0]"""
            dW = (torch.sqrt(h) * z[:,i-1]).reshape(2*self.MC_samples,1).to(device=self.device)
            dW1 = (torch.sqrt(h) * z1[:,i-1]).reshape(2*self.MC_samples,1).to(device=self.device)
            jump = (jump_size[:,i-1]*poisson[:,i-1]).reshape(2*self.MC_samples,1).to(device=self.device)
            current_time = (torch.ones(1, 1)*timegrid[i-1]).to(device=self.device)
            input_time  = torch.repeat_interleave(current_time, 2*self.MC_samples,dim=0).to(device=self.device)
            inputNN = torch.cat([input_time.reshape(2*self.MC_samples,1),S_old.float(), V_old.float(),rate.float()],1).to(device=self.device)
            

            S_old = (S_old + self.driftS(inputNN)*h + self.diffusion(inputNN)*dW+jump).float()
            S_old = torch.cat([S_old,zeros],1)
            S_old = torch.max(S_old,1,keepdim=True)[0]
            Sample_path = torch.cat([Sample_path,S_old],1)
            V_old = V_old + self.driftV(inputNN)*h +self.diffusionV(inputNN)*dW + self.diffusionV1(inputNN)*dW1
            
        
        price_all = torch.Tensor()
        for i in range(len(x)):
            strike = x[i,1]*torch.ones(1,1)
            strike_extend = torch.repeat_interleave(strike, 2*self.MC_samples, dim=0).to(device=self.device)
            S_now = Sample_path[:,int(x[i,0])]
            price = torch.cat([S_now-strike_extend,zeros],1)
            price = torch.max(price, 1, keepdim=True)[0]*torch.exp(-self.rate*1*x[i,0]) 
            price = price.mean().view(1,1)
            price_all = torch.cat([price_all,price],0)
    
                    
        return price_all
        