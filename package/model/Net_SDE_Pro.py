import torch
import torch.nn as nn
import numpy as np
from model import Net_Timestep

class Net_SDE_Pro(nn.Module):
    
    def __init__(self,heston_info, asset_info,n_dim, timegrid,  n_layers, vNetWidth, device):
        """SDE_PRO的模型

        Args:
            heston_info (list): heston calibrate parameter 0. rho 1. theta 2.kappa 3. lambd
            asset_info (list): asset info  0. S0 1. V0 2. rate 无风险利率
            n_dim (int): 输入维度
            timegrid (list): 做euler scheme的步数，基本是是以天为单位，最大到期日到期
            strikes_call (list): in the money call的strike price
            strikes_put (list): out the money call的strike price
            n_layers (int): nn网络有几层
            vNetWidth (int): nn网络宽度
            device (str): 决定是cpu还是gpu
        """
        super(Net_SDE_Pro, self).__init__()
        self.timegrid = timegrid
        self.device = device
        self.S0 = asset_info[0]
        self.V0 = asset_info[1]
        self.rate = asset_info[2]
        
        self.rho = heston_info[0]
        self.theta = heston_info[1]
        self.kappa = heston_info[2]
        self.lamdb = heston_info[3]
        
        #Input to each coefficient (NN) will be (t,S_t,V_t)
        self.drift = Net_Timestep(n_dim = n_dim, n_out=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusion =Net_Timestep(n_dim = n_dim, n_out=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.driftV = Net_Timestep(n_dim = n_dim, n_out=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusionV = Net_Timestep(n_dim = n_dim, n_out=1, n_layers=n_layers, vNetWidth=vNetWidth)
        
    def forward(self, strikes_call, strikes_put, indices, z,z1, MC_samples): 
        S_old = torch.repeat_interleave(self.S0, MC_samples, dim=0)
        V_old_abs = torch.repeat_interleave(self.V0, MC_samples, dim=0)    
        K_call = strikes_call
        K_put = strikes_put
        zeros = torch.repeat_interleave(torch.zeros(1,1), MC_samples, dim=0)
        average_SS = torch.Tensor()
        average_SS1 = torch.Tensor()
        # use fixed step size
        h = (self.timegrid[1]-self.timegrid[0]).to(device = self.device)
        n_steps = len(self.timegrid)-1

        
        # Solve for S_t, V_t (Euler)
        for i in range(1, len(self.timegrid)):
            dW = (torch.sqrt(h) * z[:,i-1]).reshape(MC_samples,1)
            dW1 = (torch.sqrt(h) * z1[:,i-1]).reshape(MC_samples,1)
            current_time = (torch.ones(1, 1)*self.timegrid[i-1]).to(device=self.device)
            input_time  = torch.repeat_interleave(current_time, MC_samples,dim=0).to(device=self.device)
            inputNN = torch.cat([input_time.reshape(MC_samples,1),S_old, V_old_abs],1).to(device=self.device)
            S_old = S_old + (S_old*self.rate + self.drift(inputNN))*h + (S_old*torch.sqrt(V_old_abs) + self.diffusion(inputNN))*dW
            V_new = V_old_abs + (self.kappa*(self.theta - V_old_abs) + self.driftV(inputNN))*h + (self.lamdb + self.diffusionV(inputNN))*dW1
            V_old_abs = torch.cat([V_new,zeros+0.01],1)
            V_old_abs= torch.max(V_old_abs,1,keepdim=True)[0]
        
            # If particular timestep is a maturity for Vanilla option
            
            if int(i) in indices:
                
                Z_new=torch.Tensor()
                countstrikecall=-1
                if(torch.any(torch.isnan(S_old))):
                    print(S_old)
                    print(torch.sum(torch.isnan(S_old)))
                    
            # Evaluate call (OTM) option prices 
                
                for strike in K_call:
                    countstrikecall+=1
                    strike = torch.ones(1,1)*strike
                    K_extended = torch.repeat_interleave(strike, MC_samples, dim=0).float()
                    # Since we use the same number of maturities for vanilla calls and puts: 
                    
                    price = torch.cat([S_old-K_extended,zeros],1) #call OTM
                    # Discounting assumes we use 2-year time horizon 
                    
                    price = torch.max(price, 1, keepdim=True)[0]*torch.exp(-self.rate*1*i/n_steps)
                    Z_new= torch.cat([Z_new,price],1)  
               # MC step:
            
                avg_S = torch.cat([torch.nanmean(p).view(1,1) for p in Z_new.T], 0)
                #print("OTM")
                #print(avg_S)
                
                average_SS = torch.cat([average_SS,avg_S.T],0) #call OTM    
                
                
          # Evaluate call (ITM) option prices  
                Z_new=torch.Tensor()
                for strike in K_put:
                    strike = torch.ones(1,1)*strike
                    K_extended = torch.repeat_interleave(strike, MC_samples, dim=0).float()
                    price = torch.cat([S_old-K_extended, zeros], 1) #Call ITM
                    price = torch.max(price, 1, keepdim=True)[0]*torch.exp(-self.rate*1*i/n_steps) 
                    Z_new= torch.cat([Z_new,price],1)       
            # MC step             
                avg_S = torch.cat([torch.nanmean(p).view(1,1) for p in Z_new.T], 0) 
                #print("ITM")
                #print(avg_S)                      
                average_SS1 = torch.cat([average_SS1,avg_S.T],0)      
                
            # Return model implied vanilla option prices    
                
        return torch.cat([average_SS,average_SS1],0)