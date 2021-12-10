import torch
import torch.nn as nn
import numpy as np
from model import Net_Timestep

class Net_SDE(nn.Module):
    
    def __init__(self, asset_info, n_dim, timegrid, strikes_call, strikes_put, n_layers, vNetWidth, device):   
        """SDE的模型

        Args:
            asset_info (list): asset info  0. S0 1. V0 2. rate 无风险利率
            n_dim (int): 输入维度
            timegrid (list): 做euler scheme的步数，基本是是以天为单位，最大到期日到期
            strikes_call (list): in the money call的strike price
            strikes_put (list): out the money call的strike price
            n_layers (int): nn网络有几层
            vNetWidth (int): nn网络宽度
            device (str): 决定是cpu还是gpu
        """
        super(Net_SDE, self).__init__()
        self.n_dim = n_dim
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        self.strikes_put = strikes_put
        self.S0 = asset_info[0]
        self.V0 = asset_info[1]
        self.rate = asset_info[2]
        
        
        #Input to each coefficient (NN) will be (t,S_t,V_t)
        self.drift = Net_Timestep(n_dim=n_dim+2, n_out=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusion =Net_Timestep(n_dim=n_dim+2, n_out=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.driftV = Net_Timestep(n_dim=n_dim+2, n_out=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusionV = Net_Timestep(n_dim=n_dim+2, n_out=1, n_layers=n_layers, vNetWidth=vNetWidth)
        
    def forward(self, indices, z,z1, MC_samples): 
        S_old = torch.repeat_interleave(self.S0, MC_samples, dim=0).to(device=self.device)
        V_old = torch.repeat_interleave(self.V0, MC_samples, dim=0).to(device=self.device)  
        K_call = self.strikes_call
        K_put = self.strikes_put
        zeros = torch.repeat_interleave(torch.zeros(1,1), MC_samples, dim=0).to(device=self.device)
        average_SS = torch.Tensor().to(device=self.device)
        average_SS1 = torch.Tensor().to(device=self.device)
        # use fixed step size
        h = (self.timegrid[1]-self.timegrid[0]).to(device=self.device)
        n_steps = len(self.timegrid)-1
        # set maturity counter
        countmat=-1
        
        # Solve for S_t, V_t (Euler)
        
        for i in range(1, len(self.timegrid)):
            dW = (torch.sqrt(h) * z[:,i-1]).reshape(MC_samples,1).to(device=self.device)
            dW1 = (torch.sqrt(h) * z1[:,i-1]).reshape(MC_samples,1).to(device=self.device)
            current_time = (torch.ones(1, 1)*self.timegrid[i-1]).to(device=self.device)
            input_time  = torch.repeat_interleave(current_time, MC_samples,dim=0).to(device=self.device)
            inputNN = torch.cat([input_time.reshape(MC_samples,1),S_old, V_old],1).to(device=self.device)
            S_new = S_old + S_old*self.rate*h + self.diffusion(inputNN)*dW 
            S_new = torch.cat([S_new,zeros],1)
            S_new=torch.max(S_new,1,keepdim=True)[0]
            S_old = S_new
            V_new = V_old + self.driftV(inputNN)*h +self.diffusionV(inputNN)*dW1 
            V_old = V_new 
        
            # If particular timestep is a maturity for Vanilla option
            
            if int(i) in indices:
                countmat+=1
                Z_new=torch.Tensor()
                countstrikecall=-1
                
            # Evaluate put (OTM) and call (OTM) option prices 
                
                for strike in K_call:
                    countstrikecall+=1
                    strike = torch.ones(1,1)*strike
                    K_extended = torch.repeat_interleave(strike, MC_samples, dim=0).float()

                    # Since we use the same number of maturities for vanilla calls and puts: 
                    
                    price = torch.cat([S_old-K_extended,zeros],1) #call OTM
                    
                    # Discounting assumes we use 2-year time horizon 
                    price = torch.max(price, 1, keepdim=True)[0]*torch.exp(-self.rate*1*i/n_steps)
               # Comment out Z_new and Z_newP_OTM if using CV with Black-Scholes:          
                    Z_new= torch.cat([Z_new,price],1)  
                    
               # MC step:
            
                avg_S = torch.cat([p.mean().view(1,1) for p in Z_new.T], 0)
                average_SS = torch.cat([average_SS,avg_S.T],0) #call OTM 
                countstrikeput=-1
                
          # Evaluate put (ITM) and call (ITM) option prices 
                
                Z_new=torch.Tensor()
                for strike in K_put:
                    countstrikeput+=1
                    strike = torch.ones(1,1)*strike
                    K_extended = torch.repeat_interleave(strike, MC_samples, dim=0).float()
                    price = torch.cat([S_old-K_extended,zeros],1) #Call ITM
                    price = torch.max(price, 1, keepdim=True)[0]*torch.exp(-self.rate*1*i/n_steps)
                    
                    
                    Z_new= torch.cat([Z_new,price],1) 
                    
            # MC step         
                    
                avg_S = torch.cat([p.mean().view(1,1) for p in Z_new.T], 0)                        
                average_SS1 = torch.cat([average_SS1,avg_S.T],0)      
                    
            # Return model implied vanilla option prices    
                
        return torch.cat([average_SS,average_SS1],0)  