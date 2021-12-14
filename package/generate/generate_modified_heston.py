import torch
import torch.nn as nn
import numpy as np



class heston_modified(nn.Module):
    
    def __init__(self, timegrid, strikes_call, strikes_put, device):
        
        super(heston_modified, self).__init__()
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        self.strikes_put = strikes_put
        
        #Input to each coefficient (NN) will be (t,S_t,V_t)

        
    def forward(self, S0, V0, rate, indices, z,z1, MC_samples): 
        S_old = torch.repeat_interleave(S0, MC_samples, dim=0)
        V_old_abs = torch.repeat_interleave(V0, MC_samples, dim=0)    
        K_call = self.strikes_call
        K_put = self.strikes_put
        zeros = torch.repeat_interleave(torch.zeros(1,1), MC_samples, dim=0)
        average_SS = torch.Tensor()
        average_SS1 = torch.Tensor()
        average_SS_OTM = torch.Tensor()
        average_SS1_ITM = torch.Tensor()
        # use fixed step size
        h = self.timegrid[1]-self.timegrid[0]
        n_steps = len(self.timegrid)-1
        # set maturity counter
        countmat=-1
        
        # Solve for S_t, V_t (Euler)
        
        for i in range(1, len(self.timegrid)):
            dW = (torch.sqrt(h) * z[:,i-1]).reshape(MC_samples,1)
            dW1 = (torch.sqrt(h) * z1[:,i-1]).reshape(MC_samples,1)
            S_new = S_old+S_old*rate*h+S_old*torch.sqrt(V_old_abs)*dW1
            S_old = S_new
            V_new = V_old_abs + 1.5*(0.04-V_old_abs)*h+0.3*dW
            V_old_abs = torch.cat([V_new,zeros],1)
            V_old_abs= torch.max(V_old_abs,1,keepdim=True)[0]
        
            # If particular timestep is a maturity for Vanilla option
            
            if int(i) in indices:
                countmat+=1
                Z_new=torch.Tensor()
                Z_newP_ITM = torch.Tensor()
                Z_newP_OTM = torch.Tensor()
                countstrikecall=-1
                
            # Evaluate put (OTM) and call (OTM) option prices 
                
                for strike in K_call:
                    countstrikecall+=1
                    strike = torch.ones(1,1)*strike
                    strike_put = torch.ones(1,1)*K_put[countstrikecall]
                    K_extended = torch.repeat_interleave(strike, MC_samples, dim=0).float()
                    K_extended_put = torch.repeat_interleave(strike_put, MC_samples, dim=0).float()

                    # Since we use the same number of maturities for vanilla calls and puts: 
                    
                    price = torch.cat([S_old-K_extended,zeros],1) #call OTM
                    price_OTM = torch.cat([K_extended_put-S_old,zeros],1) #put OTM
                    
                    # Discounting assumes we use 2-year time horizon 
                    
                    price = torch.max(price, 1, keepdim=True)[0]*torch.exp(-rate*1*i/n_steps)
                    price_OTM = torch.max(price_OTM, 1, keepdim=True)[0]*torch.exp(-rate*1*i/n_steps)
                    
                
                    Z_new= torch.cat([Z_new,price],1)  
                    Z_newP_OTM= torch.cat([Z_newP_OTM,price_OTM],1)  
                    
               # MC step:
            
                avg_S = torch.cat([p.mean().view(1,1) for p in Z_new.T], 0)
                avg_SSP_OTM = torch.cat([p.mean().view(1,1) for p in Z_newP_OTM.T], 0)
                average_SS = torch.cat([average_SS,avg_S.T],0) #call OTM
                average_SS_OTM = torch.cat([average_SS_OTM,avg_SSP_OTM.T],0) #put OTM       
                countstrikeput=-1
                
          # Evaluate put (ITM) and call (ITM) option prices 
                
                Z_new=torch.Tensor()
                for strike in K_put:
                    countstrikeput+=1
                    strike = torch.ones(1,1)*strike
                    strike_call = torch.ones(1,1)*K_call[countstrikeput]
                    K_extended = torch.repeat_interleave(strike, MC_samples, dim=0).float()
                    K_extended_call = torch.repeat_interleave(strike_call, MC_samples, dim=0).float()
                    price_ITM = torch.cat([K_extended_call-S_old,zeros],1) #put ITM
                    price = torch.cat([S_old-K_extended,zeros],1) #Call ITM
                    price = torch.max(price, 1, keepdim=True)[0]*torch.exp(-rate*1*i/n_steps)
                    price_ITM = torch.max(price_ITM, 1, keepdim=True)[0]*torch.exp(-rate*1*i/n_steps)
                    
                    
                    Z_new= torch.cat([Z_new,price],1) 
                    Z_newP_ITM= torch.cat([Z_newP_ITM,price_ITM],1)    
                    
            # MC step         
                    
                avg_S = torch.cat([p.mean().view(1,1) for p in Z_new.T], 0)
                avg_SSP_ITM = torch.cat([p.mean().view(1,1) for p in Z_newP_ITM.T], 0)
                average_SS1_ITM = torch.cat([average_SS1_ITM,avg_SSP_ITM.T],0)                            
                average_SS1 = torch.cat([average_SS1,avg_S.T],0)      
                    
            # Return model implied vanilla option prices    
                
        return torch.cat([average_SS,average_SS_OTM,average_SS1,average_SS1_ITM  ],0)  
    


if __name__ == "__main__":
    import os
    device = "cpu"
    path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../.."))
    ## setting
    MC_samples=200000
    n_steps=360
    rho = -0.7
    Z2=np.zeros((40,9))
    strikes_put=np.arange(60, 101, 5).tolist()
    strikes_call=np.arange(100, 141, 5).tolist()
    n_steps=360
    timegrid = torch.linspace(0,1,n_steps+1) 
    indices = torch.tensor([30,60,90,120,150,180,240,270,300,360])  
    model = heston_modified(timegrid=timegrid, strikes_call=strikes_call,strikes_put=strikes_put, device=device)
    S0 = torch.ones(1, 1)*100
    V0 = torch.ones(1,1)*0.034
    rate = torch.ones(1, 1)*0.032
    
    
    ## generate
    z_1 = np.random.normal(size=(MC_samples, n_steps))
    z_2 = np.random.normal(size=(MC_samples, n_steps))
    z_1 = np.append(z_1,-z_1,axis=0)
    z_2 = np.append(z_2,-z_2,axis=0)
    zz  = rho*z_1+np.sqrt(1-rho ** 2)*z_2
    z_1 = torch.tensor(z_1).to(device=device).float()
    z_2 = torch.tensor(z_2).to(device=device).float()
    zz = torch.tensor(zz).to(device=device).float()
    model=model.to(device=device)
    Z=model(S0, V0, rate, indices, z_1,zz, 2*MC_samples).float().to(device=device)
    Z=Z.detach().to(device='cpu').numpy()
    
    
    ## saving
    Call_OTM_Unit=Z[0:10,:]
    Put_OTM_Unit=Z[10:20,:]
    Call_ITM_Unit=Z[20:30,:]
    Put_ITM_Unit=Z[30:40,:]
    
    torch.save(Call_OTM_Unit,path+'/data/Call_OTM_modified_Heston_train.pt')
    torch.save(Put_OTM_Unit,path+'/data/Put_OTM_modified_Heston_train.pt')
    torch.save(Call_ITM_Unit,path+'/data/Call_ITM_modified_Heston_train.pt')
    torch.save(Put_ITM_Unit,path+'/data/Put_ITM_modified_Heston_train.pt') 