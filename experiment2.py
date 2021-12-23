import numpy as np
import torch
import os
import torch.nn as nn
from scipy.optimize import least_squares
import setenv
from matplotlib import pyplot as plt
from model import Net_SDE,Net_SDE_Pro
from generate import heston


# training function
def train_models(model,target,n_steps,option_info,indices,MC_samples,seedused=1):
    """train nsde model

    Args:
        model (torch.model): nsde model  
        target (np.numpy): 真实期权价格，用于做训练
        n_steps (int): euler scheme的步数
        option_info (list): 存option_info的数组, 0. 资产价格S0 1. 初始波动率V0 2.无风险利率rate 3.heston模型中的相关系数rho
        indices (list): 到期日
        MC_samples (int): 做仿真的sample数
        seedused (int, optional): 随机种子. Defaults to 1.
    """
    S0 = option_info[0]
    V0 = option_info[1]
    rate = option_info[2]
    rho = option_info[3]
    loss_fn = nn.MSELoss() 
    seedused=seedused+1
    torch.manual_seed(seedused)
    np.random.seed(seedused)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001, eps=1e-08,amsgrad=False,betas=(0.9, 0.999), weight_decay=0 )
   #optimizer= torch.optim.Rprop(model.parameters(), lr=0.001, etas=(0.5, 1.2), step_sizes=(1e-07, 1))
    n_epochs = 40
    itercount = 0
    losses_val = [] # for recording the loss val each epoch
    losses = [] # for recording the loss each small batch
    
    for epoch in range(n_epochs):

    # fix the seeds for reproducibility
        np.random.seed(epoch+seedused*1000)
        z_1 = np.random.normal(size=(MC_samples, n_steps))
        z_2 = np.random.normal(size=(MC_samples, n_steps))
        z_1 = np.append(z_1,-z_1,axis=0)
        z_2 = np.append(z_2,-z_2,axis=0)
        z_2  = rho*z_1+np.sqrt(1-rho ** 2)*z_2
        z_1 = torch.tensor(z_1).to(device=device).float()
        z_2 = torch.tensor(z_2).to(device=device).float()

        print('epoch:', epoch)
        
#evaluate and print RMSE validation error at the start of each epoch
        optimizer.zero_grad()
        pred = model(strikes_call,strikes_put, indices, z_1,z_2, 2*MC_samples).detach()
        loss_val=torch.sqrt(loss_fn(pred, target))
        print('validation {}, loss={}'.format(itercount, loss_val.item()))

#store the erorr value

        losses_val.append(loss_val.clone().detach())
        batch_size = 500
# randomly reshufle samples and then use subsamples for training
# this is useful when we want to reuse samples for each epoch
        permutation = torch.randperm(int(2*MC_samples))
        for i in range(0,2*MC_samples, batch_size):
            indices2 = permutation[i:i+batch_size]
            batch_x = z_1[indices2,:]
            batch_y = z_2[indices2,:]         
            optimizer.zero_grad()
            pred = model(strikes_call,strikes_put, indices, z_1,z_2, 2*MC_samples)
            loss=torch.sqrt(loss_fn(pred, target))
            losses.append(loss.clone().detach())
            itercount += 1
            loss.backward()
            optimizer.step()
            print('iteration {}, loss={}'.format(itercount, loss.item()))
        
            
    return model,losses_val,losses  


## setting and training

device='cpu'
torch.set_default_tensor_type('torch.FloatTensor')

path = os.path.abspath(os.path.dirname(__file__))
model_path = path + "/result_model/"
picture_path = path + "/result_picture/"
data_path = path + "/data/"
loss_path = path + "/result_loss/"

ITM_call= torch.Tensor(torch.load(data_path+'Call_ITM_VG_train.pt')).to(device=device)
ITM_put= torch.Tensor(torch.load(data_path+'Put_ITM_VG_train.pt')).to(device=device)
OTM_call= torch.Tensor(torch.load(data_path+'Call_OTM_VG_train.pt')).to(device=device)
OTM_put = torch.Tensor(torch.load(data_path+'Put_OTM_VG_train.pt')).to(device=device)

## settings
MC_samples = 5000
strikes_put=np.arange(60, 101, 5).tolist()
strikes_call=np.arange(100, 141, 5).tolist()
S0 = torch.ones(1, 1)*100
V0 = torch.ones(1,1)*0.04
rate = torch.ones(1, 1)*0.032
asset_info = [S0,V0,rate]
rho = -0.7
option_info = [S0,V0,rate,rho]
n_steps = 360
# generate subdivisions of 1 year interval
timegrid = torch.linspace(0,1,n_steps+1) 
# If using n_steps=48 those corresponds to monthly maturities:
indices = torch.tensor([30,60,90,120,150,180,240,270,300,360])  
target=torch.cat([OTM_call, ITM_call],0)

model = Net_SDE(asset_info = asset_info, n_dim = 3,timegrid = timegrid,
                n_layers= 2,vNetWidth = 20,device = device)

print("==="*10+"training the neural sde model"+"==="*10)
model,losses, losses_val=train_models(model,target,n_steps,option_info ,indices,MC_samples)

#save model
np.save(loss_path+'ex2_nsde_losses.npy', losses)
np.save(loss_path+'ex2_nsde_losses_val.npy', losses_val)
torch.save(model, model_path+"ex2_nsde.pth") 


# first step: calibrate the heston model

S0 = 100
V0 = 0.04
r = 0.032

strikes_all = np.append(strikes_call, strikes_put)
indices = torch.tensor([30,60,90,120,150,180,240,270,300,360])  
asset_input = []
for j in range(len(indices)):
    for i in range(len(strikes_all)):
        asset_input.append([strikes_all[i],indices.numpy()[j]])

def fun(x,asset_input,y):
    return heston(S0,V0,r,x,asset_input) - y



x0 = np.array([-0.3, 0.03, 1.3, 0.3])

"""
#如果用了新的数据集请将这个注释删掉，调用下面的优化，同时删掉下面的赋值
#res_lsq = least_squares(fun, x0, args=(asset_input, target.ravel().numpy()))
#heston_info = res_lsq.x
"""

heston_info = [0.75,0.89,2.23,0.3]
print( "*"*20,"The calibrated heston model params: \
        rho = {},theta = {}, kappa = {} and lambda = {}".format(heston_info[0],heston_info[1],\
        heston_info[2],heston_info[3]),"*"*20)


model_pro = Net_SDE_Pro(heston_info,asset_info,3,timegrid,
                        n_layers=2,vNetWidth = 20,device=device)

for layer in model_pro.modules():
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.constant_(layer.weight, val=0.0)

print("==="*10+"training the neural sde pro model"+"==="*10)
model_pro,losses_pro, losses_val_pro=train_models(model_pro,target,n_steps,option_info ,indices,MC_samples)

# save model
np.save(loss_path+'ex2_nsde_pro_losses.npy', losses_pro)
np.save(loss_path+'ex2_nsde_pro_losses_val.npy', losses_val_pro)
torch.save(model_pro, model_path+"ex2_nsde_pro.pth") 


z_1 = np.random.normal(size=(MC_samples, n_steps))
z_2 = np.random.normal(size=(MC_samples, n_steps))
z_1 = np.append(z_1,-z_1,axis=0)
z_2 = np.append(z_2,-z_2,axis=0)
z_2  = rho*z_1+np.sqrt(1-rho ** 2)*z_2
z_1 = torch.tensor(z_1).to(device=device).float()
z_2 = torch.tensor(z_2).to(device=device).float()


pred = model(strikes_call, strikes_put,indices, z_1,z_2, 2*MC_samples).detach()
pred_pro = model_pro(strikes_call, strikes_put,indices, z_1,z_2, 2*MC_samples).detach()

loss_fn = nn.L1Loss() 
print("The train loss for nsde:    ",loss_fn(pred,target))
print("The train loss for nsde pro:    ",loss_fn(pred_pro,target))

strikes_put_test = np.arange(60, 101, 2.5).tolist()
strikes_call_test = np.arange(100, 141, 2.5).tolist()
pred_test = model(strikes_call_test, strikes_put_test,indices, z_1,z_2, 2*MC_samples).detach()
pred_pro_test = model_pro(strikes_call_test, strikes_put_test,indices, z_1,z_2, 2*MC_samples).detach()

ITM_call_test = torch.Tensor(torch.load(data_path+'Call_ITM_VG_test.pt')).to(device=device)
OTM_call_test = torch.Tensor(torch.load(data_path+'Call_OTM_VG_test.pt')).to(device=device)
target_test = torch.cat([OTM_call_test, ITM_call_test],0)

nsde_test_loss = (loss_fn(pred_test,target_test)*target_test.shape[0]*target_test.shape[1]
                  -loss_fn(pred,target)*target.shape[0]*target.shape[1])/\
                 (target_test.shape[0]*target_test.shape[1]-target.shape[0]*target.shape[1])
nsde_pro_test_loss = (loss_fn(pred_pro_test,target_test)*target_test.shape[0]*target_test.shape[1]
                  -loss_fn(pred_pro,target)*target.shape[0]*target.shape[1])/\
                 (target_test.shape[0]*target_test.shape[1]-target.shape[0]*target.shape[1])

print("The test loss for nsde:  ",nsde_test_loss)
print("The test loss for nsde pro:  ",nsde_pro_test_loss)


## save picture
fig,ax = plt.subplots()
ax.plot(np.arange(1,201),losses_val[0:200])
ax.set(xlabel='iteration',ylabel='loss')
ax.set_title('loss for NSDE')
plt.savefig(picture_path+"ex2_NSDE_loss.png")

fig,ax = plt.subplots()
ax.plot(np.arange(1,201),losses_val_pro[0:200])
ax.set(xlabel='iteration',ylabel='loss')
ax.set_title('loss for NSDE_PRO')
plt.savefig(picture_path+"ex2_NSDE_PRO_loss.png")

