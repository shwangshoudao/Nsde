import numpy as np
import pandas as pd
from sklearn import model_selection
import torch
import os
import torch.nn as nn
from scipy.optimize import least_squares
import setenv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from model import Net_SDE_Revised,Net_SDE_Revised_Pro,two_gate
from generate import heston


# training function
def train_models(model,target,train_x,path,losses_val,n_epochs,lr,seedused=1):
    """train nsde model

    Args:
        model (torch.model): nsde model  
        target (np.numpy): 真实期权价格，用于做训练
        train_x(np.numpy): 用于训练的的数据，第一列到期日，第二列行权价格
        path(str): 存储model的path
        losses_val(list): 记录loss的list
        n_epochs(int): 做多少个epoch的训练
        lr(float): 优化器的学习率
        seedused (int, optional): 随机种子. Defaults to 1.
    """
    loss_fn = nn.MSELoss() 
    loss_fn_test = nn.L1Loss()
    seedused=seedused+1
    torch.manual_seed(seedused)
    np.random.seed(seedused)
    #evaluate and print RMSE validation error at the start of each epoch
    
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr, eps=1e-08,amsgrad=False,betas=(0.9, 0.999), weight_decay=0 )
    
    for epoch in range(n_epochs):

        print('epoch:', epoch)
        
#evaluate and print RMSE validation error at the start of each epoch
        optimizer.zero_grad()
        pred = model(train_x).float()
        loss_val= (torch.sqrt(loss_fn(pred, target))).float()
        if loss_val.clone().detach() < min(losses_val):
                torch.save(model,path)
                print("------------sucess-----------------")
        print('validation {}, loss={}'.format(epoch, loss_fn_test(pred,target).item()))
        losses_val.append(loss_val.clone().detach())
        loss_val.backward()
        optimizer.step()   
    return losses_val


## setting and training

device='cpu'
torch.set_default_tensor_type('torch.FloatTensor')

path = os.path.abspath(os.path.dirname(__file__))
model_path = path + "/result_model/"
picture_path = path + "/result_picture/"
data_path = path + "/data/"
loss_path = path + "/result_loss/"

origin_data = pd.read_excel(data_path+"Call.xlsx")
origin_data["volume"] = origin_data["volume"].\
                        apply(lambda x: x.replace(',','') if type(x)==str else x).astype(int)
origin_data = origin_data[origin_data["volume"]>=200]
spy_data = origin_data.loc[:,["Time to Maturity","strike","last"]].astype(float).values

train_data,test_data = train_test_split(spy_data,test_size=0.2,random_state=42)
x_train = train_data[:,0:2]
Y_train = train_data[:,2]
x_test = test_data[:,0:2]
Y_test = test_data[:,2]

nsde_path = model_path + "ex3_nsde.pth"
nsde_pro_path = model_path + "ex3_nsde_pro.pth"
gate_path = model_path + "ex3_gated.pth"
losses_val_nsde_path = loss_path+'ex3_nsde_losses_val.npy'
losses_val_pro_path = loss_path+'ex3_nsde_pro_losses_val.npy'
losses_val_gate_path = loss_path+'ex3_gate_losses_val.npy'
losses_val_nsde =[1000]
losses_val_pro = [1000]
losses_val_gate = [1000]

S0 = 318.92
V0 = 0.03
rate = 0.06
asset_info = [S0,V0,rate]

if(not os.path.exists(nsde_path) or not os.path.exists(losses_val_nsde_path)):
    model = Net_SDE_Revised(asset_info, 4,2,20,1000,device)
    print("==="*10+"training the neural sde model"+"==="*10)
    losses_val_nsde = train_models(model,torch.tensor(Y_train,dtype=torch.float32).view(len(Y_train),1),
                x_train,nsde_path,losses_val_nsde,400,0.01)
    np.save(losses_val_nsde_path, losses_val_nsde) 

if(not os.path.exists(nsde_pro_path) or not os.path.exists(losses_val_pro_path)):
    # heston calibration
    asset_input = x_train.copy()
    asset_input[:,0] = asset_input[:,0]/360

    def fun(x,asset_input,y):
        return heston(S0,V0,rate,x,asset_input) - y

    x0 = np.array([-0.3, 0.03, 1.3, 0.3])

    res_lsq = res_lsq = least_squares(fun, x0, 
                    bounds=([-1, -np.inf,-np.inf,-np.inf], [1,np.inf, np.inf, np.inf]),
                    args=(asset_input, Y_train.ravel()))
    heston_info = res_lsq.x

    print( "*"*20,"The calibrated heston model params: \
            rho = {},theta = {}, kappa = {} and lambda = {}".format(heston_info[0],heston_info[1],\
            heston_info[2],heston_info[3]),"*"*20)


    model_pro = Net_SDE_Revised_Pro(heston_info,asset_info, 4,2,20,1000,device)

    for layer in model_pro.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.constant_(layer.weight, val=0.0)

    print("==="*10+"training the neural sde pro model"+"==="*10)
    losses_val_pro = train_models(model_pro,torch.tensor(Y_train,dtype=torch.float32).view(len(Y_train),1),
                x_train,nsde_pro_path,losses_val_pro,400,0.01)

    # save loss
    np.save(losses_val_pro_path, losses_val_pro) 



if(not os.path.exists(gate_path) or not os.path.exists(losses_val_gate_path)):
    model_gate = two_gate(1,30)
    print("==="*10+"training the two gate model"+"==="*10)
    losses_val_gate = train_models(model_gate,torch.tensor(Y_train,dtype=torch.float32),
                x_train,gate_path,losses_val_gate,100000,0.001)
    
    # save loss
    np.save(losses_val_gate_path, losses_val_gate) 


#result
model = torch.load(nsde_path)
model_pro = torch.load(nsde_pro_path)
model_gate = torch.load(gate_path)
losses_val_nsde = np.load(losses_val_nsde_path)
losses_val_pro = np.load(losses_val_pro_path)

pred = model(x_train).detach()
pred_pro = model_pro(x_train).detach()
pred_gate = model_gate(x_train).detach().view(len(x_train),1)

loss_fn = nn.L1Loss() 
print("The train loss for nsde:    ",loss_fn(pred,torch.tensor(Y_train,dtype=torch.float32).view(len(Y_train),1)))
print("The train loss for nsde pro:    ",loss_fn(pred_pro,torch.tensor(Y_train,dtype=torch.float32).view(len(Y_train),1)))
print("The train loss for gate:    ",loss_fn(pred_gate,torch.tensor(Y_train,dtype=torch.float32).view(len(Y_train),1)))

pred_test = model(x_test).detach()
pred_pro_test = model_pro(x_test).detach()
pred_gate_test = model_gate(x_test).detach().view(len(x_test),1)

nsde_test_loss = loss_fn(pred_test,torch.tensor(Y_test,dtype=torch.float32).view(len(Y_test),1))
nsde_pro_test_loss = loss_fn(pred_pro_test,torch.tensor(Y_test,dtype=torch.float32).view(len(Y_test),1))
gate_test_loss = loss_fn(pred_gate_test,torch.tensor(Y_test,dtype=torch.float32).view(len(Y_test),1))
print("The test loss for nsde:  ",nsde_test_loss)
print("The test loss for nsde pro:  ",nsde_pro_test_loss)
print("The test loss for gate:  ",gate_test_loss)

## save picture
fig,ax = plt.subplots()
ax.plot(np.arange(1,401),losses_val_nsde[1:401])
ax.set(xlabel='iteration',ylabel='loss')
ax.set_title('loss for NSDE')
plt.savefig(picture_path+"ex3_NSDE_loss.png")

fig,ax = plt.subplots()
ax.plot(np.arange(1,401),losses_val_pro[1:401])
ax.set(xlabel='iteration',ylabel='loss')
ax.set_title('loss for NSDE_PRO')
plt.savefig(picture_path+"ex3_NSDE_PRO_loss.png")
