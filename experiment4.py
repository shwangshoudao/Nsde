#用于衡量两个相似股票之间的相似性

import numpy as np
from numpy.lib.function_base import select
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
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001, eps=1e-08,amsgrad=False,betas=(0.9, 0.999), weight_decay=0 )
    
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


device='cpu'
torch.set_default_tensor_type('torch.FloatTensor')

path = os.path.abspath(os.path.dirname(__file__))
model_path = path + "/result_model/"
picture_path = path + "/result_picture/"
data_path = path + "/data/option/"
loss_path = path + "/result_loss/"
aapl_path = data_path + "AAPL_2021-05-28_call_revised.xlsx"
spy_path = data_path + "SPY_2021-05-28_call_revised.xlsx"
amzn_path = data_path + "AMZN_2021-05-28_call_revised.xlsx"

aapl_data = pd.read_excel(aapl_path)
spy_data = pd.read_excel(spy_path)
amzn_data = pd.read_excel(amzn_path)
appl_data = aapl_data[aapl_data["volume"]>=50]
spy_data = spy_data[spy_data["volume"]>=50]
amzn_data = amzn_data[amzn_data["volume"]>=50]

aapl_data = aapl_data.loc[:,["Maturity","strike","lastPrice"]].astype(float).values
spy_data = spy_data.loc[:,["Maturity","strike","lastPrice"]].astype(float).values
amzn_data = amzn_data.loc[:,["Maturity","strike","lastPrice"]].astype(float).values

aapl_train_data,aapl_test_data = train_test_split(aapl_data,test_size=0.2,random_state=42)
aapl_x_train = aapl_train_data[:,0:2]
aapl_Y_train = aapl_train_data[:,2]
aapl_x_test = aapl_test_data[:,0:2]
aapl_Y_test = aapl_test_data[:,2]

spy_train_data,spy_test_data = train_test_split(spy_data,test_size=0.2,random_state=42)
spy_x_train = spy_train_data[:,0:2]
spy_Y_train = spy_train_data[:,2]
spy_x_test = spy_test_data[:,0:2]
spy_Y_test = spy_test_data[:,2]

amzn_train_data,amzn_test_data = train_test_split(amzn_data,test_size=0.2,random_state=42)
amzn_x_train = amzn_train_data[:,0:2]
amzn_Y_train = amzn_train_data[:,2]
amzn_x_test = amzn_test_data[:,0:2]
amzn_Y_test = amzn_test_data[:,2]

aapl_nsde_path = model_path + "ex4_aapl_nsde.pth"
aapl_nsde_pro_path = model_path + "ex4_aapl_nsde_pro.pth"
amzn_nsde_path = model_path + "ex4_amzn_nsde.pth"
amzn_nsde_pro_path = model_path + "ex4_amzn_nsde_pro.pth"
spy_nsde_path = model_path + "ex4_spy_nsde.pth"
spy_nsde_pro_path = model_path + "ex4_spy_nsde_pro.pth"

losses_val_aapl_nsde_path = loss_path+'ex4_nsde_aapl_losses_val.npy'
losses_val_aapl_pro_path = loss_path+'ex4_nsde_aapl_pro_losses_val.npy'
losses_val_spy_nsde_path = loss_path+'ex4_nsde_spy_losses_val.npy'
losses_val_spy_pro_path = loss_path+'ex4_nsde_spy_pro_losses_val.npy'
losses_val_amzn_nsde_path = loss_path+'ex4_nsde_amzn_losses_val.npy'
losses_val_amzn_pro_path = loss_path+'ex4_nsde_amzn_pro_losses_val.npy'
losses_val_spy_nsde =[1000]
losses_val_spy_pro = [1000]
losses_val_aapl_nsde =[1000]
losses_val_aapl_pro = [1000]
losses_val_amzn_nsde =[1000]
losses_val_amzn_pro = [1000]

aapl_S0 = 125.28
amzn_S0 = 3230.11
spy_S0 = 419.29
V0 = 0.03
rate = 0.06

aapl_asset_info = [aapl_S0,V0, rate]
spy_asset_info = [spy_S0,V0,rate]
amzn_asset_info = [amzn_S0,V0,rate]


if(not os.path.exists(aapl_nsde_path) or not os.path.exists(losses_val_aapl_nsde_path)):
    model = Net_SDE_Revised(aapl_asset_info, 4,2,20,1000,device)
    print("==="*10+"training the neural sde model"+"==="*10)
    losses_val_nsde = train_models(model,torch.tensor(aapl_Y_train,dtype=torch.float32).view(len(aapl_Y_train),1),
                aapl_x_train,aapl_nsde_path,losses_val_aapl_nsde,400,0.01)
    np.save(loss_path+'ex3_nsde_losses_val.npy', losses_val_nsde) 

if(not os.path.exists(aapl_nsde_pro_path) or not os.path.exists(losses_val_aapl_pro_path)):
    # heston calibration
    asset_input = aapl_x_train.copy()
    asset_input[:,0] = asset_input[:,0]/360

    def fun(x,asset_input,y):
        return heston(aapl_S0,V0,rate,x,asset_input) - y

    x0 = np.array([-0.3, 0.03, 1.3, 0.3])

    res_lsq = res_lsq = least_squares(fun, x0, 
                    bounds=([-1, -np.inf,-np.inf,-np.inf], [1,np.inf, np.inf, np.inf]),
                    args=(asset_input, aapl_Y_train.ravel()))
    heston_info = res_lsq.x

    print( "*"*20,"The calibrated heston model params: \
            rho = {},theta = {}, kappa = {} and lambda = {}".format(heston_info[0],heston_info[1],\
            heston_info[2],heston_info[3]),"*"*20)


    model_pro = Net_SDE_Revised_Pro(heston_info,aapl_asset_info, 4,2,20,1000,device)

    for layer in model_pro.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.constant_(layer.weight, val=0.0)

    print("==="*10+"training the neural sde pro model"+"==="*10)
    losses_val_appl_pro = train_models(model_pro,torch.tensor(aapl_Y_train,dtype=torch.float32).view(len(aapl_Y_train),1),
                aapl_x_train,aapl_nsde_pro_path,losses_val_aapl_pro,400,0.01)

    # save loss
    np.save(losses_val_aapl_pro_path, losses_val_aapl_pro) 


if(not os.path.exists(amzn_nsde_path) or not os.path.exists(losses_val_amzn_nsde_path)):
    model = Net_SDE_Revised(amzn_asset_info, 4,2,20,1000,device)
    print("==="*10+"training the neural sde model"+"==="*10)
    losses_val_nsde = train_models(model,torch.tensor(amzn_Y_train,dtype=torch.float32).view(len(amzn_Y_train),1),
                amzn_x_train,amzn_nsde_path,losses_val_amzn_nsde,400,0.01)
    np.save(loss_path+'ex3_nsde_losses_val.npy', losses_val_nsde) 

if(not os.path.exists(amzn_nsde_pro_path) or not os.path.exists(losses_val_amzn_pro_path)):
    # heston calibration
    asset_input = amzn_x_train.copy()
    asset_input[:,0] = asset_input[:,0]/360

    def fun(x,asset_input,y):
        return heston(amzn_S0,V0,rate,x,asset_input) - y

    x0 = np.array([-0.3, 0.03, 1.3, 0.3])

    res_lsq = res_lsq = least_squares(fun, x0, 
                    bounds=([-1, -np.inf,-np.inf,-np.inf], [1,np.inf, np.inf, np.inf]),
                    args=(asset_input, amzn_Y_train.ravel()))
    heston_info = res_lsq.x

    print( "*"*20,"The calibrated heston model params: \
            rho = {},theta = {}, kappa = {} and lambda = {}".format(heston_info[0],heston_info[1],\
            heston_info[2],heston_info[3]),"*"*20)


    model_pro = Net_SDE_Revised_Pro(heston_info,amzn_asset_info, 4,2,20,1000,device)

    for layer in model_pro.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.constant_(layer.weight, val=0.0)

    print("==="*10+"training the neural sde pro model"+"==="*10)
    losses_val_appl_pro = train_models(model_pro,torch.tensor(amzn_Y_train,dtype=torch.float32).view(len(amzn_Y_train),1),
                amzn_x_train,amzn_nsde_pro_path,losses_val_amzn_pro,400,0.01)

    # save loss
    np.save(losses_val_amzn_pro_path, losses_val_amzn_pro) 


if(not os.path.exists(spy_nsde_path) or not os.path.exists(losses_val_spy_nsde_path)):
    model = Net_SDE_Revised(spy_asset_info, 4,2,20,1000,device)
    print("==="*10+"training the neural sde model"+"==="*10)
    losses_val_nsde = train_models(model,torch.tensor(spy_Y_train,dtype=torch.float32).view(len(spy_Y_train),1),
                spy_x_train,spy_nsde_path,losses_val_spy_nsde,400,0.01)
    np.save(loss_path+'ex3_nsde_losses_val.npy', losses_val_nsde) 

if(not os.path.exists(spy_nsde_pro_path) or not os.path.exists(losses_val_spy_pro_path)):
    # heston calibration
    asset_input = spy_x_train.copy()
    asset_input[:,0] = asset_input[:,0]/360

    def fun(x,asset_input,y):
        return heston(spy_S0,V0,rate,x,asset_input) - y

    x0 = np.array([-0.3, 0.03, 1.3, 0.3])

    res_lsq = res_lsq = least_squares(fun, x0, 
                    bounds=([-1, -np.inf,-np.inf,-np.inf], [1,np.inf, np.inf, np.inf]),
                    args=(asset_input, spy_Y_train.ravel()))
    heston_info = res_lsq.x

    print( "*"*20,"The calibrated heston model params: \
            rho = {},theta = {}, kappa = {} and lambda = {}".format(heston_info[0],heston_info[1],\
            heston_info[2],heston_info[3]),"*"*20)


    model_pro = Net_SDE_Revised_Pro(heston_info,spy_asset_info, 4,2,20,1000,device)

    for layer in model_pro.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.constant_(layer.weight, val=0.0)

    print("==="*10+"training the neural sde pro model"+"==="*10)
    losses_val_appl_pro = train_models(model_pro,torch.tensor(spy_Y_train,dtype=torch.float32).view(len(spy_Y_train),1),
                spy_x_train,spy_nsde_pro_path,losses_val_spy_pro,400,0.01)

    # save loss
    np.save(losses_val_spy_pro_path, losses_val_spy_pro) 

model_aapl = torch.load(aapl_nsde_path)
model_aapl_pro = torch.load(aapl_nsde_pro_path)
model_spy = torch.load(spy_nsde_path)
model_spy_pro = torch.load(spy_nsde_pro_path)
model_amzn = torch.load(amzn_nsde_path)
model_amzn_pro = torch.load(amzn_nsde_pro_path)

pred_aapl = model_aapl(aapl_x_train).detach()
pred_pro_aapl = model_aapl_pro(aapl_x_train).detach()
pred_amzn = model_amzn(amzn_x_train).detach()
pred_pro_amzn = model_amzn_pro(amzn_x_train).detach()
pred_spy = model_spy(spy_x_train).detach()
pred_pro_spy = model_spy_pro(spy_x_train).detach()

loss_fn = nn.L1Loss() 
print("The train loss for nsde aapl:    ",loss_fn(pred_aapl,torch.tensor(aapl_Y_train,dtype=torch.float32).view(len(aapl_Y_train),1)))
print("The train loss for nsde pro aapl:    ",loss_fn(pred_pro_aapl,torch.tensor(aapl_Y_train,dtype=torch.float32).view(len(aapl_Y_train),1)))
print("The train loss for nsde amzn:    ",loss_fn(pred_amzn,torch.tensor(amzn_Y_train,dtype=torch.float32).view(len(amzn_Y_train),1)))
print("The train loss for nsde pro amzn:    ",loss_fn(pred_pro_amzn,torch.tensor(amzn_Y_train,dtype=torch.float32).view(len(amzn_Y_train),1)))
print("The train loss for nsde spy:    ",loss_fn(pred_spy,torch.tensor(spy_Y_train,dtype=torch.float32).view(len(spy_Y_train),1)))
print("The train loss for nsde pro spy:    ",loss_fn(pred_pro_spy,torch.tensor(spy_Y_train,dtype=torch.float32).view(len(spy_Y_train),1)))

pred_test_aapl = model_aapl(aapl_x_test).detach()
pred_pro_test_aapl = model_aapl_pro(aapl_x_test).detach()

aapl_nsde_test_loss = loss_fn(pred_test_aapl,torch.tensor(aapl_Y_test,dtype=torch.float32).view(len(aapl_Y_test),1))
aapl_nsde_pro_test_loss = loss_fn(pred_pro_test_aapl,torch.tensor(aapl_Y_test,dtype=torch.float32).view(len(aapl_Y_test),1))
print("The test loss for nsde appl:  ",aapl_nsde_test_loss)
print("The test loss for nsde pro aapl:  ",aapl_nsde_pro_test_loss)

pred_test_amzn = model_amzn(amzn_x_test).detach()
pred_pro_test_amzn = model_amzn_pro(amzn_x_test).detach()

amzn_nsde_test_loss = loss_fn(pred_test_amzn,torch.tensor(amzn_Y_test,dtype=torch.float32).view(len(amzn_Y_test),1))
amzn_nsde_pro_test_loss = loss_fn(pred_pro_test_amzn,torch.tensor(amzn_Y_test,dtype=torch.float32).view(len(amzn_Y_test),1))
print("The test loss for nsde appl:  ",amzn_nsde_test_loss)
print("The test loss for nsde pro amzn:  ",amzn_nsde_pro_test_loss)

pred_test_spy = model_spy(spy_x_test).detach()
pred_pro_test_spy = model_spy_pro(spy_x_test).detach()


spy_nsde_test_loss = loss_fn(pred_test_spy,torch.tensor(spy_Y_test,dtype=torch.float32).view(len(spy_Y_test),1))
spy_nsde_pro_test_loss = loss_fn(pred_pro_test_spy,torch.tensor(spy_Y_test,dtype=torch.float32).view(len(spy_Y_test),1))
print("The test loss for nsde appl:  ",spy_nsde_test_loss)
print("The test loss for nsde pro spy:  ",spy_nsde_pro_test_loss)