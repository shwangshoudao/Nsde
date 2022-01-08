##衡量实验4的几个模型之间的权重矩阵的相似性, 主要是对比spy和两个科技股之间的相似性差
import torch
from torch import nn
import os
import setenv
from utils import model_difference
    

path = os.path.abspath(os.path.dirname(__file__))
model_path = path + "/result_model/"

aapl_nsde_path = model_path + "ex4_aapl_nsde.pth"
aapl_nsde_pro_path = model_path + "ex4_aapl_nsde_pro.pth"
amzn_nsde_path = model_path + "ex4_amzn_nsde.pth"
amzn_nsde_pro_path = model_path + "ex4_amzn_nsde_pro.pth"
spy_nsde_path = model_path + "ex4_spy_nsde.pth"
spy_nsde_pro_path = model_path + "ex4_spy_nsde_pro.pth"

model_aapl = torch.load(aapl_nsde_path)
model_aapl_pro = torch.load(aapl_nsde_pro_path)
model_spy = torch.load(spy_nsde_path)
model_spy_pro = torch.load(spy_nsde_pro_path)
model_amzn = torch.load(amzn_nsde_path)
model_amzn_pro = torch.load(amzn_nsde_pro_path)


aapl_vs_spy = model_difference(model_aapl,model_spy)
print("The difference between aapl model and spy model is {:.2f}".format(aapl_vs_spy))

amzn_vs_spy = model_difference(model_amzn,model_spy)
print("The difference between amzn model and spy model is {:.2f}".format(amzn_vs_spy))

aapl_vs_amzn = model_difference(model_aapl,model_amzn)
print("The difference between aapl model and amzn model is {:.2f}".format(aapl_vs_amzn))

aapl_vs_spy_pro = model_difference(model_aapl_pro,model_spy_pro)
print("The difference between aapl model pro and spy model pro is {:.2f}".format(aapl_vs_spy_pro))

amzn_vs_spy_pro = model_difference(model_amzn_pro,model_spy_pro)
print("The difference between amzn model pro and spy model pro is {:.2f}".format(amzn_vs_spy_pro))

aapl_vs_amzn_pro = model_difference(model_aapl_pro,model_amzn_pro)
print("The difference between aapl model pro and amzn model pro is {:.2f}".format(aapl_vs_amzn_pro))