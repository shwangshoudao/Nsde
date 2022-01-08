from pandas.core.algorithms import mode
import torch
from torch import nn




def model_difference(model1, model2, method = "L1"):
    """用于衡量两个模型间的差别

    Args:
        model1 (torch module): 第一个模型
        model2 (torch module): 第二个模型
        method (str, optional): 用于衡量矩阵差距的范数. Defaults to "L1", 可选"L2"、"infinity".

    Returns:
        float: the summation of error for all weight matrix in the model
    """
    error = 0
    for layer1,layer2 in zip(model1.modules(),model2.modules()):
        if isinstance(layer1, torch.nn.Linear) and isinstance(layer2, torch.nn.Linear):
            if method == "L2":
                error = torch.linalg.norm(layer1.weight.data-layer2.weight.data)
            elif method == "infinity":
                error += torch.linalg.norm(layer1.weight.data-layer2.weight.data,float("inf"))
            else:
                error += torch.linalg.norm(layer1.weight.data-layer2.weight.data,1)
    return float(error)

if __name__ == "__main__":
    class Model(torch.nn.Module):
            # 初始化
        def __init__(self):
            super(Model, self).__init__()
            # super 用来返回Model的父类，在pytorch下定义的类都是继承一个大的父类torch.nn.Module的父类。
            # torch.nn.Module中包含了各种工具，一般我们都是写的都是子类，通过父类我们可以很容易书写子类。
            self.linear = torch.nn.Linear(1, 1, bias=False)
            # 建立一个linear类，bias表示偏置项,建立一个AX+b

            # forward 是torch.nn.Module定义好的模板，表示前向传播

        def forward(self, x):
            y_pred = self.linear(x)
            return y_pred

    model1  = Model()
    model2 = Model()
    error = model_difference(model1,model2)
    print(error)
