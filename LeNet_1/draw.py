import torch
from LeNet5 import LeNet5

model = torch.load('LeNet5.pth')  # 加载模型
# sd = model.state_dict()
layerWeight = model.state_dict()['conv1.0.bias'].clone()
print("----layerWeight is ",layerWeight)
print("----model is ", model.state_dict()['conv1.0.bias'])

layerWeight[0] = 1
print("++++layerWeight is", layerWeight)
print("++++model is", model.state_dict()['conv1.0.bias'])

print("layerWeight type is ",type(layerWeight))
print("model type is",type(model.state_dict()['conv1.0.bias']))

sd = model.state_dict()

print("****layerWeight is",layerWeight)

sd['conv1.0.bias'] = layerWeight
print("sd is ", sd['conv1.0.bias'])
# model.load_state_dict(sd)
print("****model is", model.state_dict()['conv1.0.bias'])

