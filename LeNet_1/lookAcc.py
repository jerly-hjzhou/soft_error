import torch
import copy
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import random
from faultInjection import bitFlip
from LeNet5 import LeNet5

def test():
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(target)  # 计算前要把变量变成Variable形式，因为这样子才有梯度

        output = model(data)
        prob = F.softmax(output, dim=1)
        prob = Variable(prob)
        prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式

        test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss 把所有loss值进行累加
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if target != pred:
            print("target is ", target)
            print(prob)  # prob是10个分类的概率
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加

    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print("{:.4f}".format(100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 启用GPU
    print("device is ", device)
    model = torch.load('LeNet5.pth')  # 加载模型
    test_loader = torch.utils.data.DataLoader(  # 加载训练数据，详细用法参考我的Pytorch打怪路（一）系列-（1）
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
        ])),
        batch_size=1, shuffle=True)

    model = model.to(device)
    test()

