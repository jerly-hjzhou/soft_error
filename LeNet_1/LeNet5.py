import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

class LeNet5(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # padding=2保证输入输出尺寸相同
            nn.ReLU(inplace=True),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Linear(84, num_classes)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x  # F.softmax(x, dim=1)

def test():
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(target)  # 计算前要把变量变成Variable形式，因为这样子才有梯度

        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss 把所有loss值进行累加
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加

    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print("{:.4f}".format(100. * correct / len(test_loader.dataset)), file=acc)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 启用GPU
    print("device is ", device)
    test_loader = torch.utils.data.DataLoader(  # 加载训练数据，详细用法参考我的Pytorch打怪路（一）系列-（1）
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
        ])),
        batch_size=100, shuffle=True)

    model = LeNet5()  # 实例化一个网络对象torch.utils.data.DataLoader
    model = model.to(device)


