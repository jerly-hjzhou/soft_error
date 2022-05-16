import torch
import cv2
import torch.nn.functional as F
from LeNet5 import LeNet5  ##重要，虽然显示灰色(即在次代码中没用到)，但若没有引入这个模型代码，加载模型时会找不到模型
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device is ", device)
    model = torch.load('LeNet5.pth')  # 加载模型

    # para = open("para.txt", 'w+')
    # print(model.state_dict(), file=para)

    model = model.to(device)
    model.eval()  # 把模型转为test模式

    img = cv2.imread("2_0.png")  # 读取要预测的图片
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图，因为mnist数据集都是灰度图
    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    # 扩展后，为[1，1，28，28]
    output = model(img)
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    print(prob)  # prob是10个分类的概率
    pred = np.argmax(prob)  # 选出概率最大的一个
    print(pred.item())
    uut = "fc2.0.bias"
    weight = model.state_dict()[uut].clone() #浅拷贝，只复制数值
    #for bias
    for i in range(84):
        model.state_dict()[uut][i] = -10000000
        output = model(img)
        prob = F.softmax(output, dim=1)
        print("******" ,prob)  # prob是10个分类的概率
        prob = Variable(prob)
        prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
        print(prob)  # prob是10个分类的概率
        pred = np.argmax(prob)  # 选出概率最大的一个
        print(pred.item())
        model.state_dict()[uut][i] = weight[i]

    # for conv layer
    # for i1 in range(16):
    #     for i2 in range(6):
    #         for i3 in range(5):
    #             for i4 in range(5):
    #                 model.state_dict()[uut][i1][i2][i3][i4] = 100000000
    #                 output = model(img)
    #                 prob = F.softmax(output, dim=1)
    #                 prob = Variable(prob)
    #                 prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    #                 print(prob)  # prob是10个分类的概率
    #                 pred = np.argmax(prob)  # 选出概率最大的一个
    #                 print(pred.item())
    #                 model.state_dict()[uut][i1][i2][i3][i4] = weight[i1][i2][i3][i4]

    # for fc layer
    # for i1 in range(84):
    #     print("========current {} layer==========".format(i1))
    #     for i2 in range(120):
    #         model.state_dict()[uut][i1][i2] = -10000000
    #         output = model(img)
    #         prob = F.softmax(output, dim=1)
    #         prob = Variable(prob)
    #         prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    #         print(prob)  # prob是10个分类的概率
    #         pred = np.argmax(prob)  # 选出概率最大的一个
    #         print(pred.item())
    #         model.state_dict()[uut][i1][i2] = weight[i1][i2]
