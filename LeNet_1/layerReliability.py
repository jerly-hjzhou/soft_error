import torch
import copy
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import random
from faultInjection import bitFlip
from LeNet5 import LeNet5

# don‘ forget to change
experiment = open("result//layer_10000_1.txt", 'w+')

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device is ", device)
    model = torch.load('LeNet5.pth')  # 加载模型
    test_loader = torch.utils.data.DataLoader(  # 加载训练数据，详细用法参考我的Pytorch打怪路（一）系列-（1）
        datasets.MNIST('../data', download=True, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
        ])),
        batch_size=1)

    faultProbability = 1e-6
    print("fault rate is {}".format(faultProbability),file=experiment)

    loopTime = 10
    layerName = list(model.state_dict().keys())
    relativeOffset = np.zeros(len(layerName))

    for loop in range(loopTime):
        print("loop is {}".format(loop))
        print("loop is {}".format(loop), file = experiment)
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            image = Variable(image)

            # turn label to one-hot format
            x = np.zeros(10)
            x[label] = 1
            # loop times is the number of layers
            for i in range(len(layerName)):
                print("layer is {}".format(layerName[i]))
                layerWeight = model.state_dict()[layerName[i]].clone()
                weightIndex = model.state_dict()[layerName[i]].shape
                # the number of for-loop in the below
                forNumber = len(weightIndex)

                output = model(image)
                prob = F.softmax(output, dim=1)
                prob = Variable(prob)
                prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
                beforeFault = np.linalg.norm(x - prob)

                if forNumber == 1:
                    for k0 in range(weightIndex[0]):
                        for m in range(32):
                            if random.random() < faultProbability:
                                model.state_dict()[layerName[i]][k0] \
                                    = bitFlip(layerWeight[k0], m)

                if forNumber == 2:
                    for k0 in range(weightIndex[0]):
                        for k1 in range(weightIndex[1]):
                            for m in range(32):
                                if random.random() < faultProbability:
                                    model.state_dict()[layerName[i]][k0][k1] \
                                        = bitFlip(layerWeight[k0][k1], m)
                if forNumber == 4:
                    for k0 in range(weightIndex[0]):
                        for k1 in range(weightIndex[1]):
                            for k2 in range(weightIndex[2]):
                                for k3 in range(weightIndex[3]):
                                    for m in range(32):
                                        if random.random() < faultProbability:
                                            model.state_dict()[layerName[i]][k0][k1][k2][k3] \
                                                = bitFlip(layerWeight[k0][k1][k2][k3], m)

                output = model(image)
                prob = F.softmax(output, dim=1)
                prob = Variable(prob)
                prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
                afterFault = np.linalg.norm(x-prob)
                print("after fault is ", afterFault)
                print("before fault is ", beforeFault)
                if not np.isnan(afterFault) and not np.isnan(beforeFault):
                    relativeOffset[i] = relativeOffset[i] + afterFault - beforeFault

                print("relativeOffset is ", relativeOffset)

                sd = model.state_dict()
                sd[layerName[i]] = layerWeight
                model.load_state_dict(sd)

    relativeOffset = relativeOffset / (loop * len(test_loader))
    for i in range(len(layerName)):
        print("layer is: {}".format(layerName[i]), file= experiment)
        print("relativeOffset is {}".format(relativeOffset[i]), file=experiment)
    experiment.close()