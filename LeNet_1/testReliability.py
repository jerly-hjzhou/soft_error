import torch
# import copy
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
import random
from faultInjection import bitFlip
from LeNet5 import LeNet5

experiment = open("result//LeNet5.txt", 'w+')
correctFile = open("result//correctSet.txt", 'w+')

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


    # get the correct image set under the model before injected error
    correctSet = []
    correctNumberBefore = np.zeros(10, dtype=int)
    for image, label in test_loader:
        imageTemp = image.to(device)
        labelTemp = label.to(device)
        imageTemp = Variable(imageTemp)
        output = model(imageTemp)
        pred = output.data.max(1, keepdim=True)[1]
        if pred.eq(labelTemp):
            correctNumberBefore[int(labelTemp)] += 1
            correctSet.append([image, label])

    for i in range(10):
        print("{}: {}".format(i, correctNumberBefore[i]), file=correctFile)
    correctFile.close()

    faultProbability = 1e-5
    print("fault rate is {}".format(faultProbability),file=experiment)
    correctNumberAfter = np.zeros(10, dtype=int)
    correctSample = 0
    totalSample = 0
    for loop in range(10):
        for image, label in correctSet:
            image = image.to(device)
            label = label.to(device)
            image = Variable(image)
            layerName = list(model.state_dict().keys())
            # loop times is the number of layers
            for i in range(len(layerName)):
                layerWeight = model.state_dict()[layerName[i]].clone()

                weightIndex = model.state_dict()[layerName[i]].shape
                # the number of for-loop in the below
                forNumber = len(weightIndex)

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

            pred = output.data.max(1, keepdim=True)[1]
            totalSample += 1
            if pred.eq(label):
                correctSample += 1
            print("loop is {}, reliability rate is {}, correctSample is {}, totalSample is {}".format
                  (loop, correctSample/totalSample, correctSample, totalSample), file=experiment)
            print("loop is {}, reliability rate is {}, correctSample is {}, totalSample is {}".format
                  (loop, correctSample/totalSample, correctSample, totalSample))
            model = torch.load('LeNet5.pth')  # 加载模型
    experiment.close()