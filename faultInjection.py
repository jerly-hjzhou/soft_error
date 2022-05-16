import struct
import random
import torch.nn.functional as F

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

getBin = lambda x: x >= 0 and str(bin(x))[2:] or "-" + str(bin(x))[3:]

def floatToBinary32(value):
    val = struct.unpack('L', struct.pack('f', value))[0]
    return getBin(val)

def binaryToFloat32(value):
    hx = hex(int(value, 2))
    return struct.unpack("f", struct.pack("L", int(hx, 16)))[0]

# flip the index_th(from left to right) bit in parameter
def bitFlip(parameter, index):
    # change the parameter type into float
    para = float(parameter)
    binary_str = floatToBinary32(para)
    # suplement the '0' in the head
    for i in range(32 -  len(binary_str)):
        binary_str = '0' + binary_str
    binary_str = binary_str[:index] + '0' + binary_str[index + 1:] \
        if binary_str[index] == '1' else binary_str[:index] + '1' + binary_str[index + 1:]
    return binaryToFloat32(binary_str)

def evaluateTestSet(model, test_loader, device):
    lossSum = 0.0
    curLoss = 0.0
    accuracy = 0.0
    correctNum = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        lossSum = lossSum + F.cross_entropy(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correctNum += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
    curLoss = lossSum / len(test_loader.dataset)
    accuracy = correctNum / len(test_loader.dataset)
    return accuracy


# def modelInjectFault(modelName = '/LeNet/LeNet5.pth' , faultProbability = 1e-6):
#     model = torch.load(modelName)  # 加载模型
#     layerName = list(model.state_dict().keys())
#     # loop times is the number of layers
#     for i in range(len(layerName)):
#         layerWeight = model.state_dict()[layerName[i]].clone()
#         weightIndex = model.state_dict()[layerName[i]].shape
#         # the number of for-loop in the below
#         forNumber = len(weightIndex)
#
#         if forNumber == 1:
#             for k0 in range(weightIndex[0]):
#                 for m in range(32):
#                     if random.random() < faultProbability:
#                         layerWeight[k0] = bitFlip(layerWeight[k0], m)
#         if forNumber == 2:
#             for k0 in range(weightIndex[0]):
#                 for k1 in range(weightIndex[1]):
#                     for m in range(32):
#                         if random.random() < faultProbability:
#                             layerWeight[k0][k1] = bitFlip(layerWeight[k0][k1], m)
#         if forNumber == 4:
#             for k0 in range(weightIndex[0]):
#                 for k1 in range(weightIndex[1]):
#                     for k2 in range(weightIndex[2]):
#                         for k3 in range(weightIndex[3]):
#                             for m in range(32):
#                                 if random.random() < faultProbability:
#                                     layerWeight[k0][k1][k2][k3] = bitFlip(layerWeight[k0][k1][k2][k3], m)

def modelInjectFault(model , faultProbability):
    error = 0
    for layerPara in model.parameters():
        weightShape = layerPara.shape
        weightDim = layerPara.dim()
        try:
            if (weightDim != 1 and weightDim != 4 and weightDim != 2):
                raise ValueError("the dimension of weight is {} instead of 1 or 4 or 2".format(weightDim))
            else:
                if weightDim == 1:
                    one = 1
                    for k0 in range(weightShape[0]):
                        judge = random.random()
                        if judge < faultProbability:
                            error=error+1
                            print("=============chosen==============")
                            print(layerPara[k0])
                            if layerPara[k0] >= 1 and layerPara[k0] < 2 \
                                    or layerPara[k0] <= -1 and layerPara[k0] > -2:
                                layerPara[k0] = 10e30
                            else:
                                layerPara[k0] = bitFlip(layerPara[k0], 1)
                            print("=========finish=================")
                            print(layerPara[k0])
                if weightDim == 4:
                    for k0 in range(weightShape[0]):
                        for k1 in range(weightShape[1]):
                            for k2 in range(weightShape[2]):
                                for k3 in range(weightShape[3]):
                                    judge = random.random()
                                    if judge < faultProbability:
                                        error = error + 1
                                        print("=============chosen==============")
                                        print(layerPara[k0][k1][k2][k3])
                                        if layerPara[k0][k1][k2][k3] >= 1 and layerPara[k0][k1][k2][k3] < 2 \
                                                or layerPara[k0][k1][k2][k3] <= -1 and layerPara[k0][k1][k2][k3] > -2:
                                            layerPara[k0][k1][k2][k3] = 10e30
                                        else:
                                            layerPara[k0][k1][k2][k3] = bitFlip(layerPara[k0][k1][k2][k3], 1)
                                        print("=========finish=================")
                                        print(layerPara[k0][k1][k2][k3])
                if weightDim == 2:
                    for k0 in range(weightShape[0]):
                        for k1 in range(weightShape[1]):
                            judge = random.random()
                            if judge < faultProbability:
                                error = error + 1
                                print("=============chosen==============")
                                print(layerPara[k0][k1])
                                if layerPara[k0][k1] >= 1 and layerPara[k0][k1] < 2 \
                                        or layerPara[k0][k1] <= -1 and layerPara[k0][k1] > -2:
                                    layerPara[k0][k1] = 10e30
                                else:
                                    layerPara[k0][k1] = bitFlip(layerPara[k0][k1], 1)
                                print("=========finish=================")
                                print(layerPara[k0][k1])
        except ValueError as e:
            print(repr(e))
    print("*** error = ", error)

