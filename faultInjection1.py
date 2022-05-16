import struct
import torch
import random

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

def modelInjectFault(modelName = '/LeNet/LeNet5.pth' , faultProbability = 1e-6):
    model = torch.load(modelName)  # 加载模型
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
                        layerWeight[k0] = bitFlip(layerWeight[k0], m)
        if forNumber == 2:
            for k0 in range(weightIndex[0]):
                for k1 in range(weightIndex[1]):
                    for m in range(32):
                        if random.random() < faultProbability:
                            layerWeight[k0][k1] = bitFlip(layerWeight[k0][k1], m)
        if forNumber == 4:
            for k0 in range(weightIndex[0]):
                for k1 in range(weightIndex[1]):
                    for k2 in range(weightIndex[2]):
                        for k3 in range(weightIndex[3]):
                            for m in range(32):
                                if random.random() < faultProbability:
                                    layerWeight[k0][k1][k2][k3] = bitFlip(layerWeight[k0][k1][k2][k3], m)


