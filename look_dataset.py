from torchvision import datasets, transforms
import numpy as np

#计算某个样本出现的次数
def count_number(train_dataset, label):
    number=0 #该样本出现的次数
    for i in range(len(train_dataset.targets)):
        if(train_dataset.targets[i] == label):
            number=number+1
    return number

train_dataset = datasets.MNIST('./data', train=True,
                               transform=transforms.Compose([transforms.ToTensor()]),
                               download=True)
print("the number of training dataset={}".format(len(train_dataset.targets)))
for i in range(10):
    print("the number of {}={}".format(i, count_number(train_dataset, i)))



