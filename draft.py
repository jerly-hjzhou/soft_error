#coding=gbk
from torchvision import models
import time
from faultInjection import model_structure

if __name__ == '__main__':
    model_ft = models.googlenet(pretrained=True)
    print(model_structure(model_ft))
