import torch
from torchvision import datasets, transforms
import xlwt, xlrd, xlutils.copy
from faultInjection import evaluateTestSet, bitFlip
import time
from LeNet5 import LeNet5

def testReability(faultProb):
    model = torch.load('LeNet5.pth')  # 加载模型
    model.to(device)
    styleBlueBkg = xlwt.easyxf('pattern: pattern solid, fore_colour yellow;')
    mark = 1
    beginLayer = 1
    with torch.no_grad():
        for layerName, layerPara in model.named_parameters():
            if mark < beginLayer:
                mark = mark+1
                continue
            since = time.time()
            f = open('D:\\practice\\project\\NN_test\\reability_test\\LeNet\\result\\time.txt', 'a')
            readbook = xlrd.open_workbook(filename, formatting_info=True)
            workbook = xlutils.copy.copy(readbook)
            worksheet = workbook.add_sheet(layerName)
            weightShape = layerPara.shape
            weightDim = layerPara.dim()
            try:
                if (weightDim != 1 and weightDim != 4 and weightDim != 2):
                    raise ValueError("the dimension of weight is {} instead of 1 or 4 or 2".format(weightDim))
                else:
                    if weightDim == 1:
                        for k0 in range(weightShape[0]):
                            if layerPara[k0] >= 1 and layerPara[k0] < 2 \
                                    or layerPara[k0] <= -1 and layerPara[k0] > -2:
                                layerPara[k0] = 10e30
                            else:
                                layerPara[k0] = bitFlip(layerPara[k0], 1)
                            print(" * current layer is ", layerName,
                                  " * index is [{}]".format(k0),
                                  " * accuracy is ", evaluateTestSet(model, test_loader, device))
                            worksheet.write(k0, 0, "[%d]" % (k0), styleBlueBkg)
                            worksheet.write(k0, 1, "%f " % (evaluateTestSet(model, test_loader, device)))
                            layerPara[k0] = bitFlip(layerPara[k0], 1)
                    if weightDim == 4:
                        for k0 in range(weightShape[0]):
                            index = 0
                            for k1 in range(weightShape[1]):
                                for k2 in range(weightShape[2]):
                                    for k3 in range(weightShape[3]):
                                        if layerPara[k0][k1][k2][k3] >= 1 and layerPara[k0][k1][k2][k3] < 2 \
                                                or layerPara[k0][k1][k2][k3] <= -1 and layerPara[k0][k1][k2][k3] > -2:
                                            layerPara[k0][k1][k2][k3] = 10e30
                                        else:
                                            layerPara[k0][k1][k2][k3] = bitFlip(layerPara[k0][k1][k2][k3], 1)
                                        print(" * current layer is ", layerName,
                                              " * index is [{}, {}, {}, {}]".format(k0, k1, k2, k3),
                                              " * accuracy is ", evaluateTestSet(model, test_loader, device))
                                        worksheet.write(index, 2*k0,
                                                        "[%d, %d, %d, %d]" % (k0, k1, k2, k3), styleBlueBkg)
                                        worksheet.write(index, 2*k0+1,
                                                        "%f " % (evaluateTestSet(model, test_loader, device)))
                                        layerPara[k0][k1][k2][k3] = bitFlip(layerPara[k0][k1][k2][k3], 1)
                                        index = index + 1
                    if weightDim == 2:
                        for k0 in range(weightShape[0]):
                            index = 0
                            for k1 in range(weightShape[1]):
                                if layerPara[k0][k1] >= 1 and layerPara[k0][k1] < 2 \
                                        or layerPara[k0][k1] <= -1 and layerPara[k0][k1] > -2:
                                    layerPara[k0][k1] = 10e30
                                else:
                                    layerPara[k0][k1] = bitFlip(layerPara[k0][k1], 1)
                                print(" * current layer is ",layerName,
                                      " * index is [{}, {}]".format(k0, k1),
                                      " * accuracy is ", evaluateTestSet(model, test_loader, device))
                                worksheet.write(index, 2*k0, "[%d, %d]" % (k0, k1), styleBlueBkg)
                                worksheet.write(index, 2*k0+1, "%f " % (evaluateTestSet(model, test_loader, device)))
                                layerPara[k0][k1] = bitFlip(layerPara[k0][k1], 1)
                                index = index + 1
                time_elapsed = time.time() - since
                print('{} complete in {:.0f}h {:.0f}m {:.0f}s'.format(layerName,
                    time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60), file=f)
                f.close()
                workbook.save(filename)
            except ValueError as e:
                print(repr(e))

if __name__ == '__main__':
    filename = "D:\\practice\\project\\NN_test\\reability_test\\LeNet\\result\\everyLayer.xls"
    batchSize=250
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device is ", device)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_datasets = datasets.MNIST(root='../data', train=False,
                                    download=True, transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(image_datasets, batch_size=2,
                                              shuffle=True, num_workers=0, prefetch_factor=2)
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet('sheet0')
    workbook.save(filename)
    testReability(1)