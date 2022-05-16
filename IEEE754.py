import torch
from LeNet5 import LeNet5
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
experiment = open("result//fc3_weight2.txt", 'w+')

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

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device is ", device)
    model = torch.load('LeNet5.pth')  # 加载模型
    test_loader = torch.utils.data.DataLoader(  # 加载训练数据
        datasets.MNIST('../data', download=True, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
        ])),
        batch_size=1000, shuffle=True)

    uut = "fc3.weight"


    weight = model.state_dict()[uut].clone()
    print("orignal accuracy: {}".format(test()), file=experiment)
    testTool = [1000000]
    print("**layer is {}**".format(uut), file=experiment)
    print("**layer is {}**".format(uut))

    # for convolutional layer
    # for i in range(2,6):
    #     for j in range(1):
    #         for k in range(5):
    #             for m in range(5):
    #                 print("===index is [{}, {}, {}, {}]===\n"
    #                       "weight\t\t\taccuracy".format(i, j, k, m), file=experiment)
    #                 print("===index is [{}, {}, {}, {}]===\n"
    #                       "weight\t\t\taccuracy".format(i, j, k, m))
    #                 for t in range(len(testTool)):
    #                 # for t in range(-22,10):
    #                     model.state_dict()[uut][i][j][k][m] = testTool[t]
    #                     print("{}\t\t\t{:.2f}".format(testTool[t], test()), file=experiment)
    #                     print("{}\t\t\t{:.2f}".format(testTool[t], test()))
    #                     # model.state_dict()[uut][i][j][k][m] = t
    #                     # print("{}\t\t{:.2f}".format(t, test()), file=experiment)
    #                     # print("{}\t\t{:.2f}".format(t, test()))
    #                 model.state_dict()[uut][i][j][k][m] = weight[i][j][k][m]

    # for bias
    # for i in range(6):
    #     print("===index is [{}]===\n"
    #           "weight\t\t\taccuracy".format(i), file=experiment)
    #     print("===index is [{}]===\n"
    #           "weight\t\t\taccuracy".format(i))
    #     for t in range(len(testTool)):
    #     # for t in range(-22,10):
    #         model.state_dict()[uut][i] = testTool[t]
    #         print("{}\t\t{:.2f}".format(testTool[t], test()), file=experiment)
    #         print("{}\t\t{:.2f}".format(testTool[t], test()))
    #         # model.state_dict()[uut][i][j][k][m] = t
    #         # print("{}\t\t{:.2f}".format(t, test()), file=experiment)
    #         # print("{}\t\t{:.2f}".format(t, test()))
    #     model.state_dict()[uut][i] = weight[i]

    # for fully connected layer
    for i in range(10):
        for j in range(84):
            print("===index is [{}, {}]===\n"
                  "weight\t\t\taccuracy".format(i, j), file=experiment)
            print("===index is [{}, {}]===\n"
                  "weight\t\t\taccuracy".format(i, j))
            for t in range(len(testTool)):
            # for t in range(-22,10):
                model.state_dict()[uut][i][j] = testTool[t]
                print("{}\t\t{:.2f}".format(testTool[t], test()), file=experiment)
                print("{}\t\t{:.2f}".format(testTool[t], test()))
                # model.state_dict()[uut][i][j][k][m] = t
                # print("{}\t\t{:.2f}".format(t, test()), file=experiment)
                # print("{}\t\t{:.2f}".format(t, test()))
            model.state_dict()[uut][i][j] = weight[i][j]