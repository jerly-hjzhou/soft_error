import torch
from torchvision import datasets, transforms
import time

model = torch.load('resnet-18/resnet_18-cifar_10.pth')  # 加载模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model.to(device)
model.eval()
accuracy = 0.0
correctNum = 0
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image_datasets = datasets.CIFAR10(root='./data', train=False,
                                  download=True, transform=data_transforms)
test_loader = torch.utils.data.DataLoader(image_datasets, batch_size=250,
                                          shuffle=False, num_workers=0, prefetch_factor=2)

since = time.time()
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correctNum += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
accuracy = correctNum / len(test_loader.dataset)
print("accuracy is {:.4f}%".format(accuracy * 100))
time_elapsed = time.time() - since
print('Test complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))