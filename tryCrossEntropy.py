import torch.nn.functional as F
import torch

# input = torch.tensor([[0.0, 0.0, 1.0]])
# target = torch.tensor([2])
# print(cross_entropy(output, target))
input = torch.tensor([[ 2.1392e+32,  3.1363e+32, 1.1646e+32]])
prob = F.softmax(input, dim=1)
print(prob)
target = torch.tensor([0])
print(input)
print(target)
loss = F.cross_entropy(input, target, reduction='sum').item()
print(loss)
