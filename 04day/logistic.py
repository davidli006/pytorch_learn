import torch
import torchvision
from torch import optim, nn
from torch.nn import functional as F


def surprise(a):
    return -(a*torch.log2(a)).sum()

a = torch.full([4], 1/4.)
print("a", surprise(a))

b = torch.tensor([0.1, 0.1, 0.1, 0.7])
print("b", surprise(b))

c = torch.tensor(([0.001, 0.001, 0.001, 0.997]))
print("c", surprise(c))

print("-"*50)
"""
- cross_entropy = softmax + log + nll_loss
"""

x = torch.randn(1, 784)
w = torch.randn(10, 784)

logits = x @ w.t()
print("logits: ", logits)

pred = F.softmax(logits, dim=1)
print("pred: ", pred)

pred_log = torch.log(pred)

res = F.cross_entropy(logits, torch.tensor([3]))
print("res: ", res)

pre_res = F.nll_loss(pred_log, torch.tensor([3]))
print("pre_res: ", pre_res)

print("-"*50)
"""
# question
"""
batch_size = 200
learning_rate = 0.01
epochs = 10

w1, b1 = torch.randn(200, 784, requires_grad=True), torch.zeros(200, requires_grad=True)
w2, b2 = torch.randn(200, 200, requires_grad=True), torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(10, 200, requires_grad=True), torch.zeros(10, requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)

def forward(x):
    x = F.relu(x @ w1.t() + b1)
    x = F.relu(x @ w2.t() + b2)
    x = F.relu(x @ w3.t() + b3)
    return x

optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
criteon = nn.CrossEntropyLoss()



train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST("../data", train=True, download=True,
                                                                      transform=torchvision.transforms.Compose([
                                                                          torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize(
                                                                              (0.1307,),(0.3081,)
                                                                          )])),
                                           batch_size= batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST("../data", train=False, download=True,
                                                                      transform=torchvision.transforms.Compose([
                                                                          torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize(
                                                                              (0.1307,),(0.3081,)
                                                                          )])),
                                          batch_size= batch_size, shuffle=False)

for epoch in range(3):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        logits = forward(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(epoch, batch_idx, loss.item())


