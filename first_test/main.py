import os

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import torchvision

from utils import plot_curve, one_hot, plot_image

batch_size = 512
# load dataset
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

x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
# plot_image(x, y, "image :")


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [b, 1, 28, 28
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
train_res = []

for i in range(3):
    for idx, (x, y) in enumerate(train_loader):
        # [b, 1, 28, 28] => [b, feature]
        x = x.view(x.size(0), 28*28)
        out = net(x)
        # [b, 10]
        y_one = one_hot(y)
        loss = F.mse_loss(out, y_one)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_res.append(loss.item())

        if idx % 10 == 0:
            print(i, idx, loss.item())

# optimal [w1, b1, w2, b2, w3, b3]
plot_curve(train_res)

total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

num = len(test_loader.dataset)
print(total_correct/num)

x,y = next(iter(test_loader))
out = net(x.view(x.size(0), 28*28))
pred = out.argmax(dim=1)
plot_image(x, pred, "test")




