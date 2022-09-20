"""
@DATE: 2022/9/19
@Author  : ld
"""
import torch
from torch import nn, optim
from torch.nn import functional as F

from data.data import get_loader


input_size = 28
num_classes = 10
num_epochs = 3
batch_size = 64

train_loader, test_loader = get_loader()


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def accuracy(predictions, labels):
    """
    评估函数
    :param predictions:
    :param labels:
    :return:
    """
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(num_epochs):
    train_rights = []

    for batch_idx,(data, target) in enumerate(train_loader):
        net.train()
        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = accuracy(output, target)
        train_rights.append(right)

        if batch_idx % 100 == 0:
            net.eval()
            val_rights = []
            for (data, target) in test_loader:
                output = net(data)
                right = accuracy(output, target)
                val_rights.append(right)

            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))


            print("""当前epoch: {} [{}/{} ({:.2f}%)] \t损失: {:.6f}\t训练R: {:.2f}%\t测试R: {:.2f}%""".format(
                epoch,
                batch_idx * batch_size,
                len(train_loader.dataset),
                100 * batch_idx / len(train_loader),
                loss.data,
                100 * train_r[0].numpy() / train_r[1],
                100 * val_r[0].numpy() / val_r[1]
            ))

