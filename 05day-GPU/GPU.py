"""
@DATE: 2022/4/27
@Author  : ld
"""
import torch
from torch import nn
from torch import optim

from data.data import train_loader, test_loader, send_to_gpu

batch_size = 200
learning_rate = 0.01
epochs = 10


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200), nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200), nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10), nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda:0")
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

# Q: 如何提前将数据转移到 GPU ?
train_d = send_to_gpu(train_loader, device)

for step in range(10):  # 不跑3遍,准确度低, 到5准确度可以提高1个百分点
    for batch_idx, (data, target) in enumerate(train_d):
        data = data.view(-1, 28 * 28)
        # data, target = data.to(device), target.to(device)  # 每次转移数据 太消耗性能
        logits = net.forward(data)  # _call_impl
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(step, batch_idx, loss.item())

test_loss = 0
correct = 0
num = len(test_loader.dataset)
test_d = send_to_gpu(test_loader, device)

for data, target in test_d:
    data = data.view(-1, 28 * 28)
    # data, target = data.to(device), target.to(device)
    logits = net(data)  # net.forward(data)
    test_loss += criteon(logits, target).item()

    pred = logits.argmax(dim=1)
    correct += pred.eq(target).float().sum().item()
    test_loss /= num

    print(f"Test set: average loss: {round(test_loss, 4)},Accuracy: {correct}/{num} ({round(100 * correct / num, 4)}%)")
