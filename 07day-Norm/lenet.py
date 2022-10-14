"""
@DATE: 2022/5/7
@Author  : ld
"""
import torch
from torch import nn, optim
from resnet import ResNet

from data import send_to_gpu, get_cifar_train


class Lenet5(nn.Module):

    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x:[b, 3, 32, 32] => [b, 6, 30, 30]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            # 长宽变成一半
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc_unit = nn.Sequential(
            # x: [b, 16, 5, 5]
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )


    def forward(self, x):
        batchsz = x.size(0)
        # x: [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)
        x = x.view(batchsz, 16*5*5)
        # x:[16*5*5] => [10]
        logits = self.fc_unit(x)

        return logits



def main():
    cifar_train, cifar_test = get_cifar_train()
    x, label = iter(cifar_train).next()
    print("x: ", x.shape, "label: ", label.shape)

    device = torch.device("cuda")
    # model = Lenet5().to(device)
    model = ResNet().to(device)
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    cifar_train, cifar_test = send_to_gpu(cifar_train, device), send_to_gpu(cifar_test, device)


    for epoch in range(1000):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # x, label = x.to(device), label.to(device)
            logits = model(x)

            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, loss.item())

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # x, label = x.to(device), label.to(device)
                logits = model(x)

                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print("rate: ", acc)


# 老子的 MX150 算不动
if __name__ == "__main__":
    main()
