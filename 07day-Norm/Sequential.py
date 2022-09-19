"""
@DATE: 2022/5/6
@Author  : ld
"""
from torch import nn
from torch import optim


def main():
    net = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 2))
    one = list(net.parameters())[0]
    print(one.shape)

    three = list(net.parameters())[3]
    print(three.shape)

    dic = dict(net.named_parameters())
    for n in dic.items():
        print(n)

    print("-"*50)
    optimizer = optim.SGD(net.parameters(), lr=1e-3)
    print(optimizer)


if __name__ == "__main__":
    main()
