"""
@DATE: 2022/4/27
@Author  : ld
"""
import os.path

import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_file = os.path.dirname(__file__)


def send_to_gpu(loader, device):
    """
    将数据 提前发送的gpu
    :param loader: 数据集
    :param device: gpu
    :return:
    """
    t_d = []
    for data,target in loader:
        t_d.append((data.to(device), target.to(device)))
    return t_d


def get_loader():

    batch_size = 200
    train_loader = DataLoader(torchvision.datasets.MNIST(data_file, train=True, download=True,
                                                                          transform=torchvision.transforms.Compose([
                                                                              torchvision.transforms.ToTensor(),
                                                                              torchvision.transforms.Normalize(
                                                                                  (0.1307,),(0.3081,)
                                                                              )])),
                                               batch_size= batch_size, shuffle=True)

    # train_db, val_db = torch.utils.data.random_split(train_loader, [50000, 10000])
    # train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_db, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(torchvision.datasets.MNIST(data_file, train=False, download=True,
                                                                          transform=torchvision.transforms.Compose([
                                                                              torchvision.transforms.ToTensor(),
                                                                              torchvision.transforms.Normalize(
                                                                                  (0.1307,),(0.3081,)
                                                                              )])),
                                              batch_size= batch_size, shuffle=False)
    return train_loader, test_loader

def get_cifar_train():
    batchsz = 32
    cifar_train = datasets.CIFAR10(data_file, True, transform=transforms.Compose({
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        }), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10(data_file, False, transform=transforms.Compose({
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        }), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)
    return cifar_train, cifar_test

