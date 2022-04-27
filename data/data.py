"""
@DATE: 2022/4/27
@Author  : ld
"""
import torch
import torchvision

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


batch_size = 200

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST("../data", train=True, download=True,
                                                                      transform=torchvision.transforms.Compose([
                                                                          torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize(
                                                                              (0.1307,),(0.3081,)
                                                                          )])),
                                           batch_size= batch_size, shuffle=True)

train_db, val_db = torch.utils.data.random_split(train_loader, [50000, 10000])
train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_db, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST("../data", train=False, download=True,
                                                                      transform=torchvision.transforms.Compose([
                                                                          torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize(
                                                                              (0.1307,),(0.3081,)
                                                                          )])),
                                          batch_size= batch_size, shuffle=False)



