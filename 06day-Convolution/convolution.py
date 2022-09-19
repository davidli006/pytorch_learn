"""
@DATE: 2022/4/28
@Author  : ld
"""
import torch
from torch import nn
from torch.nn import functional as F

layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)
x = torch.rand(1, 1, 28, 28)

out = layer(x)
print("padding_0 out: ", out.shape)

layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
out = layer(x)
print("padding_1 out: ", out.shape)

layer = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1)  # 每次移动两格
out = layer(x)
print("padding_1 s_2: ", out.shape)

print("weight: ", layer.weight.shape)
print("weight: ", layer.bias.shape)

print("-"*50)
w = torch.rand(16, 3, 5, 5)
b = torch.rand(16)

x = torch.randn(1, 3, 28, 28)
out = F.conv2d(x, w, stride=1, padding=1)
print("out_1: ", out.shape)

out = F.conv2d(x, w, stride=2, padding=1)
print("out_2: ", out.shape)

print("-"*50)
x = out
layer= nn.MaxPool2d(2, stride=2)
out = layer(x)
print("max_pool: ", out.shape)  # x为13不够移动一次了,就舍掉

layer= nn.AvgPool2d(2, stride=2)
out = layer(x)
print("avg_pool: ", out.shape)  # x为13不够移动一次了,就舍掉

res = F.max_pool2d(x, 2, stride=2)
print("F_max_pool: ", res.shape)

res = F.avg_pool2d(x, 2, stride=2)
print("F_avg_pool: ", res.shape)
print(torch.all(torch.eq(out, res)))
print(1+9+9+8+9+9)
