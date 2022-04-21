"""
# view == reshape
"""
import torch

a = torch.rand(4, 1, 28, 28)
print(a.shape)

print(a.view(4, 28*28).shape)  # 适用于全连接层

b = a.view(4*28, 28)
print(b.shape)

"""
# unsqueeze
- unsqueeze插入一个维度:
    - 正参数 在相应维度之后 插入
    - 负参数 在相位维度之前 插入
# squeeze
- squeeze 维度删减 只能挤压为 1 的维度

"""
b = a.unsqueeze(0)
print("b:", b.shape)

c = b.squeeze()
print("c", c.shape)

"""
# expand
"""
a = torch.rand(1, 32, 1, 1)
print("a", a.shape)

b = a.expand(4, 32, 14, 14)  # 只有为 1 的那个维度才可以扩张
print("b: ", b.shape)

"""
# repeat
"""
c = a.repeat(4, 32, 1, 1)  # 在该维度上的复制次数
print("c: ", c.shape)

"""
# 矩阵转置 t()
"""
d = torch.randn(3, 4)
print("d: ", d.shape)
print("d 转置: ", d.t().shape)

"""
# transpose 维度交换
"""
a = torch.rand(4, 3, 28, 28)
e = a.transpose(1, 3)  # 维度1和3交换 [4, 28, 28, 3]
print("e", e.shape)

"""
# permute 任意维度交换
"""
f = a.permute(3, 2, 0, 1)  # [28, 28, 4, 3]
print(f.shape)


