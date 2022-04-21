import torch

a = torch.rand(4, 3, 28, 28)  # [b, c, h, w]  c通道RGB
print(a.shape)
print(a[0].shape)  # 第一张图片
print(a[0, 0].shape)
print(a[0, 0, 2, 4].shape)  # 标量

# 取前两张图片
print(a[:2].shape)

# index_select
b = a.index_select(2, torch.arange(8))  # 在第2个维度上截取8行
print(b.shape)

# ... 任意多个维度
c = a[:2, 1, ...]  # 取出前两张图片第一个通道
print(c.shape)
