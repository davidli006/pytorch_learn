import numpy as np
import torch

## 默认类型 FloatTensor
# numpy
a = np.array([2, 3.3])
print(a)
a = torch.from_numpy(a)
print(a)

# 数组
b = torch.tensor([1, 2, 3])
print(b)
# c = torch.tensor(2, 3)  # 小写给现成的数据data, 大写的给shape
# print(c)

# 未初始化的数据
a = torch.empty(2, 3)
print(a, a.shape)  # 数据随机,非常不规则,需要覆盖掉

## 随机初始化 rand
a = torch.rand(3, 3)  # 0-1 均匀分布
print("0-1均匀分布: ", a)
b = torch.randint(20, 40, [3, 3])  # 20-40 均匀分布
print("20-40均匀分布", b)
c = torch.randn(3, 3)  # 正太分布
print("0-1正太分布", c)
d = torch.normal(mean=torch.full([10], 0.), std=torch.arange(1, 0, -0.1))  # 0后面要加点
print("d", d)

## full
a = torch.full([2, 3], 7.)
print("full 7", a)


