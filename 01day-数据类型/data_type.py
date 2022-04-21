"""
# 没有string类型

- One - hot
    - dog <> cat
    - dog: [1, 0]
    - cat: [0, 1]
- Embedding
    - 用数字的方法表示语言
"""
import numpy as np
import torch

# 2行3列的数据类型
a = torch.rand(2, 3)
print(type(a))
print(a.type())
print(isinstance(a, torch.FloatTensor))
print(isinstance(a, torch.Tensor))

print("-"*50)
data = torch.DoubleTensor(2, 1)
print(isinstance(data, torch.cuda.DoubleTensor))
data = data.cuda()
print(isinstance(data, torch.cuda.DoubleTensor))

print("-"*50)
# 标量
a = torch.tensor(1.)
print(a, a.shape, a.size())
b = torch.tensor(2.3)
print(b, b.shape, b.size())

print("-"*50)
# 向量
a = torch.tensor([1.1, 2.2])  # 直接生成
print("a: ", a, a.shape)

b = torch.FloatTensor(2, 3)  # 给定行列随机生成
print("b: ", b, b.shape)

c = np.ones(3)  # 使用numpy生成
print("np: ", c)
c = torch.from_numpy(c)
print("c: ", c, c.shape)

print("-"*50)
# dim 维度
a = torch.randn(2, 3, 4)  # 维度为3维
print(a.shape)
print(a.size(0))
print(a.size(1))
print(a.size(2))

"""
# 3维
- 适用于 rnn
# 4维
- 适用于 cnn
- 图片
- [b, c, h, w]
"""
print("数据量: ", a.numel()) # 2*3*4
print("维度: ", a.dim()) #
a = torch.tensor(1)
print("标量维度: ", a.dim())


