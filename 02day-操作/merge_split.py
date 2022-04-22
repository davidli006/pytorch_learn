"""
# cat
- 两个 tensor 维度要一样(3维)
- 除了要合并的维度,其他维度上数量要保持一致
"""
import torch

# [班级, 课程, 分数]
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)

c = torch.cat([a, b], dim=0)  # 在第一个维度上进行合并(班级合并)
print("cat: ", c.shape)  # 把所有人的成绩合在一个班级看

print("-"*50)
"""
# stack
- 新增一个维度 
"""

a = torch.rand(32, 8)
b = torch.rand(32, 8)

c = torch.stack([a, b], dim=0)
print("stack c: ", c.shape)  # 把两个班级的成绩, 再放进一个更大容器里

print("-"*50)
"""
# split
- 根据长度
"""
aa,bb, cc, dd = c.split([16,8,5,3], dim=1)  # 对维度 1 分成相应比例 32分成 16:8:5:3
print("split aa", aa.shape)
print("split bb", bb.shape)
print("split cc", cc.shape)
print("split dd", dd.shape)

print("-"*50)
"""
# chunk
- 分成指定块数
"""
aa,bb, cc,dd = c.chunk(4, dim=1)  # 32/4 每份8个
print("chunk aa", aa.shape)
print("chunk bb", bb.shape)
print("chunk cc", cc.shape)
print("chunk dd", dd.shape)

print("-"*50)
"""
# basic
- 加 +
- 减 -
- 乘 *
- 除 /
"""
print((aa + bb).shape)

print("-"*50)
"""
# matmul
- 矩阵相乘
- torch.mm: 只能二维
- torch.matmul
- @
- 都只会使用后面的2维来进行计算
"""
a = torch.Tensor([[3, 3],[3, 3]])
b = torch.ones(2, 2)
m = a@b
mm = torch.matmul(a, b)
print("matmul", m.shape, mm.shape)
print("m & mm is eq:", torch.all(torch.eq(m, mm)))

print("-"*50)
"""
# example
"""
a = torch.rand(4, 784)
x = torch.rand(4, 784)
w = torch.rand(512, 784)  # torch 习惯将 输出的维度放在前面
print((x@w.t()).shape)

print("-"*50)
"""
# 次方
- power
- sqrt
"""
a = torch.full([2, 2], 3)
print(a)
p = a.pow(2)
print(p)
print(p.sqrt())  # 开平方

print("-"*50)
"""
- floor() ceil()
- round()
- trunc() frac()
- clamp(10) 所有小于 10 的变成10
"""


