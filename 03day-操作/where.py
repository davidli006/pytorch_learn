"""
# where(condition, x, y)
- shape 都相等
"""
import torch

cond = torch.rand(2, 2)
print(cond)
a = torch.zeros([2, 2])
print(a)
b = torch.ones([2, 2])
print(b)
c = torch.where(cond>0.5, a, b)  # 大于0.5取a的值,小于0.5取b的值
print(c)

print("-"*50)
"""
# gather(input, dim, index, out=None)
- 
"""
prob = torch.randn(4, 10)
idx = prob.topk(dim=1, k=3)
print(idx)
idx = idx[1]

label = torch.arange(10) + 100
res = torch.gather(label.expand(4, 10), dim=1, index=idx.long())
print(res)
