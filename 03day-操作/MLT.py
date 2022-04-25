"""
感知机
"""
import torch
from torch.nn import functional as F

"""
# 单层感知机
- ∂E/∂w = (O-t)O(1-O)X
"""
x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad=True)
print("x: ", x, "w: ", w)

o = torch.sigmoid(x@w.t())
print("o: ", o)
print("o.shape", o.shape)  # [1, 1]

loss = F.mse_loss(torch.ones(1, 1), o)
print("loss: ", loss)
print("loss.shape: ", loss.shape)

loss.backward()
print(w.grad)

print("-"*50)
"""
# 多输出感知机
"""
x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)
print("x: ", x)
print("w: ", w)

o = torch.sigmoid(x@w.t())
print("o: ", o)
print("o.shape: ", o.shape)

loss = F.mse_loss(torch.ones(1, 2), o)
print("loss: ", loss)

loss.backward()
print(w.grad)



