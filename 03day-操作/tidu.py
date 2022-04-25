"""

"""
import torch
from torch.nn import functional as F

"""
# 梯度
- 所有偏微分组成的向量
- z = y ** 2 - x ** 2
- ∂z/∂y = 2 * y
- ∂z/∂x = - 2 * x
- ▽f = (∂z/∂x1; ∂z/∂x2; ∂z/∂x3;...;∂z/∂xn )
# 影响搜索结果
- 局部最小值
- 鞍点
- 初始状态
# 常见函数梯度
- y = x * w + b  => ▽ = (x, 1)
- y = x * w ** 2 + b ** 2 => ▽ = (2xw, 2b)
"""

"""
# 激活函数
- sigmoid 用来 归一化[0-1]
- tanh 将sigmoid扩大平移 [-1, 1]
- relu x小于是为0, x大于0时为x
"""
a = torch.linspace(-100, 100, 10)
print(a)
s = a.sigmoid()
print("sigmoid: ", s)
a = torch.linspace(-1, 1, 10)
t = a.tanh()
print("tanh: ", t)
r = a.relu()
print("relu: ", r)

"""
# MSE
- 均方差
"""
x = torch.ones(1)
w = torch.full([1], 2).float()
print("x", x, "w", w)
mse = F.mse_loss(torch.ones(1), w*x)
print("mse: ", mse)

# grad = torch.autograd.grad(mse, [w])
# print("grad", grad)

w = w.requires_grad_()  # 更新
mse = F.mse_loss(torch.ones(1), x*w)  # 更新后调用一下才真正更新,蓝图
res = torch.autograd.grad(mse, [w])
print("res: ", res)

"""
# softmax
"""
a = torch.rand(3)
a = a.requires_grad_()
print("a", a)

p = F.softmax(a, dim=0)
# p.backward()
print("p", p)

p = F.softmax(a, dim=0)
p_1 =  torch.autograd.grad(p[1], [a], retain_graph=True)
print("p_1: ", p_1)
p_2 = torch.autograd.grad(p[2], [a])
print("p_2: ", p_2)







