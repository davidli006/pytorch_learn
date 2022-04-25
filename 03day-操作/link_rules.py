"""
# 链式法则
"""
import torch
from torch import autograd

"""
∂y/∂x = ∂y/∂g * ∂g/∂x
"""

x = torch.tensor(1.)
w1 = torch.tensor(2., requires_grad=True)
b1 = torch.tensor(1.)
w2 = torch.tensor(2., requires_grad=True)
b2 = torch.tensor(1.)

g1 = x * w1 + b1
y2 = g1 * w2 + b2

dy2_g1 = autograd.grad(y2, [g1], retain_graph=True)[0]
dg1_w1 = autograd.grad(g1, [w1], retain_graph=True)[0]

dy2_w1 = autograd.grad(y2, [w1], retain_graph=True)[0]

# ∂y/∂g * ∂g/∂x
print(dy2_g1 * dg1_w1)

# dy2_w1
print(dy2_w1)

