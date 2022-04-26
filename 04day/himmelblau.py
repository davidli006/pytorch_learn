"""
# 优化问题
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] **2 - 7) ** 2

def show_img(X, Y, Z):
    fig = plt.figure("himmelblau")
    # 这里浪费了一大把时间 numpy包1.20.3有bug
    ax = fig.add_subplot(projection="3d")
    # ax = fig.gca(projection="3d")  # 已弃用 报警告
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print("x: ", x.shape, "y:", y.shape)

X, Y = np.meshgrid(x, y)
print("X: ", X.shape, "Y: ", Y.shape)
Z = himmelblau([X, Y])
print("Z: ", Z.shape)

show_img(X, Y, Z)

# [1., 0], [-4, 0], [4, 0]
w = torch.tensor([-4., 0.], requires_grad=True)
optimizer = torch.optim.Adam([w], lr=1e-3)

for step in range(20000):
    pred = himmelblau(w)
    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

    if step % 2000 == 0:
        print(f"step {step}: x={w.tolist()}, y={pred.item()}")



