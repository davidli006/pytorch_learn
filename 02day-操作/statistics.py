"""
# 统计属性
"""
import torch

print("-"*50)
"""
# norm-p
- p范数
"""
a = torch.full([8], 1.)
b = a.view(2, 4)
c = a.view(2, 2, 2)
print("a:", a, "--->", a.shape)
print("b:", b, "--->", b.shape)
print("c:", c, "--->", c.shape)

print(a.norm(1), b.norm(1), c.norm(1))  # 所有元素绝对值的和 ∑|x|

"""
b : [[1, 1, 1, 1]
    ,[1, 1, 1, 1]]  =>  [4, 4]
b 在 1 维度上 norm 把 这个维度里的所有数 绝对值求和 
"""
bb = b.norm(1, dim=1)
print("bb: ", bb)
"""
c: [
    [[1, 1], [1, 1]],
    [[1, 1], [1, 1]]
    ]                =>     [
                                [2], [2],
                                [2], [2]
                            ]
c 在 1 维度上 norm 把 这个维度里的所有数 绝对值求和                             
"""
cc = c.norm(1, dim=1)
print("cc", cc)

"""
b : [[1, 1, 1, 1]
    ,[1, 1, 1, 1]] => [4, 4] => [2, 2]
b 在 1 维度上 norm 把 这个维度里的所有数 绝对值求和
所有元素绝对值的和  (∑x**2)**(1/2) 
"""
bbb = b.norm(2, dim=1)
print("bbb: ", bbb)

"""
c: [
    [[1, 1], [1, 1]],
    [[1, 1], [1, 1]]
    ]                =>  [
                            [2], [2]
                            [2], [2]
                         ] => [[1.4],[1.4],
                               [1.4],[1.4]]

"""
ccc = c.norm(2, dim=1)
print("ccc: ", ccc)

"""
mean    平均值
sum     求和
min     最小值
max     最大值
prod    
"""
a = torch.arange(8).view(2, 4).float()
print(a)
print("mean: ", a.mean())
print("sum: ", a.sum())
print("min: ", a.min(), a.argmax())  # 返回索引 打平之后给索引
print("max: ", a.max(), a.argmin())  # 返回索引
print("prod: ", a.prod())




