"""
@DATE: 2022/4/27
@Author  : ld
"""
import torch
from torch.nn import functional as F
import visdom

"""
# tensorbordX 
- 只支持numpy数据
- 必须要转到 cpu
- 数据结果太大
- 运行效率不如 visdom
"""

"""
# visdom
- 安装 pip install visdom
- 启动 python -m visdom.server
- 本质是一个web服务, 往里面扔数据
"""

