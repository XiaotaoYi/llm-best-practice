import torch
print(torch.__version__)  # 应显示安装的版本
print(torch.backends.mps.is_available())  # 应返回 True