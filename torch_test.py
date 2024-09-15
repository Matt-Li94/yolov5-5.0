import torch  # 如果pytorch安装成功即可导入

print("CUDA是否可用:", torch.cuda.is_available())  # 查看CUDA是否可用
print("可用的gpu数量:", torch.cuda.device_count())  # 查看可用的CUDA数量
print("CUDA的版本号:", torch.version.cuda)  # 查看CUDA的版本号
print("torch的版本:", torch.__version__)