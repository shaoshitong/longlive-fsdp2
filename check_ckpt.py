import torch

ckpt = torch.load("/share/st_workspace/LongLive-FSDP2/logs_fsdp2/step_000700.pth")
print(ckpt['ema'].keys())