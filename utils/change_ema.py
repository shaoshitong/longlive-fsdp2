import torch

ckpt = torch.load("/share/st_workspace/LongLive-FSDP2/logs_fsdp2_ema/step_000690.pth")

ema_params = ckpt["ema"]["ema_params"]

model_params = ckpt["model"]

# print(ema_params.keys()) # "model.blocks.27.modulation"
# print(model_params.keys())

for i in range(30):
    _s = f"model.blocks.{i}.modulation"
    if _s in ema_params:
        ema_params[_s] = model_params[_s]

ckpt["ema"]["ema_params"] = ema_params

torch.save(ckpt, "/share/st_workspace/LongLive-FSDP2/logs_fsdp2_ema/step_000690_ema.pth")

