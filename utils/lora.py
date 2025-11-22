import torch
import torch.nn as nn

class LoRALayer(nn.Linear):  # Inherit directly from nn.Linear
    def __init__(self, in_dim, out_dim, rank, alpha, weak_lora_alpha=0.1):
        super().__init__(in_dim, out_dim)  # Initialize as standard Linear
        # Initialize LoRA parameters
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.weak_lora_alpha = weak_lora_alpha
        self.use_lora = True
        self.weak_lora = False

    def forward(self, x):
        linear_out = super().forward(x)  # Standard linear computation
        if self.use_lora:
            if self.weak_lora:
                return linear_out + self.alpha * (x @ self.A @ self.B) * self.weak_lora_alpha
            else:
                return linear_out + self.alpha * (x @ self.A @ self.B)
        else:
            if self.training:
                return linear_out + 0 * (x @ self.A @ self.B)  # Zero out LoRA but keep in graph for gradients
            else:
                return linear_out


def replace_linear_with_lora(module, rank=16, alpha=1., tag=0, weak_lora_alpha=0.1):
    for name, child in module.named_children():
        if name == "single_fusion_blocks":
            tag = tag + 1
        if isinstance(child, nn.Linear) and tag >= 1:
            param_device = getattr(child.weight, "device", None)
            use_meta = param_device is not None and param_device.type == "meta"
            if use_meta:
                with torch.device("meta"):
                    new_layer = LoRALayer(child.in_features, child.out_features, rank, alpha, weak_lora_alpha=weak_lora_alpha)
            else:
                new_layer = LoRALayer(child.in_features, child.out_features, rank, alpha, weak_lora_alpha=weak_lora_alpha)
                new_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None and new_layer.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, new_layer)
        else:
            replace_linear_with_lora(child, rank, alpha, tag, weak_lora_alpha=weak_lora_alpha)

def lora_false(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            for n, param in module.named_parameters():
                if n in ['A', 'B']:  # Only LoRA params
                    param.requires_grad_(True)
            module.use_lora = False
            module.weak_lora = False
            
def lora_true(model, alpha=0.25):
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            for n, param in module.named_parameters():
                if n in ['A', 'B']:  # Only LoRA params
                    param.requires_grad_(True)
            module.use_lora = True
            module.weak_lora_alpha = alpha
            module.weak_lora = True


def weak_lora(model, alpha=0.25):
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            for n, param in module.named_parameters():
                if n in ['A', 'B']:  # Only LoRA params
                    param.requires_grad_(True)
            module.use_lora = True
            module.weak_lora_alpha = alpha
            module.weak_lora = True

            
def ori_lora(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            for n, param in module.named_parameters():
                if n in ['A', 'B']:  # Only LoRA params
                    param.requires_grad_(True)
            module.use_lora = True
            module.weak_lora = False