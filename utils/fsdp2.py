"""
FSDP2 (Fully Sharded Data Parallel v2) wrapper for distributed training.
使用PyTorch 2.8的新FSDP2 API实现更好的device管理。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Set, List
from omegaconf import DictConfig
try:
    # Import FSDP2 APIs. Note: `wrap` must come from the `fsdp.wrap` module;
    # importing `wrap` from `torch.distributed.fsdp` yields the module object,
    # which is not callable and breaks submodule marking. (FSDP2: we won't use wrap)
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict, set_model_state_dict, StateDictOptions
    )
    HAS_FSDP2 = True
    print("FSDP2 available")
except ImportError:
    print("FSDP2 not available - falling back to FSDP1 or DDP")
    HAS_FSDP2 = False


def create_device_mesh(world_size: int) -> Optional[DeviceMesh]:
    """创建设备网格用于FSDP2（使用 init_device_mesh，默认1D形状）。"""
    if not HAS_FSDP2 or not torch.distributed.is_initialized():
        return None
    try:
        # 默认 1D mesh，利于扩展为多维（dp,tp）
        mesh_shape = [world_size]
        device_mesh = init_device_mesh("cuda", mesh_shape)
        return device_mesh
    except Exception as e:
        print(f"init_device_mesh failed: {e}, falling back to DeviceMesh")
        device_mesh = DeviceMesh("cuda", list(range(world_size)))
        return device_mesh


def _get_module_by_name(root: nn.Module, dotted_name: str) -> nn.Module:
    current = root
    for part in dotted_name.split('.'):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _set_module_by_name(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split('.')
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def apply_fsdp2_to_model(
    model: nn.Module,
    config: DictConfig,
    device_mesh: Optional[DeviceMesh] = None
) -> nn.Module:
    """
    对模型应用FSDP2包装
    
    Args:
        model: 要包装的模型（应该在meta device上初始化）
        config: FSDP2配置
        device_mesh: 设备网格
        
    Returns:
        FSDP2包装的模型
    """
    if not HAS_FSDP2:
        print("FSDP2 not available - returning unwrapped model")
        return model
    
    if not torch.distributed.is_initialized():
        print("Distributed not initialized - returning unwrapped model")
        return model
    
    print("Applying FSDP2 to model...")
    
    # 1. 确保模型在meta设备上
    if not all(param.device == torch.device("meta") for param in model.parameters()):
        print("Model parameters not on meta device - this may cause issues")
    
    # 2. 官方推荐：按 Transformer 层/Block 粒度进行 fully_shard 以降低峰值显存；
    #    若无法解析到类，则退回仅对根模型 fully_shard，以保证健壮性。
    wrap_classes: Set[type] = set()
    try:
        from engine.backends.wrap_policy import get_transformer_layer_classes
        classes: List[type] = get_transformer_layer_classes(config)
        wrap_classes = set(classes)
    except Exception as e:
        print(f"WrapPolicy resolution failed, falling back to root-only fully_shard: {e}")
    
    # A) 子模块按层/Block 粒度建组：逐个 fully_shard（非根组默认 RAAF=True）
    nonroot_raaf = bool(config.get('nonroot_reshard_after_forward', True))
    sharded_cnt = 0
    if wrap_classes:
        print(f"FSDP2: block-wise sharding for {[c.__name__ for c in wrap_classes]}")
        target_types = tuple(wrap_classes)
        for sub in model.modules():
            if sub is model:
                continue
            if isinstance(sub, target_types):
                fully_shard(
                    sub,
                    mesh=device_mesh,
                    reshard_after_forward=nonroot_raaf,
                )
                sharded_cnt += 1
        print(f"FSDP2: fully_shard() applied to {sharded_cnt} submodules (block-wise).")
    else:
        print("FSDP2: no block classes resolved; will only shard the root module.")

    # B) 根模型兜底建组：默认 RAAF=False（可由 config 覆盖）
    root_raaf = bool(config.get('root_reshard_after_forward', False))
    fully_shard(
        model,
        mesh=device_mesh,
        reshard_after_forward=root_raaf,
    )
    print("FSDP2 wrapping completed")
    return model


def initialize_fsdp2_model_parameters(model: nn.Module, device: torch.device = None):
    """
    初始化FSDP2模型参数到指定设备
    
    Args:
        model: FSDP2包装的模型
        device: 目标设备
    """
    if device is None:
        device = torch.device("cuda")
    
    print(f"Initializing FSDP2 model parameters to {device}")
    
    # 验证模型参数是否为DTensor
    param_count = 0
    for param in model.parameters():
        if isinstance(param, DTensor):
            param_count += 1
        elif param.device == torch.device("meta"):
            print(f"Found meta parameter that's not DTensor: {param.shape}")
    
    print(f"Found {param_count} DTensor parameters")
    
    # 仅 materialize 空存储到目标设备。不要 reset_parameters：
    # 后续将通过 DCP 加载预训练；避免无谓写入和潜在峰值。
    model.to_empty(device=device)
    print("FSDP2 model parameter storage materialized (to_empty); skip reset_parameters()")


    # 注意：具体类解析逻辑已移到 WrapPolicy


def get_fsdp2_state_dict(model: nn.Module, full_state_dict: bool = True) -> Dict[str, Any]:
    """
    获取FSDP2模型的状态字典 - 改进版本，避免同步死锁
    
    Args:
        model: FSDP2包装的模型
        full_state_dict: 是否返回完整状态字典（用于保存检查点）
        
    Returns:
        状态字典
    """
    if not HAS_FSDP2:
        return model.state_dict()
    
    if full_state_dict:
        try:
            # 方法1：尝试使用FSDP2的分布式checkpoint API
            from utils.distributed_logging import is_rank_zero
            
            # 先尝试使用no_sync上下文来避免不必要的同步
            try:
                import torch.distributed as dist
                
                # 使用DCP API，但采用rank0_only模式
                state_dict = get_model_state_dict(
                    model=model,
                    options=StateDictOptions(
                        full_state_dict=True,
                    )
                )
                
                if is_rank_zero():
                    print("[INFO] Successfully got FSDP2 state dict")
                    return state_dict
                else:
                    # 非rank0返回空字典，避免保存
                    return {}
                    
            except Exception as e:
                print(f"[WARNING] DCP state dict failed: {e}, trying fallback")
                
                # 方法2：回退到简单的state_dict()
                if is_rank_zero():
                    return model.state_dict()
                else:
                    return {}
                    
        except Exception as e:
            print(f"[ERROR] All FSDP2 state dict methods failed: {e}")
            # 最后的回退
            if is_rank_zero():
                return model.state_dict()
            else:
                return {}
    else:
        # 返回分片状态字典（DTensor格式）
        return model.state_dict()


def load_fsdp2_state_dict(
    model: nn.Module, 
    state_dict: Dict[str, Any],
    strict: bool = True,
    broadcast_from_rank0: bool = True
):
    """
    加载状态字典到FSDP2模型
    
    Args:
        model: FSDP2包装的模型
        state_dict: 要加载的状态字典
        strict: 严格模式
        broadcast_from_rank0: 是否从rank0广播
    """
    if not HAS_FSDP2:
        return model.load_state_dict(state_dict, strict=strict)
    
    # 使用DCP API加载状态字典
    return set_model_state_dict(
        model=model,
        model_state_dict=state_dict,
        options=StateDictOptions(
            full_state_dict=True,
            strict=strict,
            broadcast_from_rank0=broadcast_from_rank0,
        )
    )


def convert_full_to_sharded_state_dict(
    model: nn.Module,
    full_state_dict: Dict[str, torch.Tensor]
) -> Dict[str, DTensor]:
    """
    将完整状态字典转换为分片状态字典（DTensor格式）
    
    Args:
        model: FSDP2模型（用于获取DTensor规格）
        full_state_dict: 完整状态字典
        
    Returns:
        分片状态字典
    """
    if not HAS_FSDP2:
        return full_state_dict
    
    sharded_sd = {}
    meta_sharded_sd = model.state_dict()
    
    for param_name, full_tensor in full_state_dict.items():
        if param_name in meta_sharded_sd:
            sharded_meta_param = meta_sharded_sd[param_name]
            if isinstance(sharded_meta_param, DTensor):
                # 分发 tensor 到各个 rank；对齐 dtype 到目标 shard 的本地 dtype
                try:
                    target_dtype = sharded_meta_param.to_local().dtype
                except Exception:
                    target_dtype = full_tensor.dtype
                full_aligned = full_tensor.to(dtype=target_dtype, copy=False)
                sharded_tensor = distribute_tensor(
                    full_aligned,
                    sharded_meta_param.device_mesh,
                    sharded_meta_param.placements,
                )
                # 直接返回 DTensor/Tensor 本体，交由 set_model_state_dict 处理
                sharded_sd[param_name] = sharded_tensor
            else:
                sharded_sd[param_name] = full_tensor
        else:
            sharded_sd[param_name] = full_tensor
    
    return sharded_sd


def is_fsdp2_model(model: nn.Module) -> bool:
    """检查模型是否使用了FSDP2"""
    if not HAS_FSDP2:
        return False
    
    # 检查是否有DTensor参数
    for param in model.parameters():
        if isinstance(param, DTensor):
            return True
    
    return False


def get_model_device(model: nn.Module) -> torch.device:
    """获取模型设备，兼容FSDP2的DTensor"""
    for param in model.parameters():
        if isinstance(param, DTensor):
            # device_type 是字符串（如 "cuda"），转换为 torch.device
            try:
                return torch.device(param.device_mesh.device_type)
            except Exception:
                return torch.device("cuda")
        else:
            return param.device
    
    # 如果没有参数，返回默认设备
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_fsdp2_model_info(model: nn.Module):
    """打印FSDP2模型信息（用于调试）"""
    if not is_fsdp2_model(model):
        print("Model is not using FSDP2")
        return
    
    dtensor_count = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            dtensor_count += 1
            total_params += param.numel()
            print(f"DTensor param: {name}, shape: {param.shape}, placements: {param.placements}")
    
    print("FSDP2 Model Info:")
    print(f"  - DTensor parameters: {dtensor_count}")
    print(f"  - Total parameters: {total_params:,}")
