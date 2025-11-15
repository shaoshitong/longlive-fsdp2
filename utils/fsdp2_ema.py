"""
FSDP2 Compatible Exponential Moving Average (EMA) implementation.

此实现专门针对PyTorch 2.8的FSDP2环境设计，正确处理DTensor（分布式张量）。

关键策略（多卡安全）：
- 影子权重以 DTensor 存放，布局与模型参数完全一致（相同 device_mesh 与 placements）
- 仅在分片状态下做就地更新：ema = decay * ema + (1 - decay) * param
- 存储为 FP32 提高数值稳定性；应用到模型时按需临时 cast 回原 dtype
- 评估时支持分片就地切换与恢复，避免全量 all-gather
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
try:
    # DCP API 仍保留以兼容保存/加载的可能扩展（当前实现不依赖全量state_dict）
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict, set_model_state_dict, StateDictOptions
    )
    from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
    HAS_FSDP2 = True
except ImportError:
    HAS_FSDP2 = False
    print("FSDP2 DTensor support not available - falling back to standard EMA")


class FSDP2EMAModel:
    """
    FSDP2兼容的指数移动平均（EMA）实现
    
    与标准EMA不同，此实现：
    1. 不使用deepcopy创建EMA模型副本（因为无法处理DTensor）
    2. 使用状态字典方式存储和更新EMA参数
    3. 支持FSDP2的分布式参数访问
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
        update_after_step: int = 0,
        use_fsdp2: bool = True
    ):
        """
        Args:
            model: 要跟踪的FSDP2模型
            decay: 指数衰减率
            device: EMA参数存储设备（如果为None则自动检测）
            update_after_step: 在此步数之后开始EMA更新
            use_fsdp2: 是否使用FSDP2兼容模式
        """
        self.decay = decay
        self.device = device if device is not None else torch.device(
            f"cuda:{torch.distributed.get_rank()}" if torch.distributed.is_initialized() else "cuda"
        )
        self.update_after_step = update_after_step
        self.step_count = 0
        self.use_fsdp2 = use_fsdp2 and HAS_FSDP2
        
        if self.use_fsdp2:
            print("[INFO] Initializing FSDP2-compatible EMA...")
            self._init_fsdp2_ema(model)
        else:
            print("[INFO] Initializing standard EMA...")
            self._init_standard_ema(model)
    
    def _init_fsdp2_ema(self, model: nn.Module):
        """初始化FSDP2兼容的EMA（影子 DTensor-EMA，同构分片+FP32存放）"""
        try:
            # 仅跟踪需要训练的浮点参数
            self.ema_params: Dict[str, torch.Tensor] = {}
            self.param_dtypes: Dict[str, torch.dtype] = {}
            self.param_is_dtensor: Dict[str, bool] = {}
            self._param_refs: Dict[str, nn.Parameter] = {}
            tracked = 0

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if not torch.is_floating_point(param):
                    continue

                self._param_refs[name] = param
                self.param_dtypes[name] = param.dtype
                is_dtensor = isinstance(param, DTensor)
                self.param_is_dtensor[name] = is_dtensor

                if is_dtensor:
                    # 取本地 shard，克隆为 FP32，再构造同 mesh/placements 的 DTensor
                    local = param.to_local().detach().clone().to(dtype=torch.float32)
                    ema_param = DTensor.from_local(local, param.device_mesh, param.placements)
                else:
                    target_device = self.device or param.device
                    ema_param = param.detach().clone().to(device=target_device, dtype=torch.float32)

                self.ema_params[name] = ema_param
                tracked += 1

            self.trainable_param_names = set(self.ema_params.keys())
            print(f"[INFO] FSDP2 EMA initialized with {tracked} parameters")

        except Exception as e:
            print(f"[ERROR] Failed to initialize FSDP2 EMA: {e}")
            print("[INFO] Falling back to standard EMA...")
            self.use_fsdp2 = False
            self._init_standard_ema(model)
    
    def _init_standard_ema(self, model: nn.Module):
        """初始化标准EMA（回退方案）"""
        import copy
        
        # 标准EMA实现 - 仅作为回退
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        
        if self.device is not None:
            self.ema_model.to(self.device)
    
    def update(self, model: nn.Module):
        """更新EMA参数"""
        self.step_count += 1

        # 调试：记录是否达到启用阈值
        if self.step_count <= self.update_after_step:
            print(
                f"[EMA] Skipped update at step_count={self.step_count} (<={self.update_after_step})"
            )
            return

        if self.use_fsdp2:
            self._update_fsdp2(model)
        else:
            self._update_standard(model)
    
    def _update_fsdp2(self, model: nn.Module):
        """FSDP2模式下的EMA更新（分片就地，无全量聚合）"""
        try:
            # 预采样一个参数用于设备/形状日志
            sample_name = None
            for n in self.trainable_param_names:
                sample_name = n
                break

            def _mem_gb(dev: Optional[torch.device]) -> str:
                try:
                    if dev is None:
                        return "N/A"
                    torch.cuda.synchronize(dev)
                    a = torch.cuda.memory_allocated(dev) / (1024 ** 3)
                    r = torch.cuda.memory_reserved(dev) / (1024 ** 3)
                    return f"alloc={a:.2f}GB,resv={r:.2f}GB"
                except Exception:
                    return "N/A"

            # 记录更新前内存
            pre_dev = None
            if sample_name is not None:
                p0 = self._param_refs.get(sample_name)
                if isinstance(p0, DTensor):
                    try:
                        pre_dev = p0.to_local().device
                    except Exception:
                        pre_dev = None
                else:
                    pre_dev = getattr(p0, 'device', None)
            print(f"[EMA] Update start: step_count={self.step_count}, decay={self.decay}, mem_before={_mem_gb(pre_dev)}")

            # 统计类型分布
            try:
                n_total = len(self.trainable_param_names)
                n_dt = sum(1 for n in self.trainable_param_names if isinstance(self._param_refs.get(n), DTensor))
                print(f"[EMA] Trainable params: total={n_total}, dtensor={n_dt}, tensor={n_total - n_dt}")
            except Exception:
                pass

            # 核心更新
            with torch.no_grad():
                for name in self.trainable_param_names:
                    if name not in self._param_refs:
                        continue
                    model_param = self._param_refs[name]
                    ema_param = self.ema_params[name]

                    # 以 FP32 累加，确保DTensor的mesh/placements一致
                    if isinstance(model_param, DTensor):
                        current = model_param
                        if current.dtype != torch.float32:
                            current = current.to(dtype=torch.float32)
                        # 简单一致性检查
                        try:
                            if hasattr(ema_param, 'device_mesh') and hasattr(current, 'device_mesh'):
                                if ema_param.device_mesh != current.device_mesh or ema_param.placements != current.placements:
                                    print(f"[EMA] DTensor spec mismatch at '{name}'")
                        except Exception:
                            pass
                        ema_param.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
                    else:
                        current = model_param.detach()
                        if current.dtype != torch.float32:
                            current = current.to(dtype=torch.float32)
                        ema_param.mul_(self.decay).add_(current, alpha=1.0 - self.decay)

            # 记录更新后内存
            print(f"[EMA] Update end:   step_count={self.step_count}, mem_after={_mem_gb(pre_dev)}")
        except Exception as e:
            print(f"[ERROR] FSDP2 EMA update failed at step_count={self.step_count}: {e}")
            raise
    
    def _update_standard(self, model: nn.Module):
        """标准模式下的EMA更新"""
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), 
                model.parameters()
            ):
                if model_param.requires_grad:
                    # EMA update: ema = decay * ema + (1 - decay) * model
                    ema_param.data.mul_(self.decay).add_(
                        model_param.data, alpha=1.0 - self.decay
                    )
    
    def apply_shadow(self, model: nn.Module):
        """将EMA参数应用到模型（用于推理，分片就地交换）"""
        if self.use_fsdp2:
            self._apply_shadow_fsdp2(model)
        else:
            self._apply_shadow_standard(model)
    
    def _apply_shadow_fsdp2(self, model: nn.Module):
        """FSDP2模式下应用EMA参数（分片就地copy_，并缓存原值以便恢复）"""
        try:
            self._stash_original: Dict[str, torch.Tensor] = {}
            with torch.no_grad():
                for name in self.trainable_param_names:
                    if name not in self._param_refs:
                        continue
                    p = self._param_refs[name]
                    ema_p = self.ema_params[name]

                    # 缓存原始参数（保持分片）
                    self._stash_original[name] = p.detach().clone()

                    # 将 EMA 值按模型参数 dtype 写回
                    if ema_p.dtype != p.dtype:
                        p.copy_(ema_p.to(dtype=p.dtype))
                    else:
                        p.copy_(ema_p)

            print("[INFO] EMA parameters applied (sharded, in-place)")
        except Exception as e:
            print(f"[ERROR] Failed to apply EMA parameters: {e}")
    
    def _apply_shadow_standard(self, model: nn.Module):
        """标准模式下应用EMA参数"""
        with torch.no_grad():
            for model_param, ema_param in zip(
                model.parameters(), 
                self.ema_model.parameters()
            ):
                if model_param.requires_grad:
                    model_param.data.copy_(ema_param.data)
    
    def restore(self, model: nn.Module):
        """在FSDP2模式下，从缓存恢复原始参数（仅当之前调用过apply_shadow）"""
        if not self.use_fsdp2:
            # 标准模式下未实现
            return
        if not hasattr(self, '_stash_original'):
            return
        try:
            with torch.no_grad():
                for name, orig in self._stash_original.items():
                    if name in self._param_refs:
                        self._param_refs[name].copy_(orig)
            del self._stash_original
            print("[INFO] Restored original parameters after EMA")
        except Exception as e:
            print(f"[ERROR] Failed to restore original parameters: {e}")
    
    def state_dict(self) -> Dict[str, Any]:
        """获取EMA状态字典（用于保存）
        FSDP2模式下直接返回DTensor形式的参数，让DCP处理分布式保存
        """
        if self.use_fsdp2:
            # 让DCP自动处理DTensor - 不手动聚合！
            print("[INFO] Returning FSDP2 EMA state with DTensor parameters for DCP")
            
            return {
                'ema_params': dict(self.ema_params),  # 直接返回DTensor字典
                'trainable_param_names': list(self.trainable_param_names),
                'decay': self.decay,
                'step_count': self.step_count,
                'update_after_step': self.update_after_step,
                'use_fsdp2': self.use_fsdp2
            }
        else:
            return {
                'ema_model': self.ema_model.state_dict(),
                'decay': self.decay,
                'step_count': self.step_count,
                'update_after_step': self.update_after_step,
                'use_fsdp2': self.use_fsdp2
            }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载EMA状态字典"""
        self.decay = state_dict['decay']
        self.step_count = state_dict['step_count']
        self.update_after_step = state_dict['update_after_step']
        self.use_fsdp2 = state_dict.get('use_fsdp2', False)
        
        if self.use_fsdp2:
            if 'ema_params' in state_dict:
                # 新格式：直接加载DTensor参数（DCP已处理分布式）
                print("[INFO] Loading FSDP2 EMA DTensor parameters (DCP format)")
                try:
                    self.ema_params.update(state_dict['ema_params'])
                    # 重建参数映射
                    restored = 0
                    for name in state_dict.get('trainable_param_names', []):
                        if name in self.ema_params and name in self._param_refs:
                            target = self._param_refs[name]
                            self.param_dtypes[name] = target.dtype
                            self.param_is_dtensor[name] = isinstance(target, DTensor)
                            restored += 1
                    
                    self.trainable_param_names = set(self.ema_params.keys())
                    print(f"[INFO] Loaded FSDP2 EMA with {restored} DTensor parameters")
                except Exception as e:
                    print(f"[ERROR] Failed to load DTensor EMA parameters: {e}")
                    
            elif 'ema_full_cpu' in state_dict or 'ema_state_dict' in state_dict:
                # 旧格式：从full CPU tensor重新分片
                print("[INFO] Loading FSDP2 EMA from full CPU tensors (legacy format)")
                ema_full_cpu = state_dict.get('ema_full_cpu', state_dict.get('ema_state_dict', {}))
                restored = 0
                for name in state_dict.get('trainable_param_names', []):
                    if name not in ema_full_cpu:
                        continue
                    if name not in self._param_refs:
                        continue
                    target = self._param_refs[name]
                    src_full = ema_full_cpu[name]

                    if isinstance(target, DTensor):
                        # 依据当前参数的 mesh/placements 进行分发
                        ema_param = distribute_tensor(
                            src_full.detach().clone(),
                            target.device_mesh,
                            target.placements,
                        ).to(dtype=torch.float32)
                    else:
                        ema_param = src_full.detach().clone().to(device=target.device, dtype=torch.float32)

                    self.ema_params[name] = ema_param
                    self.param_dtypes[name] = target.dtype
                    self.param_is_dtensor[name] = isinstance(target, DTensor)
                    restored += 1

                self.trainable_param_names = set(self.ema_params.keys())
                print(f"[INFO] Loaded FSDP2 EMA with {restored} parameters from legacy format")
            else:
                print("[WARNING] No FSDP2 EMA parameters found in state dict")
        elif 'ema_model' in state_dict:
            self.ema_model.load_state_dict(state_dict['ema_model'])
            print("[INFO] Loaded standard EMA model")
        else:
            print("[ERROR] Invalid EMA state dict format")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取EMA内存使用情况（调试用）"""
        if self.use_fsdp2:
            total_mb = 0.0
            device_str = 'N/A'
            for _, param in self.ema_params.items():
                try:
                    if isinstance(param, DTensor):
                        local = param.to_local()
                        numel = local.numel()
                        device_str = str(local.device)
                    else:
                        numel = param.numel()
                        device_str = str(param.device)
                    total_mb += numel * param.element_size() / (1024 * 1024)
                except Exception:
                    pass
            return {
                'total_ema_params': len(self.ema_params),
                'total_memory_mb': total_mb,
                'device': device_str
            }
        else:
            # 对于标准EMA，计算模型参数大小
            total_params = sum(p.numel() for p in self.ema_model.parameters())
            total_mb = total_params * 4 / (1024 * 1024)  # 假设fp32
            
            return {
                'total_ema_params': total_params,
                'total_memory_mb': total_mb,
                'device': str(next(self.ema_model.parameters()).device)
            }


def create_ema_model(
    model: nn.Module, 
    decay,
    device = None,
    update_after_step = 0,
    use_fsdp2: bool = True
) -> FSDP2EMAModel:
    """
    工厂函数：创建EMA模型实例
    
    Args:
        model: 要跟踪的模型
        config: EMA配置
        use_fsdp2: 是否使用FSDP2兼容模式
        
    Returns:
        FSDP2EMAModel实例
    """
    return FSDP2EMAModel(
        model=model,
        decay=decay,
        device=device,
        update_after_step=update_after_step,
        use_fsdp2=use_fsdp2
    )
