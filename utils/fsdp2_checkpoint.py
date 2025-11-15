"""
FSDP2兼容的分布式checkpoint实现
基于PyTorch DCP (Distributed Checkpoint) APIs
"""

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from typing import Any
from pathlib import Path
import torch.distributed as dist

try:
    # 检查DCP可用性
    import torch.distributed.checkpoint
    HAS_DCP = True
except ImportError:
    HAS_DCP = False
    print("DCP not available - falling back to standard checkpoint")


def _sanitize_for_dcp(obj: Any) -> Any:
    """递归清理对象，移除OmegaConf类型，转换为基本Python类型"""
    try:
        # 检查是否是OmegaConf类型
        if hasattr(obj, '__module__') and obj.__module__ and 'omegaconf' in obj.__module__:
            # 是OmegaConf类型，尝试转换为基本类型
            if hasattr(obj, 'to_container'):
                # DictConfig/ListConfig
                return obj.to_container(resolve=True)
            else:
                # 其他OmegaConf类型，转换为string
                return str(obj)
        elif isinstance(obj, dict):
            # 递归处理字典
            return {k: _sanitize_for_dcp(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # 递归处理列表/元组
            result = [_sanitize_for_dcp(item) for item in obj]
            return result if isinstance(obj, list) else tuple(result)
        else:
            # 基本类型或其他可序列化类型
            return obj
    except Exception as e:
        print(f"[WARNING] Failed to sanitize object {type(obj)}: {e}, converting to string")
        return str(obj)


# 移除Stateful实现，使用官方推荐的简单方式


class FSDP2CheckpointManager:
    """
    基于DCP的FSDP2 checkpoint管理器
    按照PyTorch官方推荐的简单方式实现
    """
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        
        # 创建checkpoint目录
        if dist.get_rank() == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 同步等待目录创建
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    def save(self, model, optimizer=None, ema=None, scheduler=None, step: int = 0, state_dict: dict = None, **extra_state):
        """使用DCP保存checkpoint - 官方推荐方式"""
        if not HAS_DCP:
            print("[ERROR] DCP not available, cannot save FSDP2 checkpoint")
            return False
        
        checkpoint_id = f"step_{step:06d}"
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        try:
            print(f"[INFO] Saving FSDP2 checkpoint to {checkpoint_path}")
            
            if state_dict is None:
                # 1. 使用官方API获取model和optimizer state dict
                if optimizer is not None:
                    try:
                        print("[DEBUG] Getting model and optimizer state dict...")
                        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
                    except:
                        print("[DEBUG] Getting model state dict only...")
                        from torch.distributed.checkpoint.state_dict import get_model_state_dict
                        model_state_dict = get_model_state_dict(model)
                        optimizer_state_dict = {}     
                else:
                    print("[DEBUG] Getting model state dict only...")
                    from torch.distributed.checkpoint.state_dict import get_model_state_dict
                    model_state_dict = get_model_state_dict(model)
                    optimizer_state_dict = {}
                
                # 2. 组织state dict
                state_dict = {
                    "model": model_state_dict,
                    "optimizer": optimizer_state_dict,
                }
            
            # 3. 处理EMA (如果有)
            if ema is not None:
                try:
                    print("[DEBUG] Getting EMA state...")
                    ema_state = ema.state_dict()
                    # 清理EMA中的OmegaConf类型
                    state_dict["ema"] = _sanitize_for_dcp(ema_state)
                    print("[INFO] EMA state included")
                except Exception as e:
                    print(f"[WARNING] Failed to get EMA state: {e}")
            
            # 4. 添加基本训练状态
            for key, value in extra_state.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    state_dict[key] = value
                else:
                    print(f"[WARNING] Skipping complex extra_state '{key}' of type {type(value)}")
            
            state_dict["step"] = step
            
            # 5. 直接使用DCP保存
            print("[DEBUG] Calling dcp.save...")
            dcp.save(state_dict, checkpoint_id=str(checkpoint_path))
            print(f"[INFO] FSDP2 checkpoint saved successfully at step {step}")

            # 5.1 保存训练进度到JSON，便于可靠恢复
            try:
                if dist.get_rank() == 0:
                    import json
                    meta = {
                        'epoch': int(extra_state.get('epoch', 0)),
                        'global_step': int(extra_state.get('global_step', step)),
                        'step': int(step),
                        'is_best': bool(extra_state.get('is_best', False)),
                        'is_interrupted': bool(extra_state.get('is_interrupted', False)),
                    }
                    (checkpoint_path / "train_state.json").write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2)
                    )
            except Exception as e:
                print(f"[WARNING] Failed to write train_state.json: {e}")
            
            # 6. 清理旧checkpoint
            if dist.get_rank() == 0:
                self._cleanup_old_checkpoints()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save FSDP2 checkpoint: {e}")
            import traceback
            print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            return False
    
    def load(self, checkpoint_path: str, model, optimizer=None, ema=None, scheduler=None):
        """使用DCP加载checkpoint - 先构造目标布局，再执行加载"""
        if not HAS_DCP:
            print("[ERROR] DCP not available, cannot load FSDP2 checkpoint")
            return False
        
        try:
            print(f"[INFO] Loading FSDP2 checkpoint from {checkpoint_path}")
            # 1) 准备接收权重的目标布局（skeleton）
            if optimizer is not None:
                print("[DEBUG] Preparing model+optimizer state dict skeleton...")
                model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
                state_dict = {
                    "model": model_state_dict,
                    "optimizer": optimizer_state_dict,
                }
            else:
                print("[DEBUG] Preparing model state dict skeleton...")
                from torch.distributed.checkpoint.state_dict import get_model_state_dict
                state_dict = {"model": get_model_state_dict(model)}

            if ema is not None:
                state_dict["ema"] = {}

            # 2) 使用DCP加载
            print("[DEBUG] Calling dcp.load...")
            dcp.load(state_dict, checkpoint_id=checkpoint_path)

            # 3) 应用到模型/优化器
            if optimizer is not None and "optimizer" in state_dict:
                print("[DEBUG] Setting model and optimizer state...")
                set_state_dict(
                    model,
                    optimizer,
                    model_state_dict=state_dict["model"],
                    optim_state_dict=state_dict["optimizer"],
                )
            else:
                print("[DEBUG] Setting model state only...")
                from torch.distributed.checkpoint.state_dict import set_model_state_dict
                set_model_state_dict(model, state_dict["model"])

            # 4) EMA状态（如果存在）
            if ema is not None and "ema" in state_dict and state_dict["ema"]:
                try:
                    print("[DEBUG] Loading EMA state...")
                    ema.load_state_dict(state_dict["ema"])  # 已在保存时做过sanitize
                    print("[INFO] EMA state loaded successfully")
                except Exception as e:
                    print(f"[WARNING] Failed to load EMA state: {e}")

            # 5) 读取额外训练状态（优先从JSON读取）
            extra_state = {}
            try:
                ckpt_dir = Path(checkpoint_path)
                meta_path = ckpt_dir / "train_state.json"
                if meta_path.exists():
                    import json
                    extra_state = json.loads(meta_path.read_text())
                else:
                    # 兼容：尝试从DCP内读取（若存在）
                    for key in ["epoch", "global_step", "step", "is_best", "is_interrupted"]:
                        if key in state_dict and state_dict[key] is not None:
                            extra_state[key] = state_dict[key]
                    # 兜底：从目录名解析step（例如 step_002400）
                    try:
                        name = ckpt_dir.name
                        if name.startswith("step_"):
                            parsed = int(name.split("_")[1])
                            extra_state.setdefault("step", parsed)
                            extra_state.setdefault("global_step", parsed)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[DEBUG] Failed to read extra training state: {e}")

            print("[INFO] FSDP2 checkpoint loaded successfully")
            return extra_state
            
        except Exception as e:
            print(f"[ERROR] Failed to load FSDP2 checkpoint: {e}")
            import traceback
            print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """清理旧的checkpoint"""
        try:
            # 获取所有checkpoint目录
            checkpoints = [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('step_')]
            
            # 按步数排序
            checkpoints.sort(key=lambda x: int(x.name.split('_')[1]))
            
            # 删除多余的checkpoint
            if len(checkpoints) > self.keep_last_n:
                for checkpoint in checkpoints[:-self.keep_last_n]:
                    import shutil
                    shutil.rmtree(checkpoint)
                    print(f"[INFO] Removed old checkpoint: {checkpoint}")
                    
        except Exception as e:
            print(f"[WARNING] Failed to cleanup old checkpoints: {e}")
