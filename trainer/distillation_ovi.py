# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import gc
import logging
import random
import re
from pathlib import Path

from utils.dataset import TextDataset, TwoTextDataset, cycle
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job, fsdp2_warp_wan, fsdp2_warp_ovi
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD, DMDSwitch, DMDOvi
from model.streaming_training import StreamingTrainingModel
import torch
import wandb
import time
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig
)
from torchvision.io import write_video
from safetensors.torch import load_file
# LoRA related imports
import peft
from peft import get_peft_model_state_dict
import safetensors.torch
import copy
from typing import Any, Dict, Optional

from utils.memory import gpu, get_cuda_free_memory_gb, log_gpu_memory
from pipeline import (
    CausalInferencePipeline,
    SwitchCausalInferencePipeline,
    CausalInferenceOviPipeline
)
from utils.fsdp2 import load_fsdp2_state_dict
from utils.lora import replace_linear_with_lora, lora_false, lora_true, weak_lora, ori_lora, LoRALayer
from utils.fsdp2_ema import create_ema_model
from utils.fsdp2_checkpoint import FSDP2CheckpointManager
from utils.debug_option import DEBUG, LOG_GPU_MEMORY, DEBUG_GRADIENT
import time
import torch.nn.init as init
import re

from torch.distributed.tensor import DTensor

def clip_grad_norm_for_fsdp2_offload(parameters, max_norm, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    if len(parameters) == 0:
        return torch.tensor(0.)

    total_norm_sq = 0.0
    for p in parameters:
        grad = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
        grad_norm = torch.norm(grad, norm_type)
        
        if norm_type == 2.0:
            total_norm_sq += grad_norm.item() ** norm_type
        else:
            total_norm_sq += grad_norm.item() ** norm_type

    device = torch.device(f"cuda:{dist.get_rank()}")
    total_norm_cuda = torch.tensor(total_norm_sq, device=device)
    # 全局求和
    dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM)
    total_norm = total_norm_cuda.item() ** (1.0 / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            grad = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
            grad.mul_(clip_coef)
            
    return total_norm

def _convert_fusion_branch_key(rest: str, branch_prefix: str, block_label: str) -> Optional[str]:
    if not rest:
        return None
    parts = rest.split(".")
    head = parts[0]
    tail = ".".join(parts[1:]) if len(parts) > 1 else ""
    skip_heads = {"block_mask"}
    base_heads = {
        "patch_embedding",
        "text_embedding",
        "time_embedding",
        "time_projection",
        "head",
        "img_emb",
    }
    if head == "blocks":
        if len(parts) < 3:
            return None
        block_idx = parts[1]
        remainder = ".".join(parts[2:])
        prefix = f"single_fusion_blocks.{block_idx}.{block_label}"
        return f"{prefix}.{remainder}" if remainder else prefix
    if head == "freqs":
        return f"{branch_prefix}_freqs"
    if head in skip_heads:
        return None
    if head in base_heads:
        base = f"{branch_prefix}_{head}"
        return f"{base}.{tail}" if tail else base
    return f"{branch_prefix}_{rest}"


def maybe_convert_fusion_casual_state_dict(state_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(state_dict, dict):
        return state_dict
    keys = state_dict.keys()
    if not any(k.startswith("video_model.") or k.startswith("audio_model.") for k in keys):
        return state_dict
    new_state: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key: Optional[str] = key
        if key.startswith("video_model."):
            new_key = _convert_fusion_branch_key(key[len("video_model."):], "vid", "video_block")
        elif key.startswith("audio_model."):
            new_key = _convert_fusion_branch_key(key[len("audio_model."):], "audio", "audio_block")
        elif key.startswith("single_fusion_blocks."):
            continue
        if new_key is not None:
            new_state[new_key] = value
    return new_state


def load_fusion_model_state_dict(model, state_dict: Optional[Dict[str, Any]], **kwargs):
    if state_dict is None:
        return [], []
    converted = maybe_convert_fusion_casual_state_dict(state_dict)
    return load_fsdp2_state_dict(model=model, state_dict=converted, **kwargs)

class Trainer:
    
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        self.formatter = lambda text: re.sub(r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S)
        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        self.use_one_logger = getattr(config, "use_one_logger", True)
        if self.is_main_process and not self.disable_wandb:
            wandb.login(
                # host=config.wandb_host,
                key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir
        app_start_time = time.time_ns() / 1_000_000 
        

        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "dmd_switch":
            self.model = DMDSwitch(config, device=self.device)
        elif config.distribution_loss == "dmd_ovi":
            self.model = DMDOvi(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        self.is_lora_enabled = False
        self.lora_config = None
        if hasattr(config, 'adapter') and config.adapter is not None:
            self.lora_config = config.adapter
            if self.is_main_process:
                print("Applying LoRA to models...")
            replace_linear_with_lora(self.model.generator.model, rank=self.lora_config.get('rank', 16))
            # Configure LoRA for fake_score if needed
            if getattr(self.lora_config, 'apply_to_critic', True):
                replace_linear_with_lora(self.model.fake_score.model, rank=self.lora_config.get('rank', 16))
                if self.is_main_process:
                    print("LoRA applied to both generator and critic")
            else:
                if self.is_main_process:
                    print("LoRA applied to generator only")

            for name, param in self.model.generator.model.named_parameters():
                if ".A" in name or ".B" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

            for name, param in self.model.fake_score.model.named_parameters():
                if ".A" in name or ".B" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        else:

            for name, param in self.model.generator.model.named_parameters():
                param.requires_grad_(True)
            for name, param in self.model.fake_score.model.named_parameters():
                param.requires_grad_(True)

        self.model.generator.model = fsdp2_warp_ovi(
            self.model.generator.model,
            mixed_precision=config.mixed_precision,
        )

        self.model.real_score.model = fsdp2_warp_ovi(
            self.model.real_score.model,
            mixed_precision=config.mixed_precision,
        )

        self.model.fake_score.model = self.model.real_score.model

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )
        self.model.audio_vae = self.model.audio_vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)
        self.model.video_vae = self.model.video_vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)
        # Save pretrained model state_dicts to CPU
        # Auto resume configuration (needed for LoRA checkpoint loading)
        auto_resume = getattr(config, "auto_resume", True)  # Default to True

        # ================================= LoRA Configuration =================================

        if hasattr(config, 'adapter') and config.adapter is not None:
            self.is_lora_enabled = True
            self.lora_config = config.adapter
            
            if self.is_main_process:
                print(f"LoRA enabled with config: {self.lora_config}")
                print("Loading base model and applying LoRA before FSDP wrapping...")
            
            # 1. Load base model first (config.generator_ckpt) - before applying LoRA and FSDP
            base_checkpoint_path = getattr(config, "generator_ckpt", None)
            if base_checkpoint_path:
                if self.is_main_process:
                    print(f"Loading base model from {base_checkpoint_path} (before applying LoRA)")
                if base_checkpoint_path.endswith(".safetensors"):
                    base_checkpoint = load_file(base_checkpoint_path)
                elif base_checkpoint_path.endswith(".pth") or base_checkpoint_path.endswith(".pt"):
                    base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu")
                else:
                    raise ValueError(f"Unsupported checkpoint format: {base_checkpoint_path}")
                real_state_dict = copy.deepcopy(base_checkpoint)
                fake_state_dict = copy.deepcopy(base_checkpoint)
                # Load generator (directly; no key alignment needed since LoRA not applied yet)
                if "generator" in base_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {base_checkpoint_path}")
                    missing_keys, unexpected_keys = load_fusion_model_state_dict(
                        self.model.generator.model,
                        base_checkpoint["generator"],
                        strict=False,  # 允许部分匹配，因为可能有FSDP2相关的key差异
                        broadcast_from_rank0=True,
                    )
                    print(f"missing_keys: {missing_keys}")
                    print(f"unexpected_keys: {unexpected_keys}")
                    if self.is_main_process:
                        print("Generator weights loaded successfully")
                elif "model" in base_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {base_checkpoint_path}")
                    missing_keys, unexpected_keys = load_fusion_model_state_dict(
                        self.model.generator.model,
                        base_checkpoint["model"],
                        strict=False,  # 允许部分匹配，因为可能有FSDP2相关的key差异
                        broadcast_from_rank0=True,
                    )
                    print(f"missing_keys: {missing_keys}")
                    print(f"unexpected_keys: {unexpected_keys}")
                    if self.is_main_process:
                        print("Generator weights loaded successfully")
                else:
                    missing_keys, unexpected_keys = load_fusion_model_state_dict(
                        self.model.generator.model,
                        base_checkpoint,
                        strict=False,  # 允许部分匹配，因为可能有FSDP2相关的key差异
                        broadcast_from_rank0=True,
                    )
                    print(f"missing_keys: {missing_keys}")
                    print(f"unexpected_keys: {unexpected_keys}")
                # Load critic
                if "critic" in base_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained critic from {base_checkpoint_path}")
                    missing_keys, unexpected_keys = load_fusion_model_state_dict(
                        self.model.fake_score.model,
                        base_checkpoint["critic"],
                        strict=False,  # 允许部分匹配，因为可能有FSDP2相关的key差异
                        broadcast_from_rank0=True,
                    )
                    print(f"missing_keys: {missing_keys}")
                    print(f"unexpected_keys: {unexpected_keys}")
                    if self.is_main_process:
                        print("Critic weights loaded successfully")
                else:
                    if self.is_main_process:
                        print("Warning: Critic checkpoint not found in base model.")
            else:
                if self.is_main_process:
                    raise ValueError("No base model checkpoint specified for LoRA training.")
            
            # Load training step
            if "step" in base_checkpoint:
                self.step = base_checkpoint["step"]
                if self.is_main_process:
                    print(f"base_checkpoint step: {self.step}")
            else:
                if self.is_main_process:
                    print("Warning: Step not found in checkpoint, starting from step 0.")
            
            self._initialize_lora_params()

        if not self.is_lora_enabled:
            checkpoint_path = None
            checkpoint = None
            if getattr(config, "generator_ckpt", False):
                # Explicit checkpoint path provided
                checkpoint_path = config.generator_ckpt
                if self.is_main_process:
                    print(f"Using explicit checkpoint: {checkpoint_path}")

            if checkpoint_path:
                if self.is_main_process:
                    print(f"Loading checkpoint from {checkpoint_path}")
                if checkpoint_path.endswith(".safetensors"):
                    checkpoint = load_file(checkpoint_path)
                elif checkpoint_path.endswith(".pth") or checkpoint_path.endswith(".pt"):
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                else:
                    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
                fake_state_dict = copy.deepcopy(checkpoint)
                # Load generator
                if "generator" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {checkpoint_path}")
                    missing_keys, unexpected_keys = load_fusion_model_state_dict(
                        self.model.generator.model,
                        checkpoint["generator"],
                        strict=False,  # 允许部分匹配，因为可能有FSDP2相关的key差异
                        broadcast_from_rank0=True,
                    )
                    print(f"missing_keys: {missing_keys}")
                    print(f"unexpected_keys: {unexpected_keys}")
                elif "model" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {checkpoint_path}")
                    missing_keys, unexpected_keys = load_fusion_model_state_dict(
                        self.model.generator.model,
                        checkpoint["model"],
                        strict=False,  # 允许部分匹配，因为可能有FSDP2相关的key差异
                        broadcast_from_rank0=True,
                    )
                    print(f"missing_keys: {missing_keys}")
                    print(f"unexpected_keys: {unexpected_keys}")
                else:
                    missing_keys, unexpected_keys = load_fusion_model_state_dict(
                        self.model.generator.model,
                        checkpoint,
                        strict=False,  # 允许部分匹配，因为可能有FSDP2相关的key差异
                        broadcast_from_rank0=True,
                    )
                    print(f"missing_keys: {missing_keys}")
                    print(f"unexpected_keys: {unexpected_keys}")
            else:
                fake_state_dict = None
        missing_keys, unexpected_keys = load_fusion_model_state_dict(
            self.model.fake_score.model,
            fake_state_dict["model"] if isinstance(fake_state_dict, dict) and "model" in fake_state_dict else fake_state_dict,
            strict=False,  # 允许部分匹配，因为可能有FSDP2相关的key差异
            broadcast_from_rank0=True,
        )
        print(f"FAKE/REAL SCORE missing_keys: {missing_keys}")
        print(f"FAKE/REAL SCORE unexpected_keys: {unexpected_keys}")


        for name, module in self.model.fake_score.model.named_modules():
            if isinstance(module, LoRALayer):
                init.normal_(module.A, mean=0.0, std=0.02)
                init.zeros_(module.B)

        # Step 3: Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            if self.is_lora_enabled:
                if self.is_main_process:
                    print(f"EMA disabled in LoRA mode (LoRA provides efficient parameter updates without EMA)")
                self.generator_ema = None
            else:
                print(f"Setting up EMA with weight {ema_weight}")
                self.generator_ema = create_ema_model(self.model.generator, decay=ema_weight)

        self._reinitialize_dit_freqs(torch.device("cuda"))

        generator_trainable_params = [
            param for param in self.model.generator.parameters()
            if param.requires_grad
        ]
        critic_trainable_params = [
            param for param in self.model.fake_score.parameters()
            if param.requires_grad
        ]
        if self.is_main_process:
            generator_param_count = sum(p.numel() for p in generator_trainable_params)
            critic_param_count = sum(p.numel() for p in critic_trainable_params)
            print(f"[INFO] Generator optimizer parameter count: {generator_param_count}")
            print(f"[INFO] Critic optimizer parameter count: {critic_param_count}")

        self.generator_optimizer = torch.optim.AdamW(
            generator_trainable_params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            critic_trainable_params,
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        

        if self.config.distribution_loss == "dmd_switch":
            dataset = TwoTextDataset(config.data_path, config.switch_prompt_path)
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 6: Initialize the validation dataloader for visualization (fixed prompts)
        self.fixed_vis_batch = None
        self.vis_interval = getattr(config, "vis_interval", -1)
        if self.vis_interval > 0 and len(getattr(config, "vis_video_lengths", [])) > 0:
            # Determine validation data path
            val_data_path = getattr(config, "val_data_path", None) or config.data_path

            if self.config.distribution_loss == "dmd_switch":
                val_dataset = TwoTextDataset(val_data_path, config.val_switch_prompt_path)
            else:
                val_dataset = TextDataset(val_data_path)

            if dist.get_rank() == 0:
                print("VAL DATASET SIZE %d" % len(val_dataset))

            sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False)
            # streaming sampling to keep prompts fixed
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=getattr(config, "val_batch_size", 1),
                sampler=sampler,
                num_workers=8,
            )

            # Take the first batch as fixed visualization batch
            try:
                self.fixed_vis_batch = next(iter(val_dataloader))
            except StopIteration:
                self.fixed_vis_batch = None
            
            # ----------------------------------------------------------------------------------------------------------
            # Visualization settings
            # ----------------------------------------------------------------------------------------------------------
            # List of video lengths to visualize, e.g. [8, 16, 32]
            self.vis_video_lengths = getattr(config, "vis_video_lengths", [])

            if self.vis_interval > 0 and len(self.vis_video_lengths) > 0:
                self._setup_visualizer()

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        # Note: This should be done after potential resume to avoid accidentally deleting resumed EMA
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.previous_time = None
        
        # streaming training configuration
        self.streaming_training = getattr(config, "streaming_training", False)
        self.streaming_chunk_size = getattr(config, "streaming_chunk_size", 21)
        self.streaming_max_length = getattr(config, "streaming_max_length", 63)
        
        # Create streaming training model if enabled
        if self.streaming_training:
            self.streaming_model = StreamingTrainingModel(self.model, config)
            if self.is_main_process:
                print(f"streaming training enabled: chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}")
        else:
            self.streaming_model = None
        
        # streaming training state (simplified)
        self.streaming_active = False  # Whether we're currently in a sequence
        
        if self.is_main_process:
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            if self.gradient_accumulation_steps > 1:
                print(f"Effective batch size: {config.batch_size * self.gradient_accumulation_steps * self.world_size}")
            if self.streaming_training:
                print(f"streaming training enabled: chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}")
            if LOG_GPU_MEMORY:
                log_gpu_memory("After initialization", device=self.device, rank=dist.get_rank())

        checkpoint_dir = self.output_path
        keep_last_n = self.config.get('keep_last_n', 5)
        self.manager = FSDP2CheckpointManager(checkpoint_dir=checkpoint_dir,keep_last_n=keep_last_n)

    def _move_optimizer_to_device(self, optimizer, device):
        """Move optimizer state to the specified device."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def _get_switch_frame_index(self, max_length=None):
        if getattr(self.config, "switch_mode", "fixed") == "random":
            block = self.config.num_frame_per_block
            min_idx = self.config.min_switch_frame_index
            max_idx = self.config.max_switch_frame_index
            if min_idx == max_idx:
                switch_idx = min_idx
            else:
                choices = list(range(min_idx, max_idx, block))
                if max_length is not None:
                    choices = [choice for choice in choices if choice < max_length]
                
                if len(choices) == 0:
                    if max_length is not None:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")
                    else:
                        switch_idx = block
                else:
                    if dist.get_rank() == 0:
                        switch_idx = random.choice(choices)
                    else:
                        switch_idx = 0  # placeholder; will be overwritten by broadcast
                switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
                dist.broadcast(switch_idx_tensor, src=0)
                switch_idx = switch_idx_tensor.item()
        elif getattr(self.config, "switch_mode", "fixed") == "fixed":
            switch_idx = getattr(self.config, "fixed_switch_index", 21)
            if max_length is not None:
                assert max_length > switch_idx, f"max_length {max_length} is not greater than switch_idx {switch_idx}"
        elif getattr(self.config, "switch_mode", "fixed") == "random_choice":
            switch_choices = getattr(self.config, "switch_choices", [])
            if len(switch_choices) == 0:
                raise ValueError("switch_choices is empty")
            else:
                if max_length is not None:
                    switch_choices = [choice for choice in switch_choices if choice < max_length]
                    if len(switch_choices) == 0:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")
                
                if dist.get_rank() == 0:
                    switch_idx = random.choice(switch_choices)
                else:
                    switch_idx = 0
            switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
            dist.broadcast(switch_idx_tensor, src=0)
            switch_idx = switch_idx_tensor.item()
        else:
            raise ValueError(f"Invalid switch_mode: {getattr(self.config, 'switch_mode', 'fixed')}")
        return switch_idx


    def save(self):
        print("Start gathering distributed model states...")
        if self.is_lora_enabled:
            gen_lora_sd = self._gather_lora_state_dict(
                self.model.generator.model)
            crit_lora_sd = self._gather_lora_state_dict(
                self.model.fake_score.model)
            state_dict = {
                "generator_lora": gen_lora_sd,
                "critic_lora": crit_lora_sd,
                "step": self.step,
            }
            self.manager.save(state_dict=state_dict, step=self.step)
        else:
            self.manager.save(self.model.generator, self.generator_optimizer, self.generator_ema, step=self.step)
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 5 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        text_prompts = [self.formatter(text) for text in text_prompts]
        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        audio_shape = list(self.config.audio_shape)
        audio_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "video_unconditional_dict", None) or not getattr(self, "audio_unconditional_dict", None):
                video_unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.video_negative_prompt] * batch_size)
                video_unconditional_dict = {k: v.detach()
                                      for k, v in video_unconditional_dict.items()}
                audio_unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.audio_negative_prompt] * batch_size)
                audio_unconditional_dict = {k: v.detach()
                                      for k, v in audio_unconditional_dict.items()}
                self.audio_unconditional_dict = audio_unconditional_dict  # cache the audio_unconditional_dict
                self.video_unconditional_dict = video_unconditional_dict  # cache the video_unconditional_dict
            else:
                audio_unconditional_dict = self.audio_unconditional_dict
                video_unconditional_dict = self.video_unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            video_generator_loss, audio_generator_loss, generator_log_dict = self.model.generator_loss(
                video_shape=image_or_video_shape,
                audio_shape=audio_shape,
                video_conditional_dict=conditional_dict,
                audio_conditional_dict=conditional_dict,
                video_unconditional_dict=video_unconditional_dict,
                audio_unconditional_dict=audio_unconditional_dict,
                video_clean_latent=None,
                audio_clean_latent=None,
                initial_latent=None
            )

            # Scale loss for gradient accumulation and backward
            scaled_generator_loss = (video_generator_loss * self.config.video_generator_loss_weight + audio_generator_loss * self.config.audio_generator_loss_weight) / self.gradient_accumulation_steps
            scaled_generator_loss.backward()
            if LOG_GPU_MEMORY:
                log_gpu_memory("After train_generator backward pass", device=self.device, rank=dist.get_rank())
            # Return original loss for logging
            generator_log_dict.update({"video_generator_loss": video_generator_loss,
                                       "audio_generator_loss": audio_generator_loss,
                                       "generator_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation

            return generator_log_dict
        else:
            generator_log_dict = {}


        video_critic_loss, audio_critic_loss, critic_log_dict = self.model.critic_loss(
            video_shape=image_or_video_shape,
            audio_shape=audio_shape,
            video_conditional_dict=conditional_dict,
            audio_conditional_dict=conditional_dict,
            video_unconditional_dict=video_unconditional_dict,
            audio_unconditional_dict=audio_unconditional_dict,
            video_clean_latent=None,
            audio_clean_latent=None,
            initial_latent=None
        )

        # Scale loss for gradient accumulation and backward
        scaled_critic_loss = (video_critic_loss * self.config.video_critic_loss_weight + audio_critic_loss * self.config.audio_critic_loss_weight) / self.gradient_accumulation_steps
        scaled_critic_loss.backward()
        if LOG_GPU_MEMORY:
            log_gpu_memory("After train_critic backward pass", device=self.device, rank=dist.get_rank())
        # Return original loss for logging
        critic_log_dict.update({"video_critic_loss": video_critic_loss,
                                "audio_critic_loss": audio_critic_loss,
                                "critic_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation

        return critic_log_dict

    def generate_video(self, pipeline, num_frames, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, num_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, num_frames, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )
        with torch.no_grad():
            video, _ = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
            )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        pipeline.vae.model.clear_cache()
        return current_video
    
    def generate_video_with_ovi(self, pipeline, num_frames, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            video_sampled_noise = torch.randn(
                [batch_size, num_frames - 1, 48, 60, 60],
                device="cuda",
                dtype=self.dtype
            )
            audio_sampled_noise = torch.randn(
                [batch_size, 5 * (num_frames - 1), 20],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            video_sampled_noise = torch.randn(
                [batch_size, num_frames, 48, 60, 60],
                device="cuda",
                dtype=self.dtype
            )
            audio_sampled_noise = torch.randn(
                [batch_size, 5 * (num_frames), 20],
                device="cuda",
                dtype=self.dtype
            )
        with torch.no_grad():
            video, _, _, _ = pipeline.inference(
                video_noise=video_sampled_noise,
                audio_noise=audio_sampled_noise,
                text_prompts=prompts,
                return_latents=True,
            )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        pipeline.vae.model.clear_cache()
        return current_video
    
    def generate_video_with_switch(self, pipeline, num_frames, prompts, switch_prompts, switch_frame_index, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, num_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, num_frames, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )
        with torch.no_grad():
            video, _ = pipeline.inference(
                noise=sampled_noise,
                text_prompts_first=prompts,
                text_prompts_second=switch_prompts,
                switch_frame_index=switch_frame_index,
                return_latents=True
            )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        pipeline.vae.model.clear_cache()
        return current_video

    def start_new_sequence(self):
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] start_new_sequence called")
        
        if LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: Before start_new_sequence", device=self.device, rank=dist.get_rank())
        
        # Fetch a new batch
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] start_new_sequence: fetch new batch")
        batch = next(self.dataloader)

        # Prepare conditional information
        text_prompts = batch["prompts"]
        if self.config.i2v:
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] Setting up sequence: batch_size={batch_size}, i2v={self.config.i2v}")
            print(f"[SeqTrain-Trainer] image_or_video_shape={image_or_video_shape}")
        
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Created and cached conditional_dict")
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] Created and cached unconditional_dict")
            else:
                unconditional_dict = self.unconditional_dict
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: After text encoding", device=self.device, rank=dist.get_rank())
        
        if self.streaming_model.possible_max_length is not None:
            # Ensure all processes choose the same length
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    import random
                    selected_idx = random.randint(0, len(self.streaming_model.possible_max_length) - 1)
                else:
                    selected_idx = 0
                selected_idx_tensor = torch.tensor(selected_idx, device=self.device, dtype=torch.int32)
                dist.broadcast(selected_idx_tensor, src=0)
                selected_idx = selected_idx_tensor.item()
            else:
                import random
                selected_idx = random.randint(0, len(self.streaming_model.possible_max_length) - 1)
            
            temp_max_length = self.streaming_model.possible_max_length[selected_idx]
        else:
            temp_max_length = self.streaming_model.max_length
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Model] Selected temporary max length: {temp_max_length} (from {self.streaming_model.possible_max_length})")
        

        # Handle DMD Switch related information
        switch_conditional_dict = None
        switch_frame_index = None
        if isinstance(self.model, DMDSwitch) and "switch_prompts" in batch:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Processing DMDSwitch info")
                
            with torch.no_grad():
                switch_conditional_dict = self.model.text_encoder(
                    text_prompts=batch["switch_prompts"]
                )
            switch_frame_index = self._get_switch_frame_index(temp_max_length)
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] switch_frame_index={switch_frame_index}")
            
            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                log_gpu_memory(f"streaming Training: After switch text encoding", device=self.device, rank=dist.get_rank())
        
        # Set up the sequence
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] Calling streaming_model.setup_sequence")
            
        self.streaming_model.setup_sequence(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            initial_latent=image_latent,
            switch_conditional_dict=switch_conditional_dict,
            switch_frame_index=switch_frame_index,
            temp_max_length=temp_max_length,
        )
        
        self.streaming_active = True
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] streaming training sequence setup completed")
            
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: After sequence setup", device=self.device, rank=dist.get_rank())

    def fwdbwd_one_step_streaming(self, train_generator):
        """Forward/backward pass using the new StreamingTrainingModel for serialized training"""
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 5 == 0:
            torch.cuda.empty_cache()

        # If no active sequence, start a new one
        if not self.streaming_active:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] No active sequence, starting new one")
            self.start_new_sequence()
        
        # Check whether we can generate more chunks
        if not self.streaming_model.can_generate_more():
            # Current sequence is finished; start a new one
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Current sequence completed, starting new one")
            self.streaming_active = False
            self.start_new_sequence()
        
        self.kv_cache_before_generator_rollout = None
        self.kv_cache_after_generator_rollout = None
        self.kv_cache_after_generator_backward = None
        self.kv_cache_before_critic_rollout = None
        self.kv_cache_after_critic_rollout = None
        self.kv_cache_after_critic_backward = None
        
        if train_generator:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Training generator: generating next chunk")

            train_first_chunk = getattr(self.config, "train_first_chunk", False)
            if train_first_chunk:
                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=True)
            else:
                current_seq_length = self.streaming_model.state.get("current_length")
                if current_seq_length == 0:
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}, generate first chunk")
                    generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)

                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=True)
            
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}")

            # Compute generator loss
            generator_loss, generator_log_dict = self.streaming_model.compute_generator_loss(
                chunk=generated_chunk,
                chunk_info=chunk_info
            )

            # Scale loss for gradient accumulation and backward
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[DEBUG] Scaled generator loss: {scaled_generator_loss.item()}")

            try:
                scaled_generator_loss.backward()
            except RuntimeError as e:
                raise

            generator_log_dict.update({
                "generator_loss": generator_loss,
                "generator_grad_norm": torch.tensor(0.0, device=self.device),
            })
            
            return generator_log_dict
        else:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Training critic: generating next chunk")

            train_first_chunk = getattr(self.config, "train_first_chunk", False)
            if train_first_chunk:
                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)
            else:
                current_seq_length = self.streaming_model.state.get("current_length")
                if current_seq_length == 0:
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}, generate first chunk")
                    generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)

                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)
            
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}")

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Generated chunk shape: {generated_chunk.shape}")
                print(f"[SeqTrain-Trainer] Generated chunk requires_grad: {generated_chunk.requires_grad}")
            
            if generated_chunk.requires_grad:
                generated_chunk = generated_chunk.detach()

            # Compute critic loss
            critic_loss, critic_log_dict = self.streaming_model.compute_critic_loss(
                chunk=generated_chunk,
                chunk_info=chunk_info
            )
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Critic loss: {critic_loss.item()}")
            
            # Scale loss for gradient accumulation and backward
            scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
            scaled_critic_loss.backward()
            
            critic_log_dict.update({
                "critic_loss": critic_loss,
                "critic_grad_norm": torch.tensor(0.0, device=self.device),
            })
            
            return critic_log_dict

    def train(self):
        start_step = self.step
        try:
            while True:
                # Check if we should train generator on this optimization step
                TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
                if LOG_GPU_MEMORY:
                    log_gpu_memory(f"Before training", device=self.device, rank=dist.get_rank())
                
                if dist.get_rank() == 0 and DEBUG:
                    print(f"[Debug] Step {self.step}: switch_mode={getattr(self.config,'switch_mode','fixed')}")

                if self.streaming_training:
                    # Zero-out all optimizer gradients
                    if TRAIN_GENERATOR:
                        self.generator_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    
                    # Whole-cycle gradient accumulation loop
                    accumulated_generator_logs = []
                    accumulated_critic_logs = []
                    
                    for accumulation_step in range(self.gradient_accumulation_steps):
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Whole-cycle accumulation step {accumulation_step + 1}/{self.gradient_accumulation_steps}")
                        
                        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and accumulation_step == 0:
                            log_gpu_memory(f"streaming Training Step {self.step}: Before whole-cycle forward/backward", device=self.device, rank=dist.get_rank())
                        
                        # Train generator (if needed)
                        if TRAIN_GENERATOR:
                            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                                print(f"[SeqTrain-Trainer] Accumulation step {accumulation_step + 1}: Training generator")
                            extra_gen = self.fwdbwd_one_step_streaming(True)
                            accumulated_generator_logs.append(extra_gen)
                        
                        # Train critic
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Accumulation step {accumulation_step + 1}: Training critic")
                        extra_crit = self.fwdbwd_one_step_streaming(False)
                        accumulated_critic_logs.append(extra_crit)
                        
                        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and accumulation_step == 0:
                            log_gpu_memory(f"streaming Training Step {self.step}: After whole-cycle forward/backward", device=self.device, rank=dist.get_rank())
                    
                    # Compute grad norm and update parameters
                    if TRAIN_GENERATOR:
                        generator_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), self.max_grad_norm_generator)
                        generator_log_dict = merge_dict_list(accumulated_generator_logs)
                        generator_log_dict["generator_grad_norm"] = generator_grad_norm
                        
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Generator training completed, grad_norm={generator_grad_norm.item()}")
                        
                        self.generator_optimizer.step()
                        if self.generator_ema is not None:
                            self.generator_ema.update(self.model.generator)
                    else:
                        generator_log_dict = {}
                    
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.fake_score.parameters(), self.max_grad_norm_critic)
                    critic_log_dict = merge_dict_list(accumulated_critic_logs)
                    critic_log_dict["critic_grad_norm"] = critic_grad_norm
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] Critic training completed, grad_norm={critic_grad_norm.item()}")
                    
                    self.critic_optimizer.step()
                    
                    if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                        log_gpu_memory(f"streaming Training Step {self.step}: After optimizer steps", device=self.device, rank=dist.get_rank())
                    
                    # Increase step count
                    self.step += 1
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] streaming training step completed: step={self.step}")
                        if hasattr(self, 'streaming_model') and self.streaming_model is not None:
                            current_seq_length = self.streaming_model.state.get("current_length", 0)
                            print(f"[SeqTrain-Trainer] Current sequence length: {current_seq_length}/{self.streaming_model.max_length}")
                            
                    if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                        log_gpu_memory(f"streaming Training Step {self.step}: Training step completed", device=self.device, rank=dist.get_rank())
                else:
                    if TRAIN_GENERATOR:
                        self.generator_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    
                    # Whole-cycle gradient accumulation loop
                    accumulated_generator_logs = []
                    accumulated_critic_logs = []
                    
                    for accumulation_step in range(self.gradient_accumulation_steps):
                        batch = next(self.dataloader)
                        
                        # Train generator (if needed)
                        if TRAIN_GENERATOR:
                            extra_gen = self.fwdbwd_one_step(batch, True)
                            accumulated_generator_logs.append(extra_gen)
                        
                        # Train critic
                        extra_crit = self.fwdbwd_one_step(batch, False)
                        accumulated_critic_logs.append(extra_crit)
                    
                    # Compute grad norm and update parameters
                    if TRAIN_GENERATOR:
                        generator_grad_norm = clip_grad_norm_for_fsdp2_offload(self.model.generator.parameters(), self.max_grad_norm_generator)
                        generator_log_dict = merge_dict_list(accumulated_generator_logs)
                        generator_log_dict["generator_grad_norm"] = generator_grad_norm
                        
                        self.generator_optimizer.step()
                        if self.generator_ema is not None:
                            self.generator_ema.update(self.model.generator)
                    else:
                        generator_log_dict = {}
                    
                    critic_grad_norm = clip_grad_norm_for_fsdp2_offload(self.model.fake_score.parameters(), self.max_grad_norm_critic)
                    critic_log_dict = merge_dict_list(accumulated_critic_logs)
                    critic_log_dict["critic_grad_norm"] = critic_grad_norm
                    
                    self.critic_optimizer.step()

                    # Increment the step since we finished gradient update
                    self.step += 1

                # Create EMA params (if not already created)
                if (self.step >= self.config.ema_start_step) and \
                        (self.generator_ema is None) and (self.config.ema_weight > 0):
                    if not self.is_lora_enabled:
                        self.generator_ema = create_ema_model(self.model.generator, decay=self.config.ema_weight)
                        if self.is_main_process:
                            print(f"EMA created at step {self.step} with weight {self.config.ema_weight}")
                    else:
                        if self.is_main_process:
                            print(f"EMA creation skipped at step {self.step} (disabled in LoRA mode)")

                # Save the model
                if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()

                # Logging
                if self.is_main_process:
                    wandb_loss_dict = {}
                    if TRAIN_GENERATOR and generator_log_dict:
                        wandb_loss_dict.update(
                            {
                                "video_generator_loss": generator_log_dict["video_generator_loss"].mean().item(),
                                "audio_generator_loss": generator_log_dict["audio_generator_loss"].mean().item(),
                                "generator_grad_norm": generator_log_dict["generator_grad_norm"],
                                "dmdtrain_video_gradient_norm": generator_log_dict["dmdtrain_video_gradient_norm"].mean().item(),
                                "dmdtrain_audio_gradient_norm": generator_log_dict["dmdtrain_audio_gradient_norm"].mean().item()
                            }
                        )


                    wandb_loss_dict.update(
                        {
                            "video_critic_loss": critic_log_dict["video_critic_loss"].mean().item(),
                            "audio_critic_loss": critic_log_dict["audio_critic_loss"].mean().item(),
                            "critic_grad_norm": critic_log_dict["critic_grad_norm"],
                        }
                    )
                    if not self.disable_wandb:
                        wandb.log(wandb_loss_dict, step=self.step)

                if self.step % self.config.gc_interval == 0:
                    if dist.get_rank() == 0:
                        logging.info("DistGarbageCollector: Running GC.")
                    gc.collect()
                    torch.cuda.empty_cache()

                if self.is_main_process:
                    current_time = time.time()
                    iteration_time = 0 if self.previous_time is None else current_time - self.previous_time
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": iteration_time}, step=self.step)
                    self.previous_time = current_time
                    # Log training progress
                    if TRAIN_GENERATOR and generator_log_dict:
                        print(f"step {self.step}, per iteration time {iteration_time}, video generator_loss {generator_log_dict['video_generator_loss'].mean().item()}, audio generator_loss {generator_log_dict['audio_generator_loss'].mean().item()}, generator_grad_norm {generator_log_dict['generator_grad_norm']}, video_critic_loss {critic_log_dict['video_critic_loss'].mean().item()}, audio_critic_loss {critic_log_dict['audio_critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm']}")
                    else:
                        print(f"step {self.step}, per iteration time {iteration_time}, video_critic_loss {critic_log_dict['video_critic_loss'].mean().item()}, audio_critic_loss {critic_log_dict['audio_critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm']}")

                # ---------------------------------------- Visualization ---------------------------------------------------

                if self.vis_interval > 0 and (self.step % self.vis_interval == 0):

                    try:
                        self._visualize()
                    except Exception as e:
                        print(f"[Warning] Visualization failed at step {self.step}: {e}")
                
                if self.step > self.config.max_iters:
                    break

        
        except Exception as e:
            if self.is_main_process:
                print(f"[ERROR] Training crashed at step {self.step} with exception: {e}")
                print(f"[ERROR] Exception traceback:", flush=True)
                import traceback
                traceback.print_exc()
        finally:
            # Clean up resources
            pass


    def _gather_lora_state_dict(self, lora_model):
        "On rank-0, gather FULL_STATE_DICT, then filter only LoRA weights"
        from torch.distributed.checkpoint.state_dict import get_model_state_dict
        state_dict = get_model_state_dict(lora_model)
        new_lora_state_dict = {}
        for key, value in state_dict.items():
            if ".A" in key or ".B" in key:
                new_lora_state_dict[key] = value
        return new_lora_state_dict
    
    # --------------------------------------------------------------------------------------------------------------
    # Visualization helpers
    # --------------------------------------------------------------------------------------------------------------

    def _setup_visualizer(self):
        """Initialize the inference pipeline for visualization on CPU, to be moved to GPU only when needed."""

        # Choose pipeline class depending on causal flag
        if 'switch' in self.config.distribution_loss:
            self.vis_pipeline = SwitchCausalInferencePipeline(
                args=self.config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae)
        elif 'ovi' in self.config.distribution_loss:
            self.vis_pipeline = CausalInferenceOviPipeline(
                args=self.config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                audio_vae=self.model.audio_vae,
                video_vae=self.model.video_vae)

        else:
            self.vis_pipeline = CausalInferencePipeline(
                args=self.config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae)

        # Visualization output directory (default: <logdir>/vis)
        self.vis_output_dir = os.path.join(os.path.dirname(self.output_path), "vis")
        os.makedirs(self.vis_output_dir, exist_ok=True)
        if self.config.vis_ema:
            raise NotImplementedError("Visualization with EMA is not implemented")

    def _visualize(self):
        """Generate and save sample videos to monitor training progress."""
        if self.vis_interval <= 0 or not hasattr(self, "vis_pipeline"):
            return

        # Use the fixed batch of prompts/images prepared from val_loader
        if not getattr(self, "fixed_vis_batch", None):
            print("[Warning] No fixed validation batch available for visualization.")
            return
        step_vis_dir = os.path.join(self.vis_output_dir, f"step_{self.step:07d}")
        os.makedirs(step_vis_dir, exist_ok=True)
        batch = self.fixed_vis_batch
        if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
            prompts = batch["prompts"]
            switch_prompts = batch["switch_prompts"]
            switch_frame_index = self._get_switch_frame_index()
        else:
            prompts = batch["prompts"]

        image = None
        if self.config.i2v and ("image" in batch):
            image = batch["image"]

        # Prepare model mode info for filename
        mode_info = ""
        if self.is_lora_enabled:
            mode_info = "_lora"
            if self.is_main_process:
                print(f"Generating videos in LoRA mode (step {self.step})")
        
        for vid_len in self.vis_video_lengths:
            print(f"Generating video of length {vid_len}")
            if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
                videos = self.generate_video_with_switch(self.vis_pipeline, vid_len, prompts, switch_prompts, switch_frame_index, image=image)
            elif isinstance(self.vis_pipeline, CausalInferenceOviPipeline):
                videos = self.generate_video_with_ovi(self.vis_pipeline, vid_len, prompts, image=image)
            else:
                videos = self.generate_video(self.vis_pipeline, vid_len, prompts, image=image)

            # Save each sample
            for idx, video_np in enumerate(videos):
                if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
                    video_name = f"step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}_switch_frame_{switch_frame_index}.mp4"
                else:
                    video_name = f"step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}.mp4"
                out_path = os.path.join(
                    step_vis_dir,
                    video_name,
                )
                video_tensor = torch.from_numpy(video_np.astype("uint8"))
                write_video(out_path, video_tensor, fps=16)

            # After saving current length videos, release related tensors to reduce peak memory
            del videos, video_np, video_tensor  # type: ignore
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        import gc
        gc.collect()

    def _reinitialize_dit_freqs(self, target_device: torch.device):
        """重新初始化DiT模型的freqs tensor到指定设备"""
        # 获取DiT模型的配置
        vid_meta = self.model.generator.model.branch_meta["vid"]
        video_dim = vid_meta["dim"]
        video_num_heads = vid_meta["num_heads"]
        video_d = video_dim // video_num_heads
        audio_meta = self.model.generator.model.branch_meta["audio"]
        audio_dim = audio_meta["dim"]
        audio_num_heads = audio_meta["num_heads"]
        audio_d = audio_dim // audio_num_heads
        audio_temporal_rope_scaling_factor = audio_meta["temporal_rope_scaling_factor"]
        # 导入rope参数函数
        from ovi.modules.model import rope_params
        video_freqs = torch.cat([
            rope_params(1024, video_d - 4 * (video_d // 6)),
            rope_params(1024, 2 * (video_d // 6)),
            rope_params(1024, 2 * (video_d // 6))
        ], dim=1)
        audio_freqs = rope_params(1024, audio_d - 4 * (audio_d // 6), freqs_scaling=audio_temporal_rope_scaling_factor)

        # 移动到目标设备并注册为buffer
        video_freqs = video_freqs.to(device=target_device)
        audio_freqs = audio_freqs.to(device=target_device)
        if dist.is_initialized() and dist.get_rank() == 0:
            print("Video freqs shape: ", video_freqs.shape)
            print("Audio freqs shape: ", audio_freqs.shape)
        # 确保freqs被正确注册为buffer而不是参数
        if hasattr(self.model.generator.model, 'vid_freqs'):
            delattr(self.model.generator.model, 'vid_freqs')
        if hasattr(self.model.generator.model, 'audio_freqs'):
            delattr(self.model.generator.model, 'audio_freqs')
        if hasattr(self.model.fake_score.model, 'vid_freqs'):
            delattr(self.model.fake_score.model, 'vid_freqs')
        if hasattr(self.model.fake_score.model, 'audio_freqs'):
            delattr(self.model.fake_score.model, 'audio_freqs')
        self.model.generator.model.register_buffer('vid_freqs', video_freqs, persistent=False)
        self.model.generator.model.register_buffer('audio_freqs', audio_freqs, persistent=False)
        self.model.fake_score.model.register_buffer('vid_freqs', video_freqs, persistent=False)
        self.model.fake_score.model.register_buffer('audio_freqs', audio_freqs, persistent=False)


    def _initialize_lora_params(self):
        """手动初始化 LoRA 参数（在 to_empty 后）"""
        print("Initializing LoRA parameters after to_empty...")
        for name, module in self.model.generator.model.named_modules():
            if isinstance(module, LoRALayer):
                init.normal_(module.A, mean=0.0, std=0.2)
                init.zeros_(module.B)

        for name, module in self.model.fake_score.model.named_modules():
            if isinstance(module, LoRALayer):
                init.normal_(module.A, mean=0.0, std=0.2)
                init.zeros_(module.B)

    def _load_sharded_weights(self, model_path):
        """
        加载分片权重文件（支持HuggingFace格式）
        
        Returns:
            dict: 合并后的状态字典，如果失败则返回None
        """
        import re
        from safetensors.torch import load_file
        import glob
        # 查找所有相关文件
        bin_files = glob.glob(os.path.join(model_path, "*.bin"))
        pth_files = glob.glob(os.path.join(model_path, "*.pth"))
        safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        
        all_files = bin_files + pth_files + safetensor_files
        exclude_keywords = ['vae', 't5', 'encoder', 'tokenizer']
        
        # 过滤掉不相关的文件
        model_files = []
        for f in all_files:
            basename = os.path.basename(f).lower()
            if not any(keyword in basename for keyword in exclude_keywords):
                model_files.append(f)
        
        if not model_files:
            print("[WARNING] No model files found")
            return None
        
        # 检测是否为分片文件（HuggingFace格式: diffusion_pytorch_model-00001-of-00006.safetensors）
        sharded_files = []
        pattern = r'(.+)-(\d+)-of-(\d+)\.(safetensors|bin|pth)$'
        
        for f in model_files:
            basename = os.path.basename(f)
            match = re.search(pattern, basename)
            if match:
                prefix, shard_idx, total_shards, ext = match.groups()
                sharded_files.append((f, int(shard_idx), int(total_shards), prefix, ext))
        
        if sharded_files:
            # 处理分片文件
            print(f"[INFO] Found {len(sharded_files)} sharded model files")
            
            # 按分片索引排序
            sharded_files.sort(key=lambda x: x[1])
            
            # 验证分片完整性
            total_shards = sharded_files[0][2]
            prefix = sharded_files[0][3]
            ext = sharded_files[0][4]
            
            if len(sharded_files) != total_shards:
                print(f"[WARNING] Expected {total_shards} shards, but found {len(sharded_files)}")
            
            # 加载所有分片并合并
            merged_state_dict = {}
            print(f"[INFO] Loading {len(sharded_files)} shards...")
            
            for file_path, shard_idx, _, _, file_ext in sharded_files:
                print(f"[INFO] Loading shard {shard_idx}/{total_shards}: {os.path.basename(file_path)}")
                
                try:
                    if file_ext == 'safetensors':
                        shard_dict = load_file(file_path)
                    else:
                        shard_dict = torch.load(file_path, map_location='cpu')
                    
                    # 合并到总的状态字典中
                    for key, value in shard_dict.items():
                        if key in merged_state_dict:
                            print(f"[WARNING] Duplicate key found: {key}")
                        merged_state_dict[key] = value
                    
                    print(f"[INFO] Shard {shard_idx} loaded: {len(shard_dict)} keys")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to load shard {shard_idx}: {e}")
                    return None
            
            print(f"[INFO] Successfully merged {len(sharded_files)} shards into {len(merged_state_dict)} total keys")
            return merged_state_dict
            
        else:
            # 没有分片文件，尝试加载单个文件
            for f in model_files:
                basename = os.path.basename(f).lower()
                if 'model' in basename:
                    print(f"[INFO] Found single model file: {os.path.basename(f)}")
                    
                    try:
                        if f.endswith('.safetensors'):
                            return load_file(f)
                        else:
                            return torch.load(f, map_location='cpu')
                    except Exception as e:
                        print(f"[ERROR] Failed to load {f}: {e}")
                        continue
            
            # 如果都没找到合适的，使用第一个文件
            if model_files:
                f = model_files[0]
                print(f"[INFO] Using fallback file: {os.path.basename(f)}")
                try:
                    if f.endswith('.safetensors'):
                        return load_file(f)
                    else:
                        return torch.load(f, map_location='cpu')
                except Exception as e:
                    print(f"[ERROR] Failed to load {f}: {e}")
            
            return None
