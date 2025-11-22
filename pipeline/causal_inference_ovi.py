# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple
import torch
import os
import time

# 引入 Ovi 相关的 Wrapper
from utils.ovi_wrapper import OviDiffusionWrapper, OviTextEncoder, OviVideoVAEWrapper, OviAudioVAEWrapper

from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation, log_gpu_memory
from utils.debug_option import DEBUG
import torch.distributed as dist

class CausalInferenceOviPipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            video_vae=None,
            audio_vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        if DEBUG:
            print(f"args.model_kwargs: {args.model_kwargs}")
        
        # 使用 OviDiffusionWrapper
        self.generator = OviDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        
        # 使用 OviTextEncoder
        self.text_encoder = OviTextEncoder() if text_encoder is None else text_encoder
        
        # 初始化 Video 和 Audio VAE
        self.video_vae = OviVideoVAEWrapper() if video_vae is None else video_vae
        self.audio_vae = OviAudioVAEWrapper() if audio_vae is None else audio_vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Hard code for Wan2.2-TI2V-5B (OVI specific settings)
        self.num_transformer_blocks = 30
        self.video_frame_seq_length = 900  # Video seq len per frame
        self.audio_frame_seq_length = 1    # Audio seq len per frame (usually smaller)

        # Cache placeholders
        self.video_kv_cache1 = None
        self.video_crossattn_cache = None
        self.audio_kv_cache1 = None
        self.audio_crossattn_cache = None

        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = args.model_kwargs.local_attn_size

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference (OVI) with {self.num_frame_per_block} frames per block")

        # Ovi wrapper handles num_frame_per_block logic internally usually, but we set it if attribute exists
        if self.num_frame_per_block > 1 and hasattr(self.generator.model, "num_frame_per_block"):
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform inference on the given noise and text prompts for both Video and Audio.
        Inputs:
            video_noise: [batch_size, video_frames, C, H, W]
            audio_noise: [batch_size, audio_frames, D]
            text_prompts: List[str]
        Outputs:
            video: generated video tensor
            audio: generated audio tensor
        """
        video_batch_size, video_num_output_frames, video_num_channels, video_height, video_width = video_noise.shape
        audio_batch_size, audio_num_output_frames, audio_num_channels = audio_noise.shape
        
        # Ensure Alignment: 1 video frame typically aligns with 5 audio latents in this architecture
        assert video_num_output_frames % self.num_frame_per_block == 0
        assert audio_num_output_frames % (self.num_frame_per_block * 5) == 0
        
        num_blocks = video_num_output_frames // self.num_frame_per_block

        # Text Encoding
        # OviTextEncoder returns a dict, usually used for both video and audio context
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        
        # In OVI pipeline, we usually pass the same dict for both if prompts are shared
        video_conditional_dict = conditional_dict
        audio_conditional_dict = conditional_dict

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output_device = torch.device('cpu') if low_memory else video_noise.device
        
        # Prepare Output Tensors
        video_output = torch.zeros(
            [video_batch_size, video_num_output_frames, video_num_channels, video_height, video_width],
            device=output_device,
            dtype=video_noise.dtype
        )
        audio_output = torch.zeros(
            [audio_batch_size, audio_num_output_frames, audio_num_channels],
            device=output_device,
            dtype=audio_noise.dtype
        )

        # Set up profiling
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache (Both Video and Audio)
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        
        self._initialize_kv_cache(
            batch_size=video_batch_size,
            dtype=video_noise.dtype,
            device=video_noise.device,
            local_attn_cfg=local_attn_cfg
        )
        self._initialize_crossattn_cache(
            batch_size=video_batch_size,
            dtype=video_noise.dtype,
            device=video_noise.device
        )

        video_current_start_frame = 0
        audio_current_start_frame = 0
        
        # Set Attention Size
        if not (isinstance(self.local_attn_size, (list, tuple)) or (hasattr(self.local_attn_size, "__iter__") and not isinstance(self.local_attn_size, (str, bytes)))):
            self.generator.model.local_attn_size = int(self.local_attn_size)
            self._set_all_modules_max_attention_size(int(self.local_attn_size))
            print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 2: Temporal denoising loop
        video_all_num_frames = [self.num_frame_per_block] * num_blocks
        audio_all_num_frames = [self.num_frame_per_block * 5] * num_blocks
        
        for video_current_num_frames, audio_current_num_frames in zip(video_all_num_frames, audio_all_num_frames):
            if profile:
                block_start.record()

            video_noisy_input = video_noise[:, video_current_start_frame:video_current_start_frame + video_current_num_frames]
            audio_noisy_input = audio_noise[:, audio_current_start_frame:audio_current_start_frame + audio_current_num_frames]

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                
                # Dynamic attention size scheduling
                if isinstance(self.local_attn_size, (list, tuple)) or (hasattr(self.local_attn_size, "__iter__") and not isinstance(self.local_attn_size, (str, bytes))):
                    self._set_all_modules_max_attention_size(int(self.local_attn_size[index]))

                # Construct Timesteps
                video_timestep = torch.ones(
                    [video_batch_size, video_current_num_frames],
                    device=video_noise.device,
                    dtype=torch.int64) * current_timestep
                audio_timestep = torch.ones(
                    [audio_batch_size, audio_current_num_frames],
                    device=audio_noise.device,
                    dtype=torch.int64) * current_timestep

                # Inference Step
                if index < len(self.denoising_step_list) - 1:
                    with torch.no_grad():
                        _, _, vid_pred_x0, audio_pred_x0 = self.generator(
                            noisy_video=video_noisy_input,
                            noisy_audio=audio_noisy_input,
                            video_conditional_dict=video_conditional_dict,
                            audio_conditional_dict=audio_conditional_dict,
                            video_timestep=video_timestep,
                            audio_timestep=audio_timestep,
                            video_kv_cache=self.video_kv_cache1,
                            audio_kv_cache=self.audio_kv_cache1,
                            video_crossattn_cache=self.video_crossattn_cache,
                            audio_crossattn_cache=self.audio_crossattn_cache,
                            video_current_start=video_current_start_frame * self.video_frame_seq_length,
                            audio_current_start=audio_current_start_frame * self.audio_frame_seq_length,
                        )

                        next_timestep = self.denoising_step_list[index + 1]
                        
                        # Add noise for next step (Scheduler)
                        video_noisy_input = self.scheduler.add_noise(
                            vid_pred_x0.flatten(0, 1),
                            torch.randn_like(vid_pred_x0.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [video_batch_size * video_current_num_frames], device=video_noise.device, dtype=torch.long)
                        ).unflatten(0, vid_pred_x0.shape[:2])
                        
                        audio_noisy_input = self.scheduler.add_noise(
                            audio_pred_x0.flatten(0, 1),
                            torch.randn_like(audio_pred_x0.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [audio_batch_size * audio_current_num_frames], device=audio_noise.device, dtype=torch.long)
                        ).unflatten(0, audio_pred_x0.shape[:2])
                else:
                    # Final step to get real output
                    with torch.no_grad():
                        _, _, vid_pred_x0, audio_pred_x0 = self.generator(
                            noisy_video=video_noisy_input,
                            noisy_audio=audio_noisy_input,
                            video_conditional_dict=video_conditional_dict,
                            audio_conditional_dict=audio_conditional_dict,
                            video_timestep=video_timestep,
                            audio_timestep=audio_timestep,
                            video_kv_cache=self.video_kv_cache1,
                            audio_kv_cache=self.audio_kv_cache1,
                            video_crossattn_cache=self.video_crossattn_cache,
                            audio_crossattn_cache=self.audio_crossattn_cache,
                            video_current_start=video_current_start_frame * self.video_frame_seq_length,
                            audio_current_start=audio_current_start_frame * self.audio_frame_seq_length,
                        )

            # Step 2.2: Record output
            video_output[:, video_current_start_frame:video_current_start_frame + video_current_num_frames] = vid_pred_x0.to(video_output.device)
            audio_output[:, audio_current_start_frame:audio_current_start_frame + audio_current_num_frames] = audio_pred_x0.to(audio_output.device)

            # Step 2.3: Rerun with timestep zero (Context Noise) to update KV cache using clean context
            video_context_timestep = torch.ones_like(video_timestep) * self.args.context_noise
            audio_context_timestep = torch.ones_like(audio_timestep) * self.args.context_noise
            
            # Add context noise (usually very small or zero)
            video_denoised_pred = self.scheduler.add_noise(
                vid_pred_x0.flatten(0, 1),
                torch.randn_like(vid_pred_x0.flatten(0, 1)),
                video_context_timestep * torch.ones(
                    [video_batch_size * video_current_num_frames], device=video_noise.device, dtype=torch.long)
            ).unflatten(0, vid_pred_x0.shape[:2])
            
            audio_denoised_pred = self.scheduler.add_noise(
                audio_pred_x0.flatten(0, 1),
                torch.randn_like(audio_pred_x0.flatten(0, 1)),
                audio_context_timestep * torch.ones(
                    [audio_batch_size * audio_current_num_frames], device=audio_noise.device, dtype=torch.long)
            ).unflatten(0, audio_pred_x0.shape[:2])

            with torch.no_grad():
                self.generator(
                    noisy_video=video_denoised_pred,
                    noisy_audio=audio_denoised_pred,
                    video_conditional_dict=video_conditional_dict,
                    audio_conditional_dict=audio_conditional_dict,
                    video_timestep=video_context_timestep,
                    audio_timestep=audio_context_timestep,
                    video_kv_cache=self.video_kv_cache1,
                    audio_kv_cache=self.audio_kv_cache1,
                    video_crossattn_cache=self.video_crossattn_cache,
                    audio_crossattn_cache=self.audio_crossattn_cache,
                    video_current_start=video_current_start_frame * self.video_frame_seq_length,
                    audio_current_start=audio_current_start_frame * self.audio_frame_seq_length,
                )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            video_current_start_frame += video_current_num_frames
            audio_current_start_frame += audio_current_num_frames

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 3: Decode the output
        # Video Decode
        video_pixels = self.video_vae.decode_to_pixel(video_output.to(video_noise.device), use_cache=False)
        video_pixels = (video_pixels * 0.5 + 0.5).clamp(0, 1)
        
        # Audio Decode
        # Audio VAE usually expects [B, C, L] or similar, ensure dimensions match what VAE wrapper expects
        # Based on OviAudioVAEWrapper: expects zs, returns waveform
        # Typically needs permute: [B, L, C] -> [B, C, L] if VAE expects channels first
        audio_pixels = self.audio_vae.decode_to_pixel(audio_output.to(audio_noise.device).permute(0, 2, 1))

        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video_pixels, audio_pixels, video_output.to(video_noise.device), audio_output.to(audio_noise.device)
        else:
            return video_pixels, audio_pixels

    def _initialize_kv_cache(self, batch_size, dtype, device, local_attn_cfg: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model (Both Video and Audio).
        """
        if local_attn_cfg is not None and local_attn_cfg != -1:
            # Local attention
            video_kv_cache_size = int(local_attn_cfg) * self.video_frame_seq_length
            audio_kv_cache_size = int(local_attn_cfg) * 5 * self.audio_frame_seq_length
        else:
            # Global attention: default fallback
            video_kv_cache_size = 8100 # 9 * 900
            audio_kv_cache_size = 45   # 9 * 5

        print(f"Init KV Cache: Video Size={video_kv_cache_size}, Audio Size={audio_kv_cache_size}")

        # Audio Cache
        kv_cache1 = []
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, audio_kv_cache_size, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, audio_kv_cache_size, 24, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.audio_kv_cache1 = kv_cache1

        # Video Cache
        kv_cache2 = []
        for _ in range(self.num_transformer_blocks):
            kv_cache2.append({
                "k": torch.zeros([batch_size, video_kv_cache_size, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, video_kv_cache_size, 24, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.video_kv_cache1 = kv_cache2

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model (Video and Audio).
        """
        # Audio CrossAttn Cache
        crossattn_cache1 = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache1.append({
                "k": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.audio_crossattn_cache = crossattn_cache1

        # Video CrossAttn Cache
        crossattn_cache2 = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache2.append({
                "k": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.video_crossattn_cache = crossattn_cache2

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it (Video and Audio).
        """
        if local_attn_size_value == -1:
            video_target_size = 18900
            audio_target_size = 105
            policy = "global"
        else:
            video_target_size = int(local_attn_size_value) * self.video_frame_seq_length
            audio_target_size = int(local_attn_size_value) * self.audio_frame_seq_length
            policy = "local"

        # Root modules
        for block in self.generator.model.single_fusion_blocks:
            if hasattr(block.video_block, "self_attn"):
                block.video_block.self_attn.max_attention_size = video_target_size
            if hasattr(block.audio_block, "self_attn"):
                block.audio_block.self_attn.max_attention_size = audio_target_size
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    setattr(module, "max_attention_size", video_target_size)
                except Exception:
                    pass