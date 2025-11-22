# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
from utils.debug_option import DEBUG, LOG_GPU_MEMORY
from utils.memory import log_gpu_memory
from pipeline.self_forcing_training import SelfForcingTrainingPipeline

class SelfForcingTrainingOviPipeline(SelfForcingTrainingPipeline):
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 9,
                 context_noise: int = 0,
                 **kwargs):
        super().__init__(
            denoising_step_list=denoising_step_list,
            scheduler=scheduler,
            generator=generator,
            num_frame_per_block=num_frame_per_block,
            independent_first_frame=independent_first_frame,
            same_step_across_blocks=same_step_across_blocks,
            last_step_only=last_step_only,
            num_max_frames=num_max_frames,
            context_noise=context_noise,
            **kwargs
        )

        del self.kv_cache1
        del self.crossattn_cache
        del self.frame_seq_length
        self.video_kv_cache1 = None
        self.video_crossattn_cache = None
        self.audio_kv_cache1 = None
        self.audio_crossattn_cache = None
        self.video_frame_seq_length = 900
        self.audio_frame_seq_length = 1
        # Context used for KV cache calculation
        num_training_frames: Optional[int] = kwargs.get("num_training_frames", 9)
        slice_last_frames: int = int(kwargs.get("slice_last_frames", 9))

        # Compute KV cache supporting list/int and global attention (-1)
        def _resolve_kv_frames(local_cfg):
            if isinstance(local_cfg, (list, tuple)):
                base = int(max(local_cfg)) if len(local_cfg) > 0 else -1
                return min(base + slice_last_frames, num_training_frames)
            else:
                base = int(local_cfg)
                return min(base + slice_last_frames, num_training_frames)

        kv_frames = _resolve_kv_frames(self.local_attn_size)
        self.video_kv_cache_size = int(kv_frames) * self.video_frame_seq_length
        self.audio_kv_cache_size = int(kv_frames) * 5 * self.audio_frame_seq_length

    def generate_chunk_with_cache(
        self,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        video_conditional_dict: dict,
        audio_conditional_dict: dict,
        *,
        video_current_start_frame: int = 0,
        audio_current_start_frame: int = 0,
        requires_grad: bool = True,
        return_sim_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        Chunk generation method tailored for sequential training
        
        Args:
            noise: noise tensor for a single chunk [batch_size, chunk_frames, C, H, W]
            conditional_dict: dictionary of conditional information
            kv_cache: externally provided KV cache (defaults to self.kv_cache1 if None)
            crossattn_cache: externally provided cross-attention cache (defaults to self.crossattn_cache if None)
            current_start_frame: start frame index of the chunk in the full sequence
            requires_grad: whether gradients are required
            return_sim_step: whether to return simulation step info
            
        Returns:
            output: generated chunk [batch_size, chunk_frames, C, H, W]
            denoised_timestep_from: starting denoise timestep
            denoised_timestep_to: ending denoise timestep
        """
        video_batch_size, video_chunk_frames, video_num_channels, video_height, video_width = video_noise.shape
        audio_batch_size, audio_chunk_frames, audio_num_channels = audio_noise.shape
        assert self.independent_first_frame == False, "independent_first_frame must be False for OVI"
        assert video_chunk_frames % self.num_frame_per_block == 0
        assert audio_chunk_frames % (self.num_frame_per_block * 5) == 0
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] generate_chunk_with_cache: video_batch_size={video_batch_size}, video_chunk_frames={video_chunk_frames}, audio_batch_size={audio_batch_size}, audio_chunk_frames={audio_chunk_frames}")
            print(f"[SeqTrain-Pipeline] video_current_start_frame={video_current_start_frame}, audio_current_start_frame={audio_current_start_frame}, requires_grad={requires_grad}")
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-Pipeline: Before chunk generation", device=video_noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)

        video_num_blocks = video_chunk_frames // self.num_frame_per_block
        audio_num_blocks = audio_chunk_frames // (self.num_frame_per_block * 5)
        video_all_num_frames = [self.num_frame_per_block] * video_num_blocks
        audio_all_num_frames = [self.num_frame_per_block * 5] * audio_num_blocks
        all_index = list(range(len(video_all_num_frames)))

        # Prepare output tensor
        video_output = torch.zeros_like(video_noise)
        audio_output = torch.zeros_like(audio_noise)
        
        # Randomly select denoising steps (synced across ranks)
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(video_all_num_frames), num_denoising_steps, device=video_noise.device)
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] Denoising steps: {num_denoising_steps}, exit_flags: {exit_flags}")
        
        # Determine gradient-enabled range — disable everywhere when requires_grad=False
        if not requires_grad:
            start_gradient_frame_index = video_chunk_frames  # Out of range: no gradients anywhere
        else:
            start_gradient_frame_index = 0
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] start_gradient_frame_index={start_gradient_frame_index}")
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-Pipeline: Before block generation loop", device=video_noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # Generate block by block
        video_local_start_frame = 0
        audio_local_start_frame = 0
        # If static local_attn_size, set it on the model before the step loop
        if not (isinstance(self.local_attn_size, (list, tuple)) or (hasattr(self.local_attn_size, "__iter__") and not isinstance(self.local_attn_size, (str, bytes)))):
            self.generator.model.local_attn_size = int(self.local_attn_size)
            self._set_all_modules_max_attention_size(int(self.local_attn_size))
        for block_index, video_current_num_frames, audio_current_num_frames in zip(all_index, video_all_num_frames, audio_all_num_frames):
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Pipeline] Processing block {block_index}: video_frames {video_local_start_frame}-{video_local_start_frame + video_current_num_frames}, audio_frames {audio_local_start_frame}-{audio_local_start_frame + audio_current_num_frames}")
            
            video_noisy_input = video_noise[:, video_local_start_frame:video_local_start_frame + video_current_num_frames]
            audio_noisy_input = audio_noise[:, audio_local_start_frame:audio_local_start_frame + audio_current_num_frames]
            
            # Spatial denoising loop
            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                # If scheduled, set local_attn_size dynamically per timestep
                if isinstance(self.local_attn_size, (list, tuple)) or (hasattr(self.local_attn_size, "__iter__") and not isinstance(self.local_attn_size, (str, bytes))):
                    self._set_all_modules_max_attention_size(int(self.local_attn_size[step_idx]))
                exit_flag = (
                    step_idx == exit_flags[0]
                    if self.same_step_across_blocks
                    else step_idx == exit_flags[block_index]
                )
                video_timestep = torch.ones(
                    [video_batch_size, video_current_num_frames],
                    device=video_noise.device,
                    dtype=torch.int64) * current_timestep
                audio_timestep = torch.ones(
                    [audio_batch_size, audio_current_num_frames],
                    device=audio_noise.device,
                    dtype=torch.int64) * current_timestep
                
                if not exit_flag:
                    # Intermediate steps: no gradients
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Pipeline] Block {block_index} intermediate steps (no grad)")
                        
                    with torch.no_grad():
                        vid_flow_pred, audio_flow_pred, vid_pred_x0, audio_pred_x0 = self.generator(
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
                            video_current_start=(video_current_start_frame+video_local_start_frame) * self.video_frame_seq_length,
                            audio_current_start=(audio_current_start_frame+audio_local_start_frame) * self.audio_frame_seq_length,
                        )
                        
                        # Add noise for the next step
                        if step_idx < len(self.denoising_step_list) - 1:
                            next_timestep = self.denoising_step_list[step_idx + 1]
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
                    # Final step may require gradients
                    enable_grad = video_local_start_frame >= start_gradient_frame_index
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Pipeline] Block {block_index} final step: enable_grad={enable_grad}")
                    
                    context_manager = torch.enable_grad() if enable_grad else torch.no_grad()
                    with context_manager:
                        vid_flow_pred, audio_flow_pred, vid_pred_x0, audio_pred_x0 = self.generator(
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
                            video_current_start=(video_current_start_frame+video_local_start_frame) * self.video_frame_seq_length,
                            audio_current_start=(audio_current_start_frame+audio_local_start_frame) * self.audio_frame_seq_length,
                        )
                    break
            
            video_output[:, video_local_start_frame:video_local_start_frame + video_current_num_frames] = vid_pred_x0
            audio_output[:, audio_local_start_frame:audio_local_start_frame + audio_current_num_frames] = audio_pred_x0
            # Update cache with context noise
            video_context_timestep = torch.ones_like(video_timestep) * self.context_noise
            audio_context_timestep = torch.ones_like(audio_timestep) * self.context_noise
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
            if DEBUG and block_index == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Pipeline] Updating cache with context_noise={self.context_noise}")
            
            with torch.no_grad():
                vid_flow_pred, audio_flow_pred, vid_pred_x0, audio_pred_x0 = self.generator(
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
                    video_current_start=(video_current_start_frame+video_local_start_frame) * self.video_frame_seq_length,
                    audio_current_start=(audio_current_start_frame+audio_local_start_frame) * self.audio_frame_seq_length,
                )
            video_local_start_frame += video_current_num_frames
            audio_local_start_frame += audio_current_num_frames
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-Pipeline: After all blocks generated", device=video_noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # Compute returned timestep information
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
            ).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0
            ).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
            ).item()
        
        if return_sim_step: # pred_video, pred_audio, denoised_timestep_from, denoised_timestep_to 
            return video_output, audio_output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return video_output, audio_output, denoised_timestep_from, denoised_timestep_to

    def inference_with_trajectory(
            self,
            video_noise: torch.Tensor,
            audio_noise: torch.Tensor,
            video_conditional_dict: dict,
            audio_conditional_dict: dict,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            slice_last_frames: int = 9,
    ) -> torch.Tensor:
        # video_noise=video_noise, audio_noise=audio_noise, video_conditional_dict=video_conditional_dict, audio_conditional_dict=audio_conditional_dict, slice_last_frames=slice_last_frames
        video_batch_size, video_num_frames, video_num_channels, video_height, video_width = video_noise.shape
        audio_batch_size, audio_num_frames, audio_num_channels = audio_noise.shape
        assert initial_latent is None, "initial_latent is not supported for OVI"
        assert self.independent_first_frame == False, "independent_first_frame must be False for OVI"
        assert video_num_frames % self.num_frame_per_block == 0
        assert audio_num_frames % (self.num_frame_per_block * 5) == 0
        num_blocks = video_num_frames // self.num_frame_per_block
        audio_num_blocks = audio_num_frames // (self.num_frame_per_block * 5)
        num_output_frames = video_num_frames
        audio_num_output_frames = audio_num_frames
        num_input_frames = 0
        video_output = torch.zeros(
            [video_batch_size, num_output_frames, video_num_channels, video_height, video_width],
            device=video_noise.device,
            dtype=video_noise.dtype
        )
        audio_output = torch.zeros(
            [audio_batch_size, audio_num_output_frames, audio_num_channels],
            device=audio_noise.device,
            dtype=audio_noise.dtype
        )
        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=video_batch_size, dtype=video_noise.dtype, device=video_noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=video_batch_size, dtype=video_noise.dtype, device=video_noise.device
        )
        # Step 2: Cache context feature
        video_current_start_frame = 0
        audio_current_start_frame = 0

        # Step 3: Temporal denoising loop
        video_all_num_frames = [self.num_frame_per_block] * num_blocks
        audio_all_num_frames = [self.num_frame_per_block * 5] * audio_num_blocks
        assert len(video_all_num_frames) == len(audio_all_num_frames)
        all_index = list(range(len(video_all_num_frames)))
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(video_all_num_frames), num_denoising_steps, device=video_noise.device)
        start_gradient_frame_index = num_output_frames - slice_last_frames
        video_grad_enable_mask = torch.zeros((video_batch_size, sum(video_all_num_frames)), dtype=torch.bool)
        audio_grad_enable_mask = torch.zeros((audio_batch_size, sum(audio_all_num_frames)), dtype=torch.bool)

        # If static local_attn_size, set it first
        if not isinstance(self.local_attn_size, (list, tuple)):
            self.generator.model.local_attn_size = int(self.local_attn_size)
            self._set_all_modules_max_attention_size(int(self.local_attn_size))
        # for block_index in range(num_blocks):
        for block_index, video_current_num_frames, audio_current_num_frames in zip(all_index, video_all_num_frames, audio_all_num_frames):
            video_noisy_input = video_noise[
                :, video_current_start_frame - num_input_frames:video_current_start_frame + video_current_num_frames - num_input_frames]
            audio_noisy_input = audio_noise[
                :, audio_current_start_frame - num_input_frames:audio_current_start_frame + audio_current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # If scheduled, set local_attn_size dynamically per timestep
                if isinstance(self.local_attn_size, (list, tuple)):
                    self._set_all_modules_max_attention_size(int(self.local_attn_size[index]))
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                video_timestep = torch.ones(
                    [video_batch_size, video_current_num_frames],
                    device=video_noise.device,
                    dtype=torch.int64) * current_timestep
                audio_timestep = torch.ones(
                    [audio_batch_size, audio_current_num_frames],
                    device=audio_noise.device,
                    dtype=torch.int64) * current_timestep
                if DEBUG and dist.get_rank() == 0:
                    print(f"rank {dist.get_rank()}, video_current_start_frame: {video_current_start_frame}, video_current_num_frames: {video_current_num_frames}, audio_current_start_frame: {audio_current_start_frame}, audio_current_num_frames: {audio_current_num_frames}, current_timestep: {current_timestep}")
                if not exit_flag:
                    with torch.no_grad():
                        if dist.is_initialized() and dist.get_rank() == 0:
                            print("DEBUG 376", video_current_start_frame * self.video_frame_seq_length, audio_current_start_frame * self.audio_frame_seq_length)
                            print("DEBUG 377", video_noisy_input.shape, audio_noisy_input.shape)
                        vid_flow_pred, audio_flow_pred, vid_pred_x0, audio_pred_x0 = self.generator(
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
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if video_current_start_frame < start_gradient_frame_index:
                        video_grad_enable_mask[:, video_current_start_frame:video_current_start_frame + video_current_num_frames] = False
                        audio_grad_enable_mask[:, audio_current_start_frame:audio_current_start_frame + audio_current_num_frames] = False
                        with torch.no_grad():
                            vid_flow_pred, audio_flow_pred, vid_pred_x0, audio_pred_x0 = self.generator(
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
                    else:
                        # print(f"enable grad: {current_start_frame}")
                        video_grad_enable_mask[:, video_current_start_frame:video_current_start_frame + video_current_num_frames] = True
                        audio_grad_enable_mask[:, audio_current_start_frame:audio_current_start_frame + audio_current_num_frames] = True
                        # print("DEBUG 429", video_noisy_input.shape, audio_noisy_input.shape)
                        vid_flow_pred, audio_flow_pred, vid_pred_x0, audio_pred_x0 = self.generator(
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
                    break

            # Step 3.2: record the model's output
            video_output[:, video_current_start_frame:video_current_start_frame + video_current_num_frames] = vid_pred_x0
            audio_output[:, audio_current_start_frame:audio_current_start_frame + audio_current_num_frames] = audio_pred_x0
            # Step 3.3: rerun with timestep zero to update the cache
            video_context_timestep = torch.ones_like(video_timestep) * self.context_noise
            audio_context_timestep = torch.ones_like(audio_timestep) * self.context_noise
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
            # add context noise
            if DEBUG and dist.get_rank() == 0:
                print(f"rank {dist.get_rank()}, video_current_start_frame: {video_current_start_frame}, video_current_num_frames: {video_current_num_frames}, audio_current_start_frame: {audio_current_start_frame}, audio_current_num_frames: {audio_current_num_frames}, current_timestep: {current_timestep}")
                print(f"rank {dist.get_rank()}, rerun_for_cache")
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

            # Step 3.4: update the start and end frame indices
            video_current_start_frame += video_current_num_frames
            audio_current_start_frame += audio_current_num_frames

        if dist.get_rank() == 0 and DEBUG:
            print(f"video_grad_enable_mask: {video_grad_enable_mask[0, :]}")
            print(f"audio_grad_enable_mask: {audio_grad_enable_mask[0, :]}")
            
        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step: # pred_video, pred_audio, denoised_timestep_from, denoised_timestep_to 
            return video_output, audio_output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return video_output, audio_output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.audio_kv_cache_size, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.audio_kv_cache_size, 24, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.audio_kv_cache1 = kv_cache1  # always store the clean cache

        kv_cache2 = []
        for _ in range(self.num_transformer_blocks):
            kv_cache2.append({
                "k": torch.zeros([batch_size, self.video_kv_cache_size, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.video_kv_cache_size, 24, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.video_kv_cache1 = kv_cache2  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache1 = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache1.append({
                "k": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.audio_crossattn_cache = crossattn_cache1

        crossattn_cache2 = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache2.append({
                "k": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.video_crossattn_cache = crossattn_cache2

    def clear_kv_cache(self):
        """
        Zero out all tensors in KV cache and cross-attention cache instead of setting them to None.
        This preserves memory allocation while clearing old information, avoiding reallocation overhead.
        """

        # Clear KV cache
        if getattr(self, "audio_kv_cache1", None) is not None:
            for blk in self.audio_kv_cache1:
                blk["k"].zero_()
                blk["v"].zero_()
                if "global_end_index" in blk:
                    blk["global_end_index"].zero_()
                if "local_end_index" in blk:
                    blk["local_end_index"].zero_()

        # Clear cross-attention cache
        if getattr(self, "audio_crossattn_cache", None) is not None:
            for blk in self.audio_crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False

        # Clear KV cache
        if getattr(self, "video_kv_cache1", None) is not None:
            for blk in self.video_kv_cache1:
                blk["k"].zero_()
                blk["v"].zero_()
                if "global_end_index" in blk:
                    blk["global_end_index"].zero_()
                if "local_end_index" in blk:
                    blk["local_end_index"].zero_()

        # Clear cross-attention cache
        if getattr(self, "video_crossattn_cache", None) is not None:
            for blk in self.video_crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set a unified upper bound for all submodules that contain the max_attention_size attribute.
        local_attn_size_value == -1 indicates global attention (use Wan's default token limit 32760).
        Otherwise set to local_attn_size_value * frame_seq_length.
        """
        if isinstance(local_attn_size_value, (list, tuple)):
            raise ValueError("_set_all_modules_max_attention_size expects an int, got list/tuple.")

        if int(local_attn_size_value) == -1:
            video_target_size = 18900
            audio_target_size = 105
            policy = "global"
        else:
            video_target_size = int(local_attn_size_value) * self.video_frame_seq_length
            audio_target_size = int(local_attn_size_value) * self.audio_frame_seq_length
            policy = "local"

        # Root module

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