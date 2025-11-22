# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import time
from einops import rearrange

from model.base import SelfForcingModel
from utils.memory import log_gpu_memory
import torch.distributed as dist
from utils.debug_option import DEBUG, LOG_GPU_MEMORY
from utils.ovi_wrapper import OviDiffusionWrapper, OviTextEncoder, OviAudioVAEWrapper, OviVideoVAEWrapper
from pipeline.self_forcing_training_ovi import SelfForcingTrainingOviPipeline
from utils.lora import replace_linear_with_lora, lora_true, lora_false


class DMDOvi(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.min_num_training_frames = getattr(args, "min_num_training_frames", 9)
        self.num_training_frames = getattr(args, "num_training_frames", 9)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.real_audio_guidance_scale = args.real_audio_guidance_scale
        self.real_video_guidance_scale = args.real_video_guidance_scale
        self.fake_guidance_scale = args.fake_guidance_scale

        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Wan2.2-TI2V-5B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.2-TI2V-5B")
        self.local_attn_size = getattr(args, "model_kwargs", {}).get("local_attn_size", -1)
        self.generator = OviDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True, if_fsdp2=True)
        self.generator.model.requires_grad_(True)

        self.real_score = OviDiffusionWrapper(model_name=self.real_model_name, is_causal=False, if_fsdp2=True)
        self.real_score.model.requires_grad_(False)

        self.fake_score = self.real_score
        replace_linear_with_lora(self.fake_score.model, rank=128, alpha=1.)
        lora_true(self.fake_score.model, alpha=1.0)
        # self.fake_score.model.requires_grad_(True)

        self.text_encoder = OviTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.audio_vae = OviAudioVAEWrapper()
        self.audio_vae.requires_grad_(False)
        self.video_vae = OviVideoVAEWrapper()
        self.video_vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _consistency_backward_simulation(
        self,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        slice_last_frames: int = 9,
        video_conditional_dict: dict = None,
        audio_conditional_dict: dict = None,
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            video_noise=video_noise, audio_noise=audio_noise, video_conditional_dict=video_conditional_dict, audio_conditional_dict=audio_conditional_dict, slice_last_frames=slice_last_frames
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        local_attn_size = getattr(self.args, "model_kwargs", {}).get("local_attn_size", -1)
        slice_last_frames = getattr(self.args, "slice_last_frames", 9)
        # do not use self.num_training_frames, because it is changed by generator_loss and critic_loss
        num_training_frames = getattr(self.args, "num_training_frames")
        self.inference_pipeline = SelfForcingTrainingOviPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=num_training_frames,
            context_noise=self.args.context_noise,
            local_attn_size=local_attn_size,
            slice_last_frames=slice_last_frames,
            num_training_frames=num_training_frames,
        )


    def _run_generator(
        self,
        video_shape,
        audio_shape,
        video_conditional_dict: dict,
        audio_conditional_dict: dict,
        initial_latent: torch.tensor = None,
        slice_last_frames: int = 9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - video_shape: a list containing the shape of the video [B, F, C, H, W].
            - audio_shape: a list containing the shape of the audio [B, F, D].
            - video_conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - audio_conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - slice_last_frames: an integer indicating the number of frames to slice from the last.
        Output:
            - pred_video: a tensor with shape [B, F, C, H, W].
            - pred_audio: a tensor with shape [B, F, D].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            video_conditional_dict["initial_latent"] = initial_latent
            audio_conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            raise NotImplementedError("i2v is not supported for OviDiffusionWrapper")
        else:
            video_noise_shape = video_shape.copy()
            audio_noise_shape = audio_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [min_num_frames, self.num_training_frames], but still being a multiple of self.num_frame_per_block.
        # If `min_num_frames` is not provided, we fallback to the original default behaviour.
        video_min_num_frames = (self.min_num_training_frames - 1) if self.args.independent_first_frame else self.min_num_training_frames
        video_max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        audio_min_num_frames = ((self.min_num_training_frames - 1) if self.args.independent_first_frame else self.min_num_training_frames) * 5
        audio_max_num_frames = (self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames) * 5
        assert video_max_num_frames % self.num_frame_per_block == 0
        assert video_min_num_frames % self.num_frame_per_block == 0
        video_max_num_blocks = video_max_num_frames // self.num_frame_per_block
        video_min_num_blocks = video_min_num_frames // self.num_frame_per_block
        audio_max_num_blocks = audio_max_num_frames // (self.num_frame_per_block * 5)
        audio_min_num_blocks = audio_min_num_frames // (self.num_frame_per_block * 5)
        num_generated_blocks = torch.randint(video_min_num_blocks, video_max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        video_num_generated_frames = num_generated_blocks * self.num_frame_per_block
        audio_num_generated_frames = num_generated_blocks * self.num_frame_per_block * 5

        if dist.get_rank() == 0 and DEBUG:
            print(f"video_num_generated_frames: {video_num_generated_frames}, audio_num_generated_frames: {audio_num_generated_frames}")
        if self.args.independent_first_frame and initial_latent is None:
            video_num_generated_frames += 1
            audio_num_generated_frames += 5
            video_min_num_frames += 1
            audio_min_num_frames += 5
        # Sync num_generated_frames across all processes
        video_noise_shape[1] = video_num_generated_frames
        audio_noise_shape[1] = audio_num_generated_frames

        pred_video, pred_audio, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation( # TODO: implement this
            video_noise=torch.randn(video_noise_shape,
                              device=self.device, dtype=self.dtype),
            audio_noise=torch.randn(audio_noise_shape,
                              device=self.device, dtype=self.dtype),
            slice_last_frames=slice_last_frames,
            video_conditional_dict=video_conditional_dict,
            audio_conditional_dict=audio_conditional_dict,
        )
        # Decide whether to slice based on `slice_last_frames`; when `slice_last_frames == -1`, keep all frames
        if slice_last_frames != -1 and pred_video.shape[1] > slice_last_frames:
            with torch.no_grad():
                # Re-encode: take all frames before the last (slice_last_frames - 1) frames for pixel decoding
                if slice_last_frames > 1:
                    video_latent_to_decode = pred_video[:, :-(slice_last_frames - 1), ...]
                    audio_latent_to_decode = pred_audio[:, :-(slice_last_frames - 1) * 5, ...]
                else:
                    video_latent_to_decode = pred_video
                    audio_latent_to_decode = pred_audio
                # Decode to video
                video_pixels = self.video_vae.decode_to_pixel(video_latent_to_decode)
                video_frame = video_pixels[:, -1:, ...].to(self.dtype)
                video_frame = rearrange(video_frame, "b t c h w -> b c t h w")
                video_video_latent = self.video_vae.encode_to_latent(video_frame).to(self.dtype)
                audio_pixels = self.audio_vae.decode_to_pixel(audio_latent_to_decode.permute(0, 2, 1))
                audio_frame = audio_pixels[:, :, -5*512:].to(self.dtype)
                audio_audio_latent = self.audio_vae.encode_to_latent(audio_frame).to(self.dtype).permute(0, 2, 1)
            if slice_last_frames > 1:
                video_last_frames = pred_video[:, -(slice_last_frames - 1):, ...]
                pred_video_sliced = torch.cat([video_video_latent, video_last_frames], dim=1)
                audio_last_frames = pred_audio[:, -(slice_last_frames - 1) * 5:, ...]
                pred_audio_sliced = torch.cat([audio_audio_latent, audio_last_frames], dim=1)
            else:
                pred_video_sliced = video_video_latent
                pred_audio_sliced = audio_audio_latent
        else:
            pred_video_sliced = pred_video
            pred_audio_sliced = pred_audio

        if video_num_generated_frames != video_min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            video_gradient_mask = torch.ones_like(pred_video_sliced, dtype=torch.bool)
            audio_gradient_mask = torch.ones_like(pred_audio_sliced, dtype=torch.bool)
            if self.args.independent_first_frame:
                video_gradient_mask[:, :1] = False
                audio_gradient_mask[:, :5] = False
            else:
                video_gradient_mask[:, :self.num_frame_per_block] = False
                audio_gradient_mask[:, :self.num_frame_per_block * 5] = False
        else:
            video_gradient_mask = None
            audio_gradient_mask = None
        pred_video_sliced = pred_video_sliced.to(self.dtype)
        pred_audio_sliced = pred_audio_sliced.to(self.dtype)
        return pred_video_sliced, pred_audio_sliced, video_gradient_mask, audio_gradient_mask, denoised_timestep_from, denoised_timestep_to



    def _compute_kl_grad(
        self,
        video_noisy_latent,
        audio_noisy_latent,
        estimated_clean_video,
        estimated_clean_audio,
        video_timestep,
        audio_timestep,
        video_conditional_dict,
        audio_conditional_dict,
        video_unconditional_dict,
        audio_unconditional_dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        lora_true(self.fake_score.model, alpha=1.0)
        _, _, vid_pred_x0_cond, audio_pred_x0_cond = self.fake_score(
            noisy_video=video_noisy_latent,
            noisy_audio=audio_noisy_latent,
            video_conditional_dict=video_conditional_dict,
            audio_conditional_dict=audio_conditional_dict,
            video_timestep=video_timestep,
            audio_timestep=audio_timestep,

        )
 
        if self.fake_guidance_scale != 0.0:
            lora_true(self.fake_score.model, alpha=1.0)
            _, _, vid_pred_x0_uncond, audio_pred_x0_uncond = self.fake_score(
                noisy_video=video_noisy_latent,
                noisy_audio=audio_noisy_latent,
                video_conditional_dict=video_unconditional_dict,
                audio_conditional_dict=audio_unconditional_dict,
                video_timestep=video_timestep,
                audio_timestep=audio_timestep,

            )
            vid_pred_x0 = vid_pred_x0_cond + (
                vid_pred_x0_cond - vid_pred_x0_uncond
            ) * self.fake_guidance_scale
            audio_pred_x0 = audio_pred_x0_cond + (
                audio_pred_x0_cond - audio_pred_x0_uncond
            ) * self.fake_guidance_scale
        else:
            vid_pred_x0 = vid_pred_x0_cond
            audio_pred_x0 = audio_pred_x0_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        lora_false(self.real_score.model)
        _, _, vid_pred_real_x0_cond, audio_pred_real_x0_cond = self.real_score(
            noisy_video=video_noisy_latent,
            noisy_audio=audio_noisy_latent,
            video_conditional_dict=video_conditional_dict,
            audio_conditional_dict=audio_conditional_dict,
            video_timestep=video_timestep,
            audio_timestep=audio_timestep,
        )

        lora_false(self.real_score.model)   
        _, _, vid_pred_real_x0_uncond, audio_pred_real_x0_uncond = self.real_score(
            noisy_video=video_noisy_latent,
            noisy_audio=audio_noisy_latent,
            video_conditional_dict=video_unconditional_dict,
            audio_conditional_dict=audio_unconditional_dict,
            video_timestep=video_timestep,
            audio_timestep=audio_timestep,
        )

        vid_pred_real_x0 = vid_pred_real_x0_cond + (
            vid_pred_real_x0_cond - vid_pred_real_x0_uncond
        ) * self.real_video_guidance_scale
        audio_pred_real_x0 = audio_pred_real_x0_cond + (
            audio_pred_real_x0_cond - audio_pred_real_x0_uncond
        ) * self.real_audio_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        vid_grad = (vid_pred_x0 - vid_pred_real_x0)
        audio_grad = (audio_pred_x0 - audio_pred_real_x0)
        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            video_p_real = (estimated_clean_video - vid_pred_real_x0)
            video_normalizer = torch.abs(video_p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            vid_grad = vid_grad / video_normalizer
            audio_p_real = (estimated_clean_audio - audio_pred_real_x0)
            audio_normalizer = torch.abs(audio_p_real).mean(dim=[1, 2], keepdim=True)
            audio_grad = audio_grad / audio_normalizer
        vid_grad = torch.nan_to_num(vid_grad)
        audio_grad = torch.nan_to_num(audio_grad)

        return vid_grad, audio_grad, {
            "dmdtrain_video_gradient_norm": torch.mean(torch.abs(vid_grad)).detach(),
            "dmdtrain_audio_gradient_norm": torch.mean(torch.abs(audio_grad)).detach(),
            "video_timestep": video_timestep.detach(),
            "audio_timestep": audio_timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        video_conditional_dict: dict,
        audio_conditional_dict: dict,
        video_unconditional_dict: dict,
        audio_unconditional_dict: dict,
        video_gradient_mask: Optional[torch.Tensor] = None,
        audio_gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        video_original_latent = video
        audio_original_latent = audio
        video_batch_size, video_num_frame = video.shape[:2]
        audio_batch_size, audio_num_frame = audio.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            video_timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                video_batch_size,
                video_num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )
            audio_timestep = video_timestep[:, :1].expand(-1, audio_num_frame)
            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                video_timestep = self.timestep_shift * \
                    (video_timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (video_timestep / 1000)) * 1000
                audio_timestep = self.timestep_shift * \
                    (audio_timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (audio_timestep / 1000)) * 1000
            video_timestep = video_timestep.clamp(self.min_step, self.max_step)
            audio_timestep = audio_timestep.clamp(self.min_step, self.max_step)


            video_noise = torch.randn_like(video)
            audio_noise = torch.randn_like(audio)
            video_noisy_latent = self.scheduler.add_noise(
                video.flatten(0, 1),
                video_noise.flatten(0, 1),
                video_timestep.flatten(0, 1)
            ).detach().unflatten(0, (video_batch_size, video_num_frame))
            audio_noisy_latent = self.scheduler.add_noise(
                audio.flatten(0, 1),
                audio_noise.flatten(0, 1),
                audio_timestep.flatten(0, 1)
            ).detach().unflatten(0, (audio_batch_size, audio_num_frame))
            # Step 2: Compute the KL grad
            video_grad, audio_grad, dmd_log_dict = self._compute_kl_grad(
                video_noisy_latent=video_noisy_latent,
                audio_noisy_latent=audio_noisy_latent,
                estimated_clean_video=video_original_latent,
                estimated_clean_audio=audio_original_latent,
                video_timestep=video_timestep,
                audio_timestep=audio_timestep,
                video_conditional_dict=video_conditional_dict,
                audio_conditional_dict=audio_conditional_dict,
                video_unconditional_dict=video_unconditional_dict,
                audio_unconditional_dict=audio_unconditional_dict
            )

        if video_gradient_mask is not None:
            video_dmd_loss = 0.5 * F.mse_loss(video_original_latent.double(
            )[video_gradient_mask], (video_original_latent.double() - video_grad.double()).detach()[video_gradient_mask], reduction="mean")
        else:
            video_dmd_loss = 0.5 * F.mse_loss(video_original_latent.double(
            ), (video_original_latent.double() - video_grad.double()).detach(), reduction="mean")

        if video_gradient_mask is not None:
            audio_dmd_loss = 0.5 * F.mse_loss(audio_original_latent.double(
            )[audio_gradient_mask], (audio_original_latent.double() - audio_grad.double()).detach()[audio_gradient_mask], reduction="mean")
        else:
            audio_dmd_loss = 0.5 * F.mse_loss(audio_original_latent.double(
            ), (audio_original_latent.double() - audio_grad.double()).detach(), reduction="mean")
        return video_dmd_loss, audio_dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        video_shape,
        audio_shape,
        video_conditional_dict: dict,
        audio_conditional_dict: dict,
        video_unconditional_dict: dict,
        audio_unconditional_dict: dict,
        video_clean_latent: torch.Tensor,
        audio_clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: Before generator unroll", device=self.device, rank=dist.get_rank())
        # Step 1: Unroll generator to obtain fake videos
        slice_last_frames = getattr(self.args, "slice_last_frames", 9)
        _t_gen_start = time.time()
        if DEBUG and dist.get_rank() == 0:
            print(f"generator_rollout")
        pred_video, pred_audio, video_gradient_mask, audio_gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            video_shape=video_shape,
            audio_shape=audio_shape,
            video_conditional_dict=video_conditional_dict,
            audio_conditional_dict=audio_conditional_dict,
            initial_latent=initial_latent,
            slice_last_frames=slice_last_frames)
        if dist.get_rank() == 0 and DEBUG:
            print(f"pred_video: {pred_video.shape}")
            print(f"pred_audio: {pred_audio.shape}")
            if video_gradient_mask is not None:   
                print(f"video_gradient_mask: {video_gradient_mask[0, :, 0, 0, 0]}")
            else:
                print(f"video_gradient_mask: None")
            if audio_gradient_mask is not None:   
                print(f"audio_gradient_mask: {audio_gradient_mask[0, :, 0]}")
            else:
                print(f"audio_gradient_mask: None")
        gen_time = time.time() - _t_gen_start
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: After generator unroll", device=self.device, rank=dist.get_rank())
        # Step 2: Compute the DMD loss
        _t_loss_start = time.time()
        video_dmd_loss, audio_dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            video=pred_video,
            audio=pred_audio,
            video_conditional_dict=video_conditional_dict,
            audio_conditional_dict=audio_conditional_dict,
            video_unconditional_dict=video_unconditional_dict,
            audio_unconditional_dict=audio_unconditional_dict,
            video_gradient_mask=video_gradient_mask,
            audio_gradient_mask=audio_gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: After compute_distribution_matching_loss", device=self.device, rank=dist.get_rank())
        try:
            loss_val = video_dmd_loss.item() + audio_dmd_loss.item()
        except Exception:
            loss_val = float('nan')
        loss_time = time.time() - _t_loss_start
        # print(f"[GeneratorLoss] loss {loss_val} | gen_time {gen_time:.3f}s | loss_time {loss_time:.3f}s")

        dmd_log_dict.update({
            "gen_time": gen_time,
            "loss_time": loss_time
        })

        return video_dmd_loss, audio_dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        video_shape,
        audio_shape,
        video_conditional_dict: dict,
        audio_conditional_dict: dict,
        video_unconditional_dict: dict,
        audio_unconditional_dict: dict,
        video_clean_latent: torch.Tensor,
        audio_clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate video/audio from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: Before generator unroll", device=self.device, rank=dist.get_rank())
        slice_last_frames = getattr(self.args, "slice_last_frames", 9)
        # Step 1: Run generator on backward simulated noisy input
        _t_gen_start = time.time()
        with torch.no_grad():
            if DEBUG and dist.get_rank() == 0:
                print(f"critic_rollout")
            pred_video, pred_audio, video_gradient_mask, audio_gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                video_shape=video_shape,
                audio_shape=audio_shape,
                video_conditional_dict=video_conditional_dict,
                audio_conditional_dict=audio_conditional_dict,
                initial_latent=initial_latent,
                slice_last_frames=slice_last_frames)
        if dist.get_rank() == 0 and DEBUG:
            print(f"pred_video: {pred_video.shape}")
            print(f"pred_audio: {pred_audio.shape}")
        gen_time = time.time() - _t_gen_start
        video_batch_size, video_num_frame = pred_video.shape[:2]
        audio_batch_size, audio_num_frame = pred_audio.shape[:2]
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: After generator unroll", device=self.device, rank=dist.get_rank())
        _t_loss_start = time.time()

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        video_critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            video_batch_size,
            video_num_frame,
            self.num_frame_per_block,
            uniform_timestep=True
        )
        audio_critic_timestep = video_critic_timestep[:, :1].expand(-1, audio_num_frame)
        if self.timestep_shift > 1:
            video_critic_timestep = self.timestep_shift * \
                (video_critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (video_critic_timestep / 1000)) * 1000
            audio_critic_timestep = self.timestep_shift * \
                (audio_critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (audio_critic_timestep / 1000)) * 1000
        video_critic_timestep = video_critic_timestep.clamp(self.min_step, self.max_step)
        audio_critic_timestep = audio_critic_timestep.clamp(self.min_step, self.max_step)
        video_critic_noise = torch.randn_like(pred_video)
        audio_critic_noise = torch.randn_like(pred_audio)
        noisy_generated_video = self.scheduler.add_noise(
            pred_video.flatten(0, 1),
            video_critic_noise.flatten(0, 1),
            video_critic_timestep.flatten(0, 1)
        ).unflatten(0, (video_batch_size, video_num_frame))
        noisy_generated_audio = self.scheduler.add_noise(
            pred_audio.flatten(0, 1),
            audio_critic_noise.flatten(0, 1),
            audio_critic_timestep.flatten(0, 1)
        ).unflatten(0, (audio_batch_size, audio_num_frame))

        lora_true(self.fake_score.model, alpha=1.0)
        _, _, pred_fake_video, pred_fake_audio = self.fake_score(
            noisy_video=noisy_generated_video,
            noisy_audio=noisy_generated_audio,
            video_conditional_dict=video_conditional_dict,
            audio_conditional_dict=audio_conditional_dict,
            video_timestep=video_critic_timestep,
            audio_timestep=audio_critic_timestep,
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            video_flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_video.flatten(0, 1),
                xt=noisy_generated_video.flatten(0, 1),
                timestep=video_critic_timestep.flatten(0, 1)
            )
            audio_flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_audio.flatten(0, 1),
                xt=noisy_generated_audio.flatten(0, 1),
                timestep=audio_critic_timestep.flatten(0, 1)
            )
            video_pred_fake_noise = None
            audio_pred_fake_noise = None
        else:
            video_flow_pred = None
            audio_flow_pred = None
            video_pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_video.flatten(0, 1),
                xt=noisy_generated_video.flatten(0, 1),
                timestep=video_critic_timestep.flatten(0, 1)
            ).unflatten(0, (video_batch_size, video_num_frame))
            audio_pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_audio.flatten(0, 1),
                xt=noisy_generated_audio.flatten(0, 1),
                timestep=audio_critic_timestep.flatten(0, 1)
            ).unflatten(0, (audio_batch_size, audio_num_frame))

        video_denoising_loss = self.denoising_loss_func(
            x=pred_video.flatten(0, 1),
            x_pred=pred_fake_video.flatten(0, 1),
            noise=video_critic_noise.flatten(0, 1),
            noise_pred=video_pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=video_critic_timestep.flatten(0, 1),
            flow_pred=video_flow_pred
        )
        audio_denoising_loss = self.denoising_loss_func(
            x=pred_audio.flatten(0, 1),
            x_pred=pred_fake_audio.flatten(0, 1),
            noise=audio_critic_noise.flatten(0, 1),
            noise_pred=audio_pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=audio_critic_timestep.flatten(0, 1),
            flow_pred=audio_flow_pred
        )

        try:
            loss_val = video_denoising_loss.item() + audio_denoising_loss.item()
        except Exception:
            loss_val = float('nan')
        loss_time = time.time() - _t_loss_start
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: After denoising loss", device=self.device, rank=dist.get_rank())
        # print(f"[CriticLoss] loss {loss_val} | gen_time {gen_time:.3f}s | loss_time {loss_time:.3f}s")


        # Step 5: Debugging Log
        critic_log_dict = {
            "video_critic_timestep": video_critic_timestep.detach(),
            "audio_critic_timestep": audio_critic_timestep.detach(),
            "gen_time": gen_time,
            "loss_time": loss_time
        }

        return video_denoising_loss, audio_denoising_loss, critic_log_dict
