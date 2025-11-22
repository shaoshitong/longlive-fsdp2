# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import types
from typing import List, Optional
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from ovi.modules.tokenizers import HuggingfaceTokenizer
from ovi.modules.fusion import FusionModel
from ovi.modules.vae2_2 import _video_vae
from ovi.modules.t5 import umt5_xxl
from ovi.modules.fusion_casual import CausalFusionModel
import logging
import torch.amp as amp
from ovi.modules.mmaudio.features_utils import FeaturesUtils
import os
import torch.distributed as dist
import json

class OviTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load("/share/st_workspace/Ovi/ckpts/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )
        
        # Move text encoder to GPU if available
        if torch.cuda.is_available():
            self.text_encoder = self.text_encoder.cuda()

        self.tokenizer = HuggingfaceTokenizer(
            name="/share/st_workspace/Ovi/ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        # ids = ids.to(torch.device('cpu'))
        # mask = mask.to(torch.device('cpu'))
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class OviVideoVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        z_dim=48
        c_dim=160
        vae_pth="/share/st_workspace/Ovi/ckpts/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
        dim_mult=[1, 2, 4, 4]
        temperal_downsample=[False, True, True]
        dtype=torch.float
        device=f"cuda:{dist.get_rank()}"
        self.dtype = dtype
        self.device = device

        self.mean = mean = torch.tensor(
            [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ],
            dtype=dtype,
            device=device,
        )
        self.std = std = torch.tensor(
            [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ],
            dtype=dtype,
            device=device,
        )
        self.scale = [mean, 1.0 / std]

        # init model
        self.model = (
            _video_vae(
                pretrained_path=vae_pth,
                z_dim=z_dim,
                dim=c_dim,
                dim_mult=dim_mult,
                temperal_downsample=temperal_downsample,
            ).eval().requires_grad_(False).to(device))

    def encode(self, videos):
        try:
            if not isinstance(videos, list):
                raise TypeError("videos should be a list")
            with amp.autocast('cuda', dtype=self.dtype):
                return [
                    self.model.encode(u.unsqueeze(0),
                                      self.scale).float().squeeze(0)
                    for u in videos
                ]
        except TypeError as e:
            logging.info(e)
            return None

    def decode(self, zs):
        try:
            if not isinstance(zs, list):
                raise TypeError("zs should be a list")
            with amp.autocast('cuda', dtype=self.dtype):
                return [
                    self.model.decode(u.unsqueeze(0),
                                      self.scale).float().clamp_(-1,
                                                                 1).squeeze(0)
                    for u in zs
                ]
        except TypeError as e:
            logging.info(e)
            return None

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

class OviAudioVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vae_config = {}
        vae_config['mode'] = '16k'
        vae_config['need_vae_encoder'] = True
        device=f"cuda:{dist.get_rank()}"
        ckpt_dir = "/share/st_workspace/Ovi/ckpts"
        tod_vae_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/v1-16.pth")
        bigvgan_vocoder_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/best_netG.pt")

        vae_config['tod_vae_ckpt'] = tod_vae_ckpt
        vae_config['bigvgan_vocoder_ckpt'] = bigvgan_vocoder_ckpt

        self.model = FeaturesUtils(**vae_config).to(device)
        self.model.eval().requires_grad_(False)
        self.model = self.model.bfloat16()

    def encode_to_latent(self, audios):
        return self.model.wrapped_encode(audios)

    def decode_to_pixel(self, zs):
        return self.model.wrapped_decode(zs)

class OviDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.2-TI2V-5B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            if_fsdp2=False
    ):
        super().__init__()
        video_config = "/share/st_workspace/LongLive-FSDP2/ovi/configs/model/dit/video.json"
        audio_config = "/share/st_workspace/LongLive-FSDP2/ovi/configs/model/dit/audio.json"
        assert os.path.exists(video_config), f"{video_config} does not exist"
        assert os.path.exists(audio_config), f"{audio_config} does not exist"

        with open(video_config) as f:
            video_config = json.load(f)

        with open(audio_config) as f:
            audio_config = json.load(f)


        if if_fsdp2:
            with torch.device("meta"):
                if is_causal:
                    self.model = CausalFusionModel(video_config, audio_config, local_attn_size=local_attn_size, sink_size=sink_size)
                else:
                    self.model = FusionModel(video_config, audio_config)
        else:
            if is_causal:
                self.model = CausalFusionModel(video_config, audio_config, local_attn_size=local_attn_size, sink_size=sink_size)
            else:
                self.model = FusionModel(video_config, audio_config)
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.video_seq_len = int(900 * local_attn_size) if local_attn_size > 9 else 8100
        self.audio_seq_len = int(5 * local_attn_size) if local_attn_size > 9 else 45
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        if flow_pred.ndim == 4:
            sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        elif flow_pred.ndim == 2:
            sigma_t = sigmas[timestep_id].reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported dimension: {flow_pred.ndim}")
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        if x0_pred.ndim == 4:
            sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        elif x0_pred.ndim == 2:
            sigma_t = sigmas[timestep_id].reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported dimension: {x0_pred.ndim}")
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        video_conditional_dict: dict,
        audio_conditional_dict: dict,
        video_timestep: torch.Tensor, 
        audio_timestep: torch.Tensor, 
        video_kv_cache: Optional[List[dict]] = None,
        video_crossattn_cache: Optional[List[dict]] = None,
        video_current_start: Optional[int] = None,
        video_cache_start: Optional[int] = None,
        audio_kv_cache: Optional[List[dict]] = None,
        audio_crossattn_cache: Optional[List[dict]] = None,
        audio_current_start: Optional[int] = None,
        audio_cache_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_video_x: Optional[torch.Tensor] = None,
        clean_audio_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        video_prompt_embeds = video_conditional_dict["prompt_embeds"]
        audio_prompt_embeds = audio_conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            video_input_timestep = video_timestep[:, 0]
            audio_input_timestep = audio_timestep[:, 0]
        else:
            video_input_timestep = video_timestep
            audio_input_timestep = audio_timestep
        assert classify_mode == False, "classify_mode is not supported for OviDiffusionWrapper"
        assert clean_video_x is None, "clean_video_x is not supported for OviDiffusionWrapper"
        assert clean_audio_x is None, "clean_audio_x is not supported for OviDiffusionWrapper"
        assert aug_t is None, "aug_t is not supported for OviDiffusionWrapper"
        logits = None
        # X0 prediction
        if audio_kv_cache is not None and video_kv_cache is not None:
            with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=True
                    ):
                vid_flow_pred, audio_flow_pred = self.model(
                    vid=noisy_video.permute(0, 2, 1, 3, 4), # B,F,C,H,W -> B,C,F,H,W
                    audio=noisy_audio, # B,L,D
                    vid_context=video_prompt_embeds,
                    audio_context=audio_prompt_embeds,
                    vid_t=video_input_timestep,
                    audio_t=audio_input_timestep,
                    vid_seq_len=self.video_seq_len,
                    audio_seq_len=self.audio_seq_len,
                    vid_kv_cache=video_kv_cache,
                    vid_crossattn_cache=video_crossattn_cache,
                    vid_current_start=video_current_start,
                    vid_cache_start=video_cache_start,
                    audio_kv_cache=audio_kv_cache,
                    audio_crossattn_cache=audio_crossattn_cache,
                    audio_current_start=audio_current_start,
                    audio_cache_start=audio_cache_start
                )
            vid_flow_pred = torch.stack(vid_flow_pred).permute(0, 2, 1, 3, 4)
            audio_flow_pred = torch.stack(audio_flow_pred)
        else:
            with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=True
                    ):
                vid_flow_pred, audio_flow_pred = self.model(
                    vid=noisy_video.permute(0, 2, 1, 3, 4), # B,F,C,H,W -> B,C,F,H,W
                    audio=noisy_audio, # B,L,D
                    vid_context=video_prompt_embeds,
                    audio_context=audio_prompt_embeds,
                    t=video_input_timestep,
                    vid_seq_len=self.video_seq_len,
                    audio_seq_len=self.audio_seq_len,
                )
            vid_flow_pred = torch.stack(vid_flow_pred).permute(0, 2, 1, 3, 4)
            audio_flow_pred = torch.stack(audio_flow_pred)
        vid_pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=vid_flow_pred.flatten(0, 1),
            xt=noisy_video.flatten(0, 1),
            timestep=video_timestep.flatten(0, 1)
        ).unflatten(0, vid_flow_pred.shape[:2])
        audio_pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=audio_flow_pred.flatten(0, 1),
            xt=noisy_audio.flatten(0, 1),
            timestep=audio_timestep.flatten(0, 1)
        ).unflatten(0, audio_flow_pred.shape[:2])

        if logits is not None:
            return vid_flow_pred, audio_flow_pred, vid_pred_x0, audio_pred_x0, logits

        return vid_flow_pred, audio_flow_pred, vid_pred_x0, audio_pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
