# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from .attention import flash_attention
from torch.utils.checkpoint import checkpoint
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
from .model import ChannelLastConv1d, ConvMLP, WanRMSNorm, WanLayerNorm, WAN_CROSSATTENTION_CLASSES, ModulationAdd, MLPProj
import torch.distributed as dist
_compiled_flex_attention = None

def get_compiled_flex_attention():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        print("首次编译 flex_attention...")
        _compiled_flex_attention = torch.compile(
            flex_attention, 
            dynamic=False, 
            mode="default"
        )
        print("flex_attention 编译完成")
    return _compiled_flex_attention

flex_attention = get_compiled_flex_attention()

def gradient_checkpointing(module: nn.Module, *args, enabled: bool, **kwargs):
    if enabled:
        return checkpoint(module, *args, use_reentrant=False, **kwargs)
    else:
        return module(*args, **kwargs)


def causal_rope_apply_3d(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    return torch.stack(output).type_as(x)

@amp.autocast('cuda', enabled=False)
def causal_rope_apply_1d(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2 ## b l h d
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"
    
    # loop over samples
    output = []
    for i, (l, ) in enumerate(grid_sizes.tolist()):
        seq_len = l
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2)) # [l n d//2]
        x_i_rope = x_i[:, :, :c_rope] * freqs[start_frame:seq_len+start_frame, None, :]  # [L, N, c_rope]
        x_i_passthrough = x_i[:, :, c_rope:]  # untouched dims
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

@amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000, freqs_scaling=1.0):
    assert dim % 2 == 0
    pos =  torch.arange(max_seq_len)
    freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    freqs = freqs_scaling * freqs
    freqs = torch.outer(pos, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs, start_frame=0):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        return causal_rope_apply_3d(x, grid_sizes, freqs, start_frame)
    else:
        return causal_rope_apply_1d(x, grid_sizes, freqs, start_frame)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    # query, key, value function
    def qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, cache_start=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start
        q, k, v = self.qkv_fn(x)
        if grid_sizes.shape[1] > 1:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        else:
            frame_seqlen = 1
        current_start_frame = current_start // frame_seqlen
        roped_query = rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
        roped_key = rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
        current_end = current_start + roped_query.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = roped_query.shape[1]

        k_clone = kv_cache["k"].clone()
        v_clone = kv_cache["v"].clone()

        # DEBUG 193 45 30 True 3 15
        # DEBUG 231 torch.Size([1, 15, 24, 128]) torch.Size([1, 15, 24, 128]) 30 45 0 15
        # DEBUG 193 8100 5400 True 2700 2700
        # DEBUG 231 torch.Size([1, 2700, 24, 128]) torch.Size([1, 2700, 24, 128]) 5400 8100 0 2700
        is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     print("DEBUG 193", current_end, current_start, is_recompute, sink_tokens, num_new_tokens)
        local_end_index = -1 # 稍后计算
        local_start_index = -1 # 稍后计算

        if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens # KV_CACHE_SIZE - NUM_EVICTED_TOKENS - SINK_TOKENS
            # if dist.is_initialized() and dist.get_rank() == 0:
            #     print("DEBUG 201", num_evicted_tokens, num_rolled_tokens, sink_tokens)
            # 在本地副本上执行滚动
            k_clone[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone().detach()
            v_clone[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone().detach()
            
            local_end_index = kv_cache["local_end_index"].item() + current_end - \
                kv_cache["global_end_index"].item() - num_evicted_tokens # KV_CACHE_SIZE - NUM_NEW_TOKENS + current_end - global_end_index
            local_start_index = local_end_index - num_new_tokens
            
            # 在本地副本上更新, roped_key 带有梯度
            write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
            roped_offset = max(0, write_start_index - local_start_index)
            write_len = max(0, local_end_index - write_start_index)
            if write_len > 0:
                # if dist.is_initialized() and dist.get_rank() == 0:
                #     print("DEBUG 217", k_clone[:, write_start_index:local_end_index].shape, roped_key[:, roped_offset:roped_offset + write_len].shape, write_start_index, local_end_index, roped_offset, roped_offset + write_len)
                k_clone[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
                v_clone[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]
        else:   
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens

            write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
            roped_offset = max(0, write_start_index - local_start_index)
            write_len = max(0, local_end_index - write_start_index)
            if write_len > 0:
                # 在本地副本上更新, roped_key 带有梯度
                # if dist.is_initialized() and dist.get_rank() == 0:
                #     print("DEBUG 231", k_clone[:, write_start_index:local_end_index].shape, roped_key[:, roped_offset:roped_offset + write_len].shape, write_start_index, local_end_index, roped_offset, roped_offset + write_len)
                k_clone[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
                v_clone[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]
        
        # --- 2. Attention 计算 ---
        # 使用 k_clone 和 v_clone (它们带有梯度) 进行 attention
        if sink_tokens > 0:
            local_budget = self.max_attention_size - sink_tokens
            k_sink = k_clone[:, :sink_tokens]
            v_sink = v_clone[:, :sink_tokens]
            # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            #     print(f"local_budget: {local_budget}")
            if local_budget > 0:
                local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                k_local = k_clone[:, local_start_for_window:local_end_index]
                v_local = v_clone[:, local_start_for_window:local_end_index]
                k_cat = torch.cat([k_sink, k_local], dim=1)
                v_cat = torch.cat([v_sink, v_local], dim=1)
            else:
                k_cat = k_sink
                v_cat = v_sink

            x = flash_attention(
                roped_query,
                k_cat,
                v_cat
            )
        else:
            window_start = max(0, local_end_index - self.max_attention_size)
            x = flash_attention(
                roped_query,
                k_clone[:, window_start:local_end_index],
                v_clone[:, window_start:local_end_index]
            )

        x = x.flatten(2)
        x = self.o(x)
        
        if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
            
            # 重新执行滚动, 这次是在真正的 kv_cache 上
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone().detach()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone().detach()

            write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
            roped_offset = max(0, write_start_index - local_start_index)
            write_len = max(0, local_end_index - write_start_index)
            if write_len > 0 and roped_key[:, roped_offset:roped_offset + write_len].shape[1] == write_len:
                kv_cache["k"][:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len].detach()
                kv_cache["v"][:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len].detach()

        else:
            if current_start == 0:
                kv_cache["local_end_index"].fill_(0)
                kv_cache["global_end_index"].fill_(0)
            
            write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
            roped_offset = max(0, write_start_index - local_start_index)
            write_len = max(0, local_end_index - write_start_index)
            if write_len > 0 and roped_key[:, roped_offset:roped_offset + write_len].shape[1] == write_len:
                kv_cache["k"][:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len].detach()
                kv_cache["v"][:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len].detach()

        if not is_recompute:
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 additional_emb_length=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        if cross_attn_type == 'i2v_cross_attn':
            assert additional_emb_length is not None, "additional_emb_length should be specified for i2v_cross_attn"
            self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                num_heads,
                                                (-1, -1),
                                                qk_norm,
                                                eps, 
                                                additional_emb_length)
        else:
            assert additional_emb_length is None, "additional_emb_length should be None for t2v_cross_attn"
            self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                num_heads,
                                                (-1, -1),
                                                qk_norm,
                                                eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.modulation = ModulationAdd(dim, 6)


    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        assert e.dtype == torch.bfloat16
        assert len(e.shape) == 4 and e.size(2) == 6
        with amp.autocast('cuda', dtype=torch.bfloat16):
            e = self.modulation(e).chunk(6, dim=2)
        assert e[0].dtype == torch.bfloat16

        # self-attention
        y = self.self_attn(
            (self.norm1(x).bfloat16().unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)
        with amp.autocast('cuda', dtype=torch.bfloat16):
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).bfloat16().unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            with amp.autocast('cuda', dtype=torch.bfloat16):
                x = x + (y.unflatten(dim=1, sizes=(num_frames,
                        frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x



class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0])).flatten(1, 2)
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video, text-to-audio.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 additional_emb_dim=None,
                 additional_emb_length=None,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=6,
                 sink_size=2,
                 qk_norm=True,
                 cross_attn_norm=True,
                 gradient_checkpointing = False,
                 temporal_rope_scaling_factor=1.0,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 't2a', 'tt2a', 'ti2v'] ## tt2a means text transcript + text description to audio (to support both TTS and T2A
        self.model_type = model_type
        is_audio_type = "a" in self.model_type
        is_video_type = "v" in self.model_type
        assert is_audio_type ^ is_video_type, "Either audio or video model should be specified"
        if is_audio_type:
            ## audio model
            assert len(patch_size) == 1 and patch_size[0] == 1, "Audio model should only accept 1 dimensional input, and we dont do patchify"

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.temporal_rope_scaling_factor = temporal_rope_scaling_factor
        self.is_audio_type = is_audio_type
        self.is_video_type = is_video_type
        # embeddings
        if is_audio_type:
            ## hardcoded to MMAudio
            self.patch_embedding = nn.Sequential(
                ChannelLastConv1d(in_dim, dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(dim, dim * 4, kernel_size=7, padding=3),
            )
        else:
            self.patch_embedding = nn.Conv3d(
                in_dim, dim, kernel_size=patch_size, stride=patch_size)
            
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        # blocks
        ## so i2v and tt2a share the same cross attention while t2v and t2a share the same cross attention
        cross_attn_type = 't2v_cross_attn' if model_type in ['t2v', 't2a', 'ti2v'] else 'i2v_cross_attn'

        if cross_attn_type == 't2v_cross_attn':
            assert additional_emb_dim is None and additional_emb_length is None, "additional_emb_length should be None for t2v and t2a model"
        else:
            assert additional_emb_dim is not None and additional_emb_length is not None, "additional_emb_length should be specified for i2v and tt2a model"

        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              local_attn_size, sink_size, qk_norm, cross_attn_norm, eps, additional_emb_length)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        self._set_gradient_checkpointing(enable=gradient_checkpointing)
        self.set_rope_params()

        if model_type in ['i2v', 'tt2a']:
            self.img_emb = MLPProj(additional_emb_dim, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 2
        self.uniform_timestep = False
        self.scheduler = None
        self.seq_len = 32760

    def set_rope_params(self):
        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        dim = self.dim
        num_heads = self.num_heads
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads

        if self.is_audio_type:
            ## to be determined
            # self.freqs = rope_params(1024, d, freqs_scaling=temporal_rope_scaling_factor)
            self.freqs = rope_params(1024, d - 4 * (d // 6), freqs_scaling=self.temporal_rope_scaling_factor)
        else:
            self.freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
                                dim=1)

    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func=None):
        self.gradient_checkpointing = enable

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 2040, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 2040, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        ends[:frame_seqlen] = frame_seqlen

        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def prepare_transformer_block_kwargs(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        first_frame_is_clean=False,
    ):

        # params
        ## need to change!
        device = next(self.patch_embedding.parameters()).device

        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x] ## x is list of [B L D] or [B C F H W]
        if self.is_audio_type:
            # [B, 1]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[1:2], dtype=torch.long) for u in x]
            )
        else:
            # [B, 3]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x] # [B C F H W] -> [B (F H W) C] -> [B L C]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len, f"Sequence length {seq_lens.max()} exceeds maximum {seq_len}."
        # x = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
        #               dim=1) for u in x
        # ]) # single [B, L, C]
        x = torch.cat(x)
        # time embeddings
        assert t.dim() == 2, "t should be a 2D tensor"
        if first_frame_is_clean:
            t = torch.ones((t.size(0), t.size(1)), device=t.device, dtype=t.dtype) * t
            for i in range(t.size(0)):
                t[i, 0] = 0

        # torch.Size([1, 3, 6, 1536]) DEBUG e0.shape                                                                                                                                                                                                        
        # torch.Size([1, 3]) DEBUG t.shape 
        with amp.autocast('cuda', dtype=torch.bfloat16):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t.flatten()).bfloat16())
            e0 = self.time_projection(e).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            assert e.dtype == torch.bfloat16 and e0.dtype == torch.bfloat16
        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        return x, e, kwargs
        
    def post_transformer_block_out(self, x, grid_sizes, e, t):
        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        if self.is_audio_type:
            ## grid_sizes is [B 1] where 1 is L, 
            # converting grid_sizes from [B 1] -> [B]
            grid_sizes = [gs[0] for gs in grid_sizes]
            assert len(x) == len(grid_sizes)
            x = [u[:gs] for u, gs in zip(x, grid_sizes)]
        else:
            ## grid_sizes is [B 3] where 3 is F H w
            x = self.unpatchify(x, grid_sizes)

        return [u.bfloat16() for u in x]


    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        first_frame_is_clean=False,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
                OR 
                List of input audio tensors, each with shape [L, C_in]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
                OR
                List of denoised audio tensors with original input shapes [L, C_in]
        """
        x, e, kwargs = self.prepare_transformer_block_kwargs(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y,
            first_frame_is_clean=first_frame_is_clean
        )

        for block_index, block in enumerate(self.blocks):
            kwargs.update(
                {
                    "kv_cache": kv_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start
                }
            )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "crossattn_cache": crossattn_cache[block_index],
                    }
                )
            x = gradient_checkpointing(
                    enabled=(self.training and self.gradient_checkpointing),
                    module=block,
                    x=x,
                    **kwargs
                )

        return self.post_transformer_block_out(x, kwargs['grid_sizes'], e, t)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            # v is [F H w] F * H * 80, 100, it was right padded by 20. 
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        # out is list of [C F H W]
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        if self.is_video_type:
            assert isinstance(self.patch_embedding, nn.Conv3d), f"Patch embedding for video should be a Conv3d layer, got {type(self.patch_embedding)}"
            nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def set_local_window_size(self, local_window_size, frame_seqlen=900):
        for name, module in self.named_modules():
            if isinstance(module, CausalWanSelfAttention):
                module.max_attention_size = int(local_window_size * frame_seqlen)