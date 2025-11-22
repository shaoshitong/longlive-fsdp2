
import math
import torch
import torch.nn as nn
from ovi.modules.model import (
    WanLayerNorm,
    WanRMSNorm,
    ChannelLastConv1d,
    ConvMLP,
    WanAttentionBlock,
    Head,
    gradient_checkpointing,
    rope_apply,
    sinusoidal_embedding_1d,
    rope_params,
)
from ovi.modules.attention import flash_attention

import torch.distributed as dist
class FusionModelBlock(nn.Module):
    def __init__(self, video_block_cfg, audio_block_cfg):
        super().__init__()
        self.video_block = self._build_block(video_block_cfg)
        self.audio_block = self._build_block(audio_block_cfg)

    @staticmethod
    def _build_block(cfg):
        return WanAttentionBlock(
            cfg["cross_attn_type"],
            cfg["dim"],
            cfg["ffn_dim"],
            cfg["num_heads"],
            cfg["window_size"],
            cfg["qk_norm"],
            cfg["cross_attn_norm"],
            cfg["eps"],
            cfg["additional_emb_length"],
        )

    def forward(
        self,
        vid,
        audio,
        vid_e,
        vid_seq_lens,
        vid_grid_sizes,
        vid_freqs,
        vid_context,
        vid_context_lens,
        audio_e,
        audio_seq_lens,
        audio_grid_sizes,
        audio_freqs,
        audio_context,
        audio_context_lens,
    ):
        assert audio_e.dtype == torch.bfloat16
        assert len(audio_e.shape) == 4 and audio_e.size(2) == 6
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            audio_e = self.audio_block.modulation(audio_e).chunk(6, dim=2)
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     print("DEBUG 61", audio_freqs.shape)
        audio_y = self.audio_block.self_attn(
            self.audio_block.norm1(audio).bfloat16() * (1 + audio_e[1].squeeze(2)) + audio_e[0].squeeze(2),
            audio_seq_lens,
            audio_grid_sizes,
            audio_freqs,
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            audio = audio + audio_y * audio_e[2].squeeze(2)

        assert vid_e.dtype == torch.bfloat16
        assert len(vid_e.shape) == 4 and vid_e.size(2) == 6
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            vid_e = self.video_block.modulation(vid_e).chunk(6, dim=2)
        vid_y = self.video_block.self_attn(
            self.video_block.norm1(vid).bfloat16() * (1 + vid_e[1].squeeze(2)) + vid_e[0].squeeze(2),
            vid_seq_lens,
            vid_grid_sizes,
            vid_freqs,
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            vid = vid + vid_y * vid_e[2].squeeze(2)

        og_audio = audio

        audio = self.single_fusion_cross_attention_ffn_forward(
            self.audio_block,
            audio,
            audio_grid_sizes,
            audio_freqs,
            vid,
            vid_seq_lens,
            vid_grid_sizes,
            vid_freqs,
            audio_context,
            audio_context_lens,
            audio_e,
        )
        assert not torch.equal(og_audio, audio), "Audio should be changed after cross-attention!"

        vid = self.single_fusion_cross_attention_ffn_forward(
            self.video_block,
            vid,
            vid_grid_sizes,
            vid_freqs,
            og_audio,
            audio_seq_lens,
            audio_grid_sizes,
            audio_freqs,
            vid_context,
            vid_context_lens,
            vid_e,
        )

        return vid, audio

    def single_fusion_cross_attention_ffn_forward(
        self,
        attn_block,
        src_seq,
        src_grid_sizes,
        src_freqs,
        target_seq,
        target_seq_lens,
        target_grid_sizes,
        target_freqs,
        context,
        context_lens,
        src_e,
    ):
        src_seq = src_seq + self.single_fusion_cross_attention_forward(
            attn_block.cross_attn,
            attn_block.norm3(src_seq),
            src_grid_sizes=src_grid_sizes,
            src_freqs=src_freqs,
            target_seq=target_seq,
            target_seq_lens=target_seq_lens,
            target_grid_sizes=target_grid_sizes,
            target_freqs=target_freqs,
            context=context,
            context_lens=context_lens,
        )
        y = attn_block.ffn(attn_block.norm2(src_seq).bfloat16() * (1 + src_e[4].squeeze(2)) + src_e[3].squeeze(2))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            src_seq = src_seq + y * src_e[5].squeeze(2)
        return src_seq

    def single_fusion_cross_attention_forward(
        self,
        cross_attn_block,
        src_seq,
        src_grid_sizes,
        src_freqs,
        target_seq,
        target_seq_lens,
        target_grid_sizes,
        target_freqs,
        context,
        context_lens,
    ):
        b, n, d = src_seq.size(0), cross_attn_block.num_heads, cross_attn_block.head_dim
        if hasattr(cross_attn_block, "k_img"):
            q, k, v, k_img, v_img = cross_attn_block.qkv_fn(src_seq, context)
        else:
            q, k, v = cross_attn_block.qkv_fn(src_seq, context)
            k_img = v_img = None

        x = flash_attention(q, k, v, k_lens=context_lens)
        if k_img is not None:
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
            x = x + img_x

        target_seq = cross_attn_block.pre_attn_norm_fusion(target_seq)
        k_target = cross_attn_block.norm_k_fusion(cross_attn_block.k_fusion(target_seq)).view(b, -1, n, d)
        v_target = cross_attn_block.v_fusion(target_seq).view(b, -1, n, d)

        q = rope_apply(q, src_grid_sizes, src_freqs)
        k_target = rope_apply(k_target, target_grid_sizes, target_freqs)

        target_x = flash_attention(q, k_target, v_target, k_lens=target_seq_lens)
        x = x + target_x
        x = x.flatten(2)
        x = cross_attn_block.o(x)
        return x


class FusionModel(nn.Module):
    def __init__(self, video_config=None, audio_config=None):
        super().__init__()
        if video_config is None or audio_config is None:
            raise ValueError("video_config and audio_config must be provided for FusionModel")

        self.branch_meta = {}
        self._init_branch_modules("vid", video_config, is_audio=False)
        self._init_branch_modules("audio", audio_config, is_audio=True)

        assert (
            self.branch_meta["vid"]["num_layers"] == self.branch_meta["audio"]["num_layers"]
        ), "Video and audio branches must have the same number of layers"
        self.num_blocks = self.branch_meta["vid"]["num_layers"]

        video_block_cfg = self._make_block_cfg("vid")
        audio_block_cfg = self._make_block_cfg("audio")
        self.single_fusion_blocks = nn.ModuleList(
            [FusionModelBlock(video_block_cfg, audio_block_cfg) for _ in range(self.num_blocks)]
        )

        self.inject_cross_attention_kv_projections()
        self.init_weights()
        self.gradient_checkpointing = False

    def inject_cross_attention_kv_projections(self):
        for fusion_block in self.single_fusion_blocks:
            vid_block = fusion_block.video_block
            audio_block = fusion_block.audio_block
            vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = (
                WanRMSNorm(vid_block.dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()
            )

            audio_block.cross_attn.k_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.v_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(audio_block.dim, elementwise_affine=True)
            audio_block.cross_attn.norm_k_fusion = (
                WanRMSNorm(audio_block.dim, eps=1e-6) if audio_block.qk_norm else nn.Identity()
            )

    def merge_kwargs(self, vid_kwargs, audio_kwargs):
        merged_kwargs = {}
        for key in vid_kwargs:
            merged_kwargs[f"vid_{key}"] = vid_kwargs[key]
        for key in audio_kwargs:
            merged_kwargs[f"audio_{key}"] = audio_kwargs[key]
        return merged_kwargs

    def forward(
        self,
        vid,
        audio,
        t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        clip_fea=None,
        clip_fea_audio=None,
        y=None,
        first_frame_is_clean=False,
        slg_layer=False,
    ):
        assert clip_fea is None
        assert y is None

        if vid is None or audio is None:
            raise ValueError("FusionModel now requires both video and audio inputs")

        vid, vid_e, vid_kwargs = self._prepare_branch_inputs(
            prefix="vid",
            x=vid,
            t=t,
            context=vid_context,
            seq_len=vid_seq_len,
            clip_fea=clip_fea,
            y=y,
            first_frame_is_clean=first_frame_is_clean,
        )

        audio, audio_e, audio_kwargs = self._prepare_branch_inputs(
            prefix="audio",
            x=audio,
            t=t,
            context=audio_context,
            seq_len=audio_seq_len,
            clip_fea=clip_fea_audio,
            y=None,
            first_frame_is_clean=False,
        )

        kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)

        for i in range(self.num_blocks):
            if slg_layer > 0 and i == slg_layer:
                continue
            fusion_block = self.single_fusion_blocks[i]
            vid, audio = gradient_checkpointing(
                enabled=(self.training and self.gradient_checkpointing),
                module=fusion_block,
                vid=vid,
                audio=audio,
                **kwargs,
            )

        vid = self._post_branch_output("vid", vid, vid_kwargs["grid_sizes"], vid_e)
        audio = self._post_branch_output("audio", audio, audio_kwargs["grid_sizes"], audio_e)

        return vid, audio

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        vid_patch = getattr(self, "vid_patch_embedding", None)
        if isinstance(vid_patch, nn.Conv3d):
            nn.init.xavier_uniform_(vid_patch.weight.flatten(1))

        audio_patch = getattr(self, "audio_patch_embedding", None)
        if audio_patch is not None:
            for submodule in audio_patch.modules():
                if isinstance(submodule, (nn.Conv1d, ChannelLastConv1d)):
                    nn.init.kaiming_normal_(submodule.weight, nonlinearity="linear")

    def set_rope_params(self):
        self._set_branch_rope("vid")
        self._set_branch_rope("audio")

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _init_branch_modules(self, prefix, config, is_audio):
        meta = {
            "is_audio": is_audio,
            "model_type": config.get("model_type", "t2v" if not is_audio else "t2a"),
            "patch_size": config.get("patch_size", (1, 2, 2) if not is_audio else (1,)),
            "text_len": config.get("text_len", 512),
            "in_dim": config.get("in_dim"),
            "dim": config.get("dim"),
            "ffn_dim": config.get("ffn_dim"),
            "freq_dim": config.get("freq_dim", 256),
            "text_dim": config.get("text_dim", 4096),
            "out_dim": config.get("out_dim", 16),
            "num_heads": config.get("num_heads"),
            "num_layers": config.get("num_layers"),
            "window_size": config.get("window_size", (-1, -1)),
            "qk_norm": config.get("qk_norm", True),
            "cross_attn_norm": config.get("cross_attn_norm", True),
            "gradient_checkpointing": config.get("gradient_checkpointing", False),
            "temporal_rope_scaling_factor": config.get("temporal_rope_scaling_factor", 1.0),
            "eps": config.get("eps", 1e-6),
            "additional_emb_dim": config.get("additional_emb_dim"),
            "additional_emb_length": config.get("additional_emb_length"),
        }

        if meta["model_type"] in ["t2v", "t2a", "ti2v"]:
            cross_attn_type = "t2v_cross_attn"
        else:
            cross_attn_type = "i2v_cross_attn"
        if meta["model_type"] in ["i2v", "tt2a"]:
            if meta["additional_emb_dim"] is None or meta["additional_emb_length"] is None:
                raise ValueError("additional_emb_dim/length must be provided for image-conditioned models")
        else:
            meta["additional_emb_dim"] = None
            meta["additional_emb_length"] = None

        meta["cross_attn_type"] = cross_attn_type
        self.branch_meta[prefix] = meta

        if is_audio:
            patch_embedding = nn.Sequential(
                ChannelLastConv1d(meta["in_dim"], meta["dim"], kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(meta["dim"], meta["dim"] * 4, kernel_size=7, padding=3),
            )
        else:
            patch_embedding = nn.Conv3d(
                in_channels=meta["in_dim"],
                out_channels=meta["dim"],
                kernel_size=meta["patch_size"],
                stride=meta["patch_size"],
            )
        setattr(self, f"{prefix}_patch_embedding", patch_embedding)

        text_embedding = nn.Sequential(
            nn.Linear(meta["text_dim"], meta["dim"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(meta["dim"], meta["dim"]),
        )
        setattr(self, f"{prefix}_text_embedding", text_embedding)

        time_embedding = nn.Sequential(
            nn.Linear(meta["freq_dim"], meta["dim"]),
            nn.SiLU(),
            nn.Linear(meta["dim"], meta["dim"]),
        )
        setattr(self, f"{prefix}_time_embedding", time_embedding)

        time_projection = nn.Sequential(nn.SiLU(), nn.Linear(meta["dim"], meta["dim"] * 6))
        setattr(self, f"{prefix}_time_projection", time_projection)

        if meta["model_type"] in ["i2v", "tt2a"]:
            img_emb = nn.Linear(meta["additional_emb_dim"], meta["dim"])
            setattr(self, f"{prefix}_img_emb", img_emb)
        else:
            setattr(self, f"{prefix}_img_emb", None)

        head = Head(meta["dim"], meta["out_dim"], meta["patch_size"], meta["eps"])
        setattr(self, f"{prefix}_head", head)

        freqs = torch.zeros(1)
        self.register_buffer(f"{prefix}_freqs", freqs, persistent=False)

    def _make_block_cfg(self, prefix):
        meta = self.branch_meta[prefix]
        return {
            "cross_attn_type": meta["cross_attn_type"],
            "dim": meta["dim"],
            "ffn_dim": meta["ffn_dim"],
            "num_heads": meta["num_heads"],
            "window_size": meta["window_size"],
            "sink_size": meta.get("sink_size", 2),
            "qk_norm": meta["qk_norm"],
            "cross_attn_norm": meta["cross_attn_norm"],
            "eps": meta["eps"],
            "additional_emb_length": meta["additional_emb_length"],
        }

    def _prepare_branch_inputs(
        self,
        prefix,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        first_frame_is_clean=False,
    ):
        meta = self.branch_meta[prefix]
        patch_embedding = getattr(self, f"{prefix}_patch_embedding")
        text_embedding = getattr(self, f"{prefix}_text_embedding")
        time_embedding = getattr(self, f"{prefix}_time_embedding")
        time_projection = getattr(self, f"{prefix}_time_projection")
        img_emb = getattr(self, f"{prefix}_img_emb")
        freqs = getattr(self, f"{prefix}_freqs")

        device = next(patch_embedding.parameters()).device
        if freqs.device != device:
            freqs = freqs.to(device)
            setattr(self, f"{prefix}_freqs", freqs)
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     print("DEBUG 448", freqs.shape)
        data = x
        if y is not None:
            data = [torch.cat([u, v], dim=0) for u, v in zip(data, y)]

        embeddings = []
        grid_sizes = []
        seq_lens_list = []
        for item in data:
            emb = patch_embedding(item.unsqueeze(0))
            if meta["is_audio"]:
                grid = torch.tensor(emb.shape[1:2], dtype=torch.long, device=device)
            else:
                grid = torch.tensor(emb.shape[2:], dtype=torch.long, device=device)
                emb = emb.flatten(2).transpose(1, 2)
            seq_lens_list.append(emb.size(1))
            grid_sizes.append(grid)
            embeddings.append(emb)

        grid_sizes = torch.stack(grid_sizes)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.long, device=device)
        if seq_lens.max().item() > seq_len:
            raise ValueError(f"Sequence length {seq_lens.max().item()} exceeds maximum {seq_len}.")

        padded_embeddings = []
        for emb in embeddings:
            if emb.size(1) < seq_len:
                pad = emb.new_zeros(emb.size(0), seq_len - emb.size(1), emb.size(2))
                emb = torch.cat([emb, pad], dim=1)
            padded_embeddings.append(emb)
        x_tensor = torch.cat(padded_embeddings)

        if t.dim() == 1:
            if first_frame_is_clean:
                t = torch.ones((t.size(0), seq_len), device=t.device, dtype=t.dtype) * t.unsqueeze(1)
                if meta["is_audio"]:
                    zero_lens = grid_sizes.squeeze(-1)
                else:
                    zero_lens = grid_sizes[:, 1:].prod(-1)
                for i in range(t.size(0)):
                    t[i, : zero_lens[i]] = 0
            else:
                t = t.unsqueeze(1).expand(t.size(0), seq_len)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            bt = t.size(0)
            t_flat = t.flatten()
            e = time_embedding(
                sinusoidal_embedding_1d(meta["freq_dim"], t_flat)
                .unflatten(0, (bt, seq_len))
                .bfloat16()
            )
            e0 = time_projection(e).unflatten(2, (6, meta["dim"]))

        context_tensors = torch.stack(
            [
                torch.cat(
                    [u, u.new_zeros(meta["text_len"] - u.size(0), u.size(1))],
                    dim=0,
                )
                for u in context
            ]
        ).to(device)
        context_emb = text_embedding(context_tensors)
        context_lens = None

        if clip_fea is not None and img_emb is not None:
            context_clip = img_emb(clip_fea)
            context_emb = torch.concat([context_clip, context_emb], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context_emb,
            context_lens=context_lens,
        )
        return x_tensor, e, kwargs

    def _post_branch_output(self, prefix, x, grid_sizes, e):
        meta = self.branch_meta[prefix]
        head = getattr(self, f"{prefix}_head")
        x = head(x, e)
        if meta["is_audio"]:
            grid_sizes_list = [gs[0].item() for gs in grid_sizes]
            outputs = [u[:gs] for u, gs in zip(x, grid_sizes_list)]
        else:
            outputs = self._unpatchify(x, grid_sizes, meta["patch_size"], meta["out_dim"])
        return [u.bfloat16() for u in outputs]

    @staticmethod
    def _unpatchify(x, grid_sizes, patch_size, out_dim):
        out = []
        for tensor, grid in zip(x, grid_sizes.tolist()):
            tensor = tensor[: math.prod(grid)].view(*grid, *patch_size, out_dim)
            tensor = torch.einsum("fhwpqrc->cfphqwr", tensor)
            tensor = tensor.reshape(out_dim, *[g * p for g, p in zip(grid, patch_size)])
            out.append(tensor)
        return out

    def _set_branch_rope(self, prefix):
        meta = self.branch_meta[prefix]
        dim = meta["dim"]
        num_heads = meta["num_heads"]
        d = dim // num_heads
        if meta["is_audio"]:
            freqs = rope_params(
                1024,
                d - 4 * (d // 6),
                freqs_scaling=meta["temporal_rope_scaling_factor"],
            )
        else:
            freqs = torch.cat(
                [
                    rope_params(1024, d - 4 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ],
                dim=1,
            )
        self._buffers[f"{prefix}_freqs"] = freqs