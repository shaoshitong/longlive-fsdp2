
import torch
import torch.nn as nn
from ovi.modules.model_casual import WanLayerNorm, CausalWanModel, WanRMSNorm, gradient_checkpointing, rope_apply
from ovi.modules.attention import flash_attention


class CasualFusionModelBlock(nn.Module):
    def __init__(self, video_block, audio_block):
        super().__init__()
        self.video_block = video_block
        self.audio_block = audio_block

    def forward(self,vid,
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
                    vid_block_mask,
                    audio_block_mask,
                    vid_kv_cache = None,
                    vid_crossattn_cache = None,
                    vid_current_start = None,
                    vid_cache_start = None,
                    audio_kv_cache = None,
                    audio_crossattn_cache = None,
                    audio_current_start = None,
                    audio_cache_start = None):
        audio_num_frames, audio_frame_seqlen = audio_e.shape[1], audio.shape[1] // audio_e.shape[1]
        vid_num_frames, vid_frame_seqlen = vid_e.shape[1], vid.shape[1] // vid_e.shape[1]
        ## audio modulation
        assert audio_e.dtype == torch.bfloat16
        assert len(audio_e.shape) == 4 and audio_e.size(2) == 6
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio_e = self.audio_block.modulation(audio_e).chunk(6, dim=2)
        assert audio_e[0].dtype == torch.bfloat16

        # audio self-attention
        audio_y = self.audio_block.self_attn(
            self.audio_block.norm1(audio).bfloat16().unflatten(dim=1, sizes=(audio_num_frames, audio_frame_seqlen)) * (1 + audio_e[1]) + audio_e[0], audio_seq_lens, audio_grid_sizes,
            audio_freqs, audio_block_mask, audio_kv_cache, audio_current_start, audio_cache_start)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio = audio + (audio_y.unflatten(dim=1, sizes=(audio_num_frames, audio_frame_seqlen)) * audio_e[2]).flatten(1, 2)

        ## video modulation
        assert vid_e.dtype == torch.bfloat16
        assert len(vid_e.shape) == 4 and vid_e.size(2) == 6
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid_e = self.video_block.modulation(vid_e).chunk(6, dim=2)
        assert vid_e[0].dtype == torch.bfloat16

        # video self-attention
        vid_y = self.video_block.self_attn(
            self.video_block.norm1(vid).bfloat16().unflatten(dim=1, sizes=(vid_num_frames, vid_frame_seqlen)) * (1 + vid_e[1]) + vid_e[0], vid_seq_lens, vid_grid_sizes,
            vid_freqs, vid_block_mask, vid_kv_cache, vid_current_start, vid_cache_start)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid = vid + (vid_y.unflatten(dim=1, sizes=(vid_num_frames, vid_frame_seqlen)) * vid_e[2]).flatten(1, 2)

        og_audio = audio

        # audio cross-attention
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
            audio_crossattn_cache,
        )

        assert not torch.equal(og_audio, audio), "Audio should be changed after cross-attention!"

        # video cross-attention
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
            vid_crossattn_cache,
        )

        return vid, audio

    def single_fusion_cross_attention_ffn_forward(self,
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
                                            crossattn_cache = None):

        num_frames, frame_seqlen = src_e.shape[1], src_seq.shape[1] // src_e.shape[1]
        src_seq = src_seq + self.single_fusion_cross_attention_forward(attn_block.cross_attn,
                                                                       attn_block.norm3(src_seq),
                                                                       src_grid_sizes=src_grid_sizes,
                                                                       src_freqs=src_freqs,
                                                                       target_seq=target_seq,
                                                                       target_seq_lens=target_seq_lens,
                                                                       target_grid_sizes=target_grid_sizes,
                                                                       target_freqs=target_freqs,
                                                                       context=context,
                                                                       context_lens=context_lens,
                                                                       crossattn_cache=crossattn_cache
                                                                       )
        y = attn_block.ffn((attn_block.norm2(src_seq).bfloat16().unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + src_e[4]) + src_e[3]).flatten(1, 2))
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            src_seq = src_seq + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * src_e[5]).flatten(1, 2)
        return src_seq


    def single_fusion_cross_attention_forward(self,
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
                                            crossattn_cache = None,
                                            ):
        b, n, d = src_seq.size(0), cross_attn_block.num_heads, cross_attn_block.head_dim
        context_img = context[:, : cross_attn_block.additional_emb_length]
        context = context[:, cross_attn_block.additional_emb_length :]

        q = cross_attn_block.norm_q(cross_attn_block.q(src_seq)).view(b, -1, n, d)
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = cross_attn_block.norm_k(cross_attn_block.k(context)).view(b, -1, n, d)
                v = cross_attn_block.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = cross_attn_block.norm_k(cross_attn_block.k(context)).view(b, -1, n, d)
            v = cross_attn_block.v(context).view(b, -1, n, d)

        if hasattr(cross_attn_block, "k_img"):
            k_img = cross_attn_block.norm_k_img(cross_attn_block.k_img(context_img)).view(b, -1, n, d)
            v_img = cross_attn_block.v_img(context_img).view(b, -1, n, d)
        else:
            k_img = v_img = None

        x = flash_attention(q, k, v, k_lens=context_lens)

        if k_img is not None:
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
            x = x + img_x
        # compute target attention
        target_seq = cross_attn_block.pre_attn_norm_fusion(target_seq)
        k_target = cross_attn_block.norm_k_fusion(cross_attn_block.k_fusion(target_seq)).view(b, -1, n, d)
        v_target = cross_attn_block.v_fusion(target_seq).view(b, -1, n, d)

        q = rope_apply(q, src_grid_sizes, src_freqs)
        k_target = rope_apply(k_target, target_grid_sizes, target_freqs)
        
        target_x = flash_attention(q, k_target, v_target, k_lens=target_seq_lens)
        
        x = x + target_x

        x = x.flatten(2) # [B, L/P, C]

        x = cross_attn_block.o(x)
        return x

class CausalFusionModel(nn.Module):
    def __init__(self, video_config=None, audio_config=None, local_attn_size=6, sink_size=2):
        super().__init__()
        has_video = True 
        has_audio = True
        if video_config is not None:
            video_config['sink_size'] = sink_size
            self.video_model = CausalWanModel(**video_config)
            self.video_model.set_local_window_size(local_window_size=local_attn_size, frame_seqlen=900)
        else:
            has_video = False
            self.video_model = None
            print("Warning: No video model is provided!")
        
        if audio_config is not None:
            audio_config['sink_size'] = sink_size
            self.audio_model = CausalWanModel(**audio_config)
            self.audio_model.set_local_window_size(local_window_size=local_attn_size, frame_seqlen=1)
        else:
            has_audio = False
            self.audio_model = None
            print("Warning: No audio model is provided!")

        if has_video and has_audio:
            assert len(self.video_model.blocks) == len(self.audio_model.blocks)
            self.num_blocks = len(self.video_model.blocks)
            self.inject_cross_attention_kv_projections()

        self.init_weights()
        self.gradient_checkpointing = False
        self.single_fusion_blocks = nn.ModuleList([CasualFusionModelBlock(self.video_model.blocks[i], self.audio_model.blocks[i]) for i in range(self.num_blocks)])
        
    def inject_cross_attention_kv_projections(self):
        for vid_block in self.video_model.blocks:
            vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = WanRMSNorm(vid_block.dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()

        
        for audio_block in self.audio_model.blocks:
            audio_block.cross_attn.k_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.v_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(audio_block.dim, elementwise_affine=True)
            audio_block.cross_attn.norm_k_fusion = WanRMSNorm(audio_block.dim, eps=1e-6) if audio_block.qk_norm else nn.Identity()


    def merge_kwargs(self, vid_kwargs, audio_kwargs):
        """
        keys in each kwarg:
        e
        seq_lens
        grid_sizes
        freqs
        context
        context_lens
        """
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
        vid_t,
        audio_t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
            clip_fea=None,
            clip_fea_audio=None,
            y=None,
            first_frame_is_clean=False,
            slg_layer=False,
        vid_kv_cache = None,
        vid_crossattn_cache = None,
        vid_current_start = None,
        vid_cache_start = None,
        audio_kv_cache = None,
        audio_crossattn_cache = None,
        audio_current_start = None,
        audio_cache_start = None
    ):  
        # (60,104)/2 -> 1560
        assert clip_fea is None 
        assert y is None

        if vid is None or all([x is None for x in vid]):
            assert vid_context is None
            assert vid_seq_len is None
            assert self.audio_model is not None

            return None, self.audio_model(x=audio, t=audio_t, context=audio_context, seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None, kv_cache = audio_kv_cache, crossattn_cache = audio_crossattn_cache, current_start = audio_current_start, cache_start = audio_cache_start)
        
        if audio is None or all([x is None for x in audio]):
            assert clip_fea_audio is None
            assert audio_context is None
            assert audio_seq_len is None
            assert self.video_model is not None

            return self.video_model(x=vid, t=vid_t, context=vid_context, seq_len=vid_seq_len, clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean, kv_cache = vid_kv_cache, crossattn_cache = vid_crossattn_cache, current_start = vid_current_start, cache_start = vid_cache_start), None
        
        vid, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
            x=vid, t=vid_t, context=vid_context, seq_len=vid_seq_len, clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean
        )

        audio, audio_e, audio_kwargs = self.audio_model.prepare_transformer_block_kwargs(
            x=audio, t=audio_t, context=audio_context, seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None, first_frame_is_clean=False
        )

        kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)

        for i in range(self.num_blocks):
            """
            1 fusion block refers to 1 audio block with 1 video block.
            """
            if slg_layer > 0 and i == slg_layer:
                continue
            vid_block = self.video_model.blocks[i]
            audio_block = self.audio_model.blocks[i]
            kwargs.update(
                {
                    "vid_kv_cache": vid_kv_cache[i],
                    "vid_current_start": vid_current_start,
                    "vid_cache_start": vid_cache_start,
                    "audio_kv_cache": audio_kv_cache[i],
                    "audio_current_start": audio_current_start,
                    "audio_cache_start": audio_cache_start,
                }
            )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "vid_crossattn_cache": vid_crossattn_cache[i],
                        "audio_crossattn_cache": audio_crossattn_cache[i],
                    }
                )
            vid, audio = gradient_checkpointing(
                    enabled=(self.training and self.gradient_checkpointing),
                    module=self.single_fusion_blocks[i],
                    vid=vid,
                    audio=audio,
                    **kwargs
                )

        vid = self.video_model.post_transformer_block_out(vid, vid_kwargs['grid_sizes'], vid_e, vid_t)
        audio = self.audio_model.post_transformer_block_out(audio, audio_kwargs['grid_sizes'], audio_e, audio_t)

        return vid, audio

    def init_weights(self):
        if self.audio_model is not None:
            self.audio_model.init_weights()

        if self.video_model is not None:
            self.video_model.init_weights()

        for name, mod in self.video_model.named_modules():
            if "fusion" in name and isinstance(mod, nn.Linear):
                with torch.no_grad():
                    mod.weight.div_(10.0)

    
    def set_rope_params(self):
        self.video_model.set_rope_params()
        self.audio_model.set_rope_params()

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True