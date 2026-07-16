# FLUX 2 transformer with optional cond_hidden_states (canvas / subject tokens).

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2AttnProcessor,
    Flux2FeedForward,
    Flux2Modulation,
    Flux2ParallelSelfAttention,
    Flux2ParallelSelfAttnProcessor,
    Flux2PosEmbed,
    Flux2TimestepGuidanceEmbeddings,
    _get_qkv_projections,
)
from diffusers.utils import USE_PEFT_BACKEND, is_torch_npu_available, is_torch_version, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)


class CondFlux2AttnProcessor(Flux2AttnProcessor):
    """Splits image stream into main + cond after attention; expects cat(main, cond) on image side."""

    def __call__(
        self,
        attn: Flux2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond: bool = False,
        main_seq_len: int = 0,
    ):
        if not use_cond:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
        encoder_query = attn.norm_added_q(encoder_query)
        encoder_key = attn.norm_added_k(encoder_key)

        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states_out = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states_out = hidden_states_out.flatten(2, 3).to(query.dtype)

        enc_len = encoder_hidden_states.shape[1]
        img_len = hidden_states_out.shape[1] - enc_len
        encoder_part, img_part = hidden_states_out.split_with_sizes([enc_len, img_len], dim=1)
        encoder_part = attn.to_add_out(encoder_part)

        cond_len = img_len - main_seq_len
        main_part, cond_part = img_part.split_with_sizes([main_seq_len, cond_len], dim=1)

        main_part = attn.to_out[0](main_part)
        main_part = attn.to_out[1](main_part)
        cond_part = attn.to_out[0](cond_part)
        cond_part = attn.to_out[1](cond_part)

        return main_part, encoder_part, cond_part


class CondFlux2ParallelSelfAttnProcessor(Flux2ParallelSelfAttnProcessor):
    def __call__(
        self,
        attn: Flux2ParallelSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond: bool = False,
        main_seq_len: int = 0,
    ):
        if not use_cond:
            return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)

        hidden = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        attn_out = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn_out = attn_out.flatten(2, 3).to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        fused = torch.cat([attn_out, mlp_hidden_states], dim=-1)
        fused = attn.to_out(fused)

        out_tm = fused[:, :main_seq_len]
        out_c = fused[:, main_seq_len:]
        return out_tm, out_c


class Flux2TransformerBlockCond(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = Flux2Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
            processor=CondFlux2AttnProcessor(),
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cond_hidden_states: Optional[torch.Tensor],
        temb_mod_params_img: torch.Tensor,
        temb_mod_params_img_cond: Optional[torch.Tensor],
        temb_mod_params_txt: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        joint_attention_kwargs = joint_attention_kwargs or {}
        use_cond = cond_hidden_states is not None

        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(
            temb_mod_params_img, 2
        )
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = Flux2Modulation.split(
            temb_mod_params_txt, 2
        )

        norm_hidden_states = (1 + scale_msa) * self.norm1(hidden_states) + shift_msa

        if use_cond:
            (cnd_shift_msa, cnd_scale_msa, cnd_gate_msa), (cnd_shift_mlp, cnd_scale_mlp, cnd_gate_mlp) = (
                Flux2Modulation.split(temb_mod_params_img_cond, 2)
            )
            norm_cond = (1 + cnd_scale_msa) * self.norm1(cond_hidden_states) + cnd_shift_msa
            norm_img_in = torch.cat([norm_hidden_states, norm_cond], dim=1)
            attn_kwargs = {**joint_attention_kwargs, "use_cond": True, "main_seq_len": hidden_states.shape[1]}
        else:
            cnd_gate_msa = cnd_shift_mlp = cnd_scale_mlp = cnd_gate_mlp = None
            norm_img_in = norm_hidden_states
            attn_kwargs = dict(joint_attention_kwargs)

        norm_encoder_hidden_states = (1 + c_scale_msa) * self.norm1_context(encoder_hidden_states) + c_shift_msa

        attention_outputs = self.attn(
            hidden_states=norm_img_in,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attn_kwargs,
        )

        if use_cond:
            attn_output, context_attn_output, cond_attn_output = attention_outputs
        else:
            attn_output, context_attn_output = attention_outputs
            cond_attn_output = None

        hidden_states = hidden_states + gate_msa * attn_output

        if use_cond:
            cond_hidden_states = cond_hidden_states + cnd_gate_msa * cond_attn_output
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
            hidden_states = hidden_states + gate_mlp * self.ff(norm_hidden_states)

            norm_cond = self.norm2(cond_hidden_states)
            norm_cond = norm_cond * (1 + cnd_scale_mlp) + cnd_shift_mlp
            cond_hidden_states = cond_hidden_states + cnd_gate_mlp * self.ff(norm_cond)
        else:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
            hidden_states = hidden_states + gate_mlp * self.ff(norm_hidden_states)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(norm_encoder_hidden_states)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states, cond_hidden_states


class Flux2SingleTransformerBlockCond(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux2ParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
            processor=CondFlux2ParallelSelfAttnProcessor(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_hidden_states: Optional[torch.Tensor],
        temb_mod_params: torch.Tensor,
        temb_mod_params_cond: Optional[torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        use_cond = cond_hidden_states is not None
        joint_attention_kwargs = joint_attention_kwargs or {}

        mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod_params, 1)[0]
        norm_tm = (1 + mod_scale) * self.norm(hidden_states) + mod_shift

        if use_cond:
            c_shift, c_scale, c_gate = Flux2Modulation.split(temb_mod_params_cond, 1)[0]
            norm_c = (1 + c_scale) * self.norm(cond_hidden_states) + c_shift
            combined = torch.cat([norm_tm, norm_c], dim=1)
            attn_kwargs = {**joint_attention_kwargs, "use_cond": True, "main_seq_len": hidden_states.shape[1]}
        else:
            combined = norm_tm
            c_gate = None
            attn_kwargs = dict(joint_attention_kwargs)

        attn_output = self.attn(
            hidden_states=combined,
            image_rotary_emb=image_rotary_emb,
            **attn_kwargs,
        )

        if use_cond:
            out_tm, out_c = attn_output
            hidden_states = hidden_states + mod_gate * out_tm
            cond_hidden_states = cond_hidden_states + c_gate * out_c
            return hidden_states, cond_hidden_states

        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states, None


def _ckpt_double_block(
    block,
    hidden_states,
    encoder_hidden_states,
    cond_hidden_states,
    d_img,
    d_img_cond,
    d_txt,
    rope,
    jkw,
    use_cond: bool,
):
    return block(
        hidden_states,
        encoder_hidden_states,
        cond_hidden_states if use_cond else None,
        d_img,
        d_img_cond if use_cond else None,
        d_txt,
        rope,
        jkw,
    )


def _ckpt_single_block(
    block,
    hidden_states,
    cond_hidden_states,
    s_mod,
    s_mod_c,
    rope,
    jkw,
    use_cond: bool,
):
    return block(
        hidden_states,
        cond_hidden_states if use_cond else None,
        s_mod,
        s_mod_c if use_cond else None,
        rope,
        jkw,
    )


class Flux2Transformer2DModelCond(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["Flux2TransformerBlockCond", "Flux2SingleTransformerBlockCond"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["Flux2TransformerBlockCond", "Flux2SingleTransformerBlockCond"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "img_ids": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "txt_ids": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels, embedding_dim=self.inner_dim, bias=False
        )
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlockCond(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlockCond(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        cond_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        use_cond = cond_hidden_states is not None
        num_txt_tokens = encoder_hidden_states.shape[1]

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is None:
            guidance = torch.zeros_like(timestep)
        guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)
        zero_ts = torch.zeros_like(timestep)
        cond_temb = self.time_guidance_embed(zero_ts, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_img_cond = self.double_stream_modulation_img(cond_temb) if use_cond else None
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)
        single_stream_mod_cond = self.single_stream_modulation(cond_temb) if use_cond else None

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if use_cond:
            cond_hidden_states = self.x_embedder(cond_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        # Flux2Pipeline._prepare_latent_ids / _prepare_text_ids use CPU torch.arange; move ids to the
        # activations device so text_rotary_emb and image_rotary_emb match for torch.cat.
        rope_dev = hidden_states.device
        img_ids = img_ids.to(device=rope_dev)
        txt_ids = txt_ids.to(device=rope_dev)

        if is_torch_npu_available():
            freqs_cos_image, freqs_sin_image = self.pos_embed(img_ids.cpu())
            image_rotary_emb = (freqs_cos_image.npu(), freqs_sin_image.npu())
            freqs_cos_text, freqs_sin_text = self.pos_embed(txt_ids.cpu())
            text_rotary_emb = (freqs_cos_text.npu(), freqs_sin_text.npu())
        else:
            image_rotary_emb = self.pos_embed(img_ids)
            text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        ckpt_kw = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

        for block in self.transformer_blocks:
            cond_for_ckpt = cond_hidden_states if use_cond else hidden_states[:, :0, :]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states, cond_hidden_states = torch.utils.checkpoint.checkpoint(
                    _ckpt_double_block,
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    cond_for_ckpt,
                    double_stream_mod_img,
                    double_stream_mod_img_cond,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                    use_cond,
                    **ckpt_kw,
                )
                if not use_cond:
                    cond_hidden_states = None
            else:
                encoder_hidden_states, hidden_states, cond_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    cond_hidden_states if use_cond else None,
                    double_stream_mod_img,
                    double_stream_mod_img_cond,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for block in self.single_transformer_blocks:
            cond_single_ckpt = cond_hidden_states if use_cond else hidden_states[:, :0, :]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, cond_hidden_states = torch.utils.checkpoint.checkpoint(
                    _ckpt_single_block,
                    block,
                    hidden_states,
                    cond_single_ckpt,
                    single_stream_mod,
                    single_stream_mod_cond,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                    use_cond,
                    **ckpt_kw,
                )
                if not use_cond:
                    cond_hidden_states = None
            else:
                hidden_states, cond_hidden_states = block(
                    hidden_states,
                    cond_hidden_states if use_cond else None,
                    single_stream_mod,
                    single_stream_mod_cond,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )

        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
