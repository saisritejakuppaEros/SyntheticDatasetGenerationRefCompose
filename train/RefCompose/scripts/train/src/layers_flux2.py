# LoRA attention processors for FLUX.2 (Flux2Attention + Flux2ParallelSelfAttention) with canvas/subject masking.

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2AttnProcessor,
    Flux2ParallelSelfAttention,
    Flux2ParallelSelfAttnProcessor,
)


class LoRALinearLayerFlux2(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        cond_width: int = 512,
        cond_height: int = 512,
        number: int = 0,
        n_loras: int = 1,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.network_alpha = network_alpha
        self.rank = rank
        self.in_features = in_features
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        self.cond_height = cond_height
        self.cond_width = cond_width
        self.number = number
        self.n_loras = n_loras

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        batch_size = hidden_states.shape[0]
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        block_size = hidden_states.shape[1] - cond_size * self.n_loras
        mask = torch.ones(
            (batch_size, hidden_states.shape[1], self.in_features), device=hidden_states.device, dtype=dtype
        )
        mask[:, : block_size + self.number * cond_size, :] = 0
        mask[:, block_size + (self.number + 1) * cond_size :, :] = 0
        hidden_states = mask * hidden_states
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank
        return up_hidden_states.to(orig_dtype)


class MultiDoubleStreamBlockFlux2LoraProcessor(nn.Module):
    def __init__(
        self,
        dim: int,
        ranks: List[int],
        network_alphas: List[int],
        lora_weights: List[float],
        device=None,
        dtype=None,
        cond_width: int = 512,
        cond_height: int = 512,
        n_loras: int = 1,
    ):
        super().__init__()
        self.n_loras = n_loras
        self.cond_width = cond_width
        self.cond_height = cond_height
        self.q_loras = nn.ModuleList(
            [
                LoRALinearLayerFlux2(dim, dim, ranks[i], network_alphas[i], device, dtype, cond_width, cond_height, i, n_loras)
                for i in range(n_loras)
            ]
        )
        self.k_loras = nn.ModuleList(
            [
                LoRALinearLayerFlux2(dim, dim, ranks[i], network_alphas[i], device, dtype, cond_width, cond_height, i, n_loras)
                for i in range(n_loras)
            ]
        )
        self.v_loras = nn.ModuleList(
            [
                LoRALinearLayerFlux2(dim, dim, ranks[i], network_alphas[i], device, dtype, cond_width, cond_height, i, n_loras)
                for i in range(n_loras)
            ]
        )
        self.proj_loras = nn.ModuleList(
            [
                LoRALinearLayerFlux2(dim, dim, ranks[i], network_alphas[i], device, dtype, cond_width, cond_height, i, n_loras)
                for i in range(n_loras)
            ]
        )
        self.lora_weights = lora_weights

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
        batch_size = hidden_states.shape[0]
        inner_dim = attn.inner_dim
        head_dim = inner_dim // attn.heads

        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        # Same layout as Flux2AttnProcessor: (batch, seq, heads, dim) — required by dispatch_attention_fn
        # (it permutes to (B, H, S, D) for SDPA internally).
        eq = encoder_hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim)
        ek = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim)
        ev = encoder_hidden_states_value_proj.view(batch_size, -1, attn.heads, head_dim)
        if attn.norm_added_q is not None:
            eq = attn.norm_added_q(eq)
        if attn.norm_added_k is not None:
            ek = attn.norm_added_k(ek)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        for i in range(self.n_loras):
            query = query + self.lora_weights[i] * self.q_loras[i](hidden_states)
            key = key + self.lora_weights[i] * self.k_loras[i](hidden_states)
            value = value + self.lora_weights[i] * self.v_loras[i](hidden_states)

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = torch.cat([eq, query], dim=1)
        key = torch.cat([ek, key], dim=1)
        value = torch.cat([ev, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        scaled_seq_len = query.shape[1]
        scaled_cond_size = cond_size
        # Rows that may attend to all keys: all text + main image tokens (exclude cond canvas tokens).
        if use_cond:
            scaled_block_size = encoder_hidden_states.shape[1] + main_seq_len
        else:
            scaled_block_size = scaled_seq_len - cond_size * self.n_loras

        mask = torch.ones((scaled_seq_len, scaled_seq_len), device=hidden_states.device)
        mask[:scaled_block_size, :] = 0
        for i in range(self.n_loras):
            start = i * scaled_cond_size + scaled_block_size
            end = (i + 1) * scaled_cond_size + scaled_block_size
            mask[start:end, start:end] = 0
        mask = mask * -1e20
        mask = mask.to(query.dtype)

        out = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=mask,
            backend=Flux2AttnProcessor._attention_backend,
            parallel_config=Flux2AttnProcessor._parallel_config,
        )
        out = out.flatten(2, 3).to(query.dtype)

        enc_len = encoder_hidden_states.shape[1]
        img_len = out.shape[1] - enc_len
        encoder_part, img_part = out.split_with_sizes([enc_len, img_len], dim=1)
        encoder_part = attn.to_add_out(encoder_part)

        if use_cond:
            cond_len = img_len - main_seq_len
            main_part, cond_part = img_part.split_with_sizes([main_seq_len, cond_len], dim=1)
            main_part = attn.to_out[0](main_part)
            for i in range(self.n_loras):
                main_part = main_part + self.lora_weights[i] * self.proj_loras[i](main_part)
            main_part = attn.to_out[1](main_part)
            cond_part = attn.to_out[0](cond_part)
            cond_part = attn.to_out[1](cond_part)
            return main_part, encoder_part, cond_part

        hidden_states = attn.to_out[0](img_part)
        for i in range(self.n_loras):
            hidden_states = hidden_states + self.lora_weights[i] * self.proj_loras[i](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states, encoder_part


class MultiSingleStreamBlockFlux2LoraProcessor(nn.Module):
    def __init__(
        self,
        dim: int,
        inner_dim: int,
        mlp_hidden_dim: int,
        mlp_mult_factor: int,
        ranks: List[int],
        network_alphas: List[int],
        lora_weights: List[float],
        device=None,
        dtype=None,
        cond_width: int = 512,
        cond_height: int = 512,
        n_loras: int = 1,
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_mult_factor = mlp_mult_factor
        self.n_loras = n_loras
        self.cond_width = cond_width
        self.cond_height = cond_height
        fused_in = inner_dim * 3 + mlp_hidden_dim * mlp_mult_factor
        self.qkv_mlp_loras = nn.ModuleList(
            [
                LoRALinearLayerFlux2(dim, fused_in, ranks[i], network_alphas[i], device, dtype, cond_width, cond_height, i, n_loras)
                for i in range(n_loras)
            ]
        )
        self.lora_weights = lora_weights

    def __call__(
        self,
        attn: Flux2ParallelSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond: bool = False,
        main_seq_len: int = 0,
    ):
        h = attn.to_qkv_mlp_proj(hidden_states)
        for i in range(self.n_loras):
            h = h + self.lora_weights[i] * self.qkv_mlp_loras[i](hidden_states)

        qkv, mlp_hidden_states = torch.split(
            h, [3 * self.inner_dim, self.mlp_hidden_dim * self.mlp_mult_factor], dim=-1
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

        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        # query layout matches Flux2ParallelSelfAttnProcessor: (batch, seq, heads, dim)
        scaled_seq_len = query.shape[1]
        if use_cond:
            scaled_block_size = main_seq_len
        else:
            scaled_block_size = hidden_states.shape[1] - cond_size * self.n_loras
        scaled_cond_size = cond_size

        mask = torch.ones((scaled_seq_len, scaled_seq_len), device=hidden_states.device)
        mask[:scaled_block_size, :] = 0
        for i in range(self.n_loras):
            start = i * scaled_cond_size + scaled_block_size
            end = (i + 1) * scaled_cond_size + scaled_block_size
            mask[start:end, start:end] = 0
        mask = mask * -1e20
        mask = mask.to(query.dtype)

        attn_out = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=mask,
            backend=Flux2ParallelSelfAttnProcessor._attention_backend,
            parallel_config=Flux2ParallelSelfAttnProcessor._parallel_config,
        )
        attn_out = attn_out.flatten(2, 3).to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        fused = torch.cat([attn_out, mlp_hidden_states], dim=-1)
        fused = attn.to_out(fused)

        if use_cond:
            return fused[:, :main_seq_len], fused[:, main_seq_len:]
        return fused
