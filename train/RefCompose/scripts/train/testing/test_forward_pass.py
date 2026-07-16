#!/usr/bin/env python3
"""
Verify Flux2Transformer2DModelCond forward (with cond_hidden_states) matches train.py wiring.

Uses a *tiny* random-init model (1 double + 1 single block) so CI/CPU can run without checkpoint weights.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

_TRAIN_DIR = Path(__file__).resolve().parent.parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

from src.flux2_train_helpers import prepare_subject_latent_ids, unpack_main_latents
from src.flux2_transformer_cond import Flux2Transformer2DModelCond


def _tiny_model(device: torch.device, dtype: torch.dtype) -> Flux2Transformer2DModelCond:
    """Small enough for a quick forward on CPU; dims match in_channels=128 packing.

    ``attention_head_dim`` must match sum(default ``axes_dims_rope``) == 128 (Flux2PosEmbed).
    """
    joint_dim = 256
    heads, head_dim = 2, 128
    return Flux2Transformer2DModelCond(
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        joint_attention_dim=joint_dim,
        in_channels=128,
        out_channels=128,
        timestep_guidance_channels=256,
        mlp_ratio=2.0,
    ).to(device=device, dtype=dtype)


def _training_like_tensors(
    *,
    device: torch.device,
    dtype: torch.dtype,
    b: int,
    l_txt: int,
    h_lat: int,
    w_lat: int,
    joint_dim: int = 256,
):
    """Spatial latents (B,128,H,W) → pack → same layout as train.py (with cond)."""
    c = 128
    noisy_spatial = torch.randn(b, c, h_lat, w_lat, device=device, dtype=dtype)
    cond_spatial = torch.randn(b, c, h_lat, w_lat, device=device, dtype=dtype)
    packed_main = Flux2Pipeline._pack_latents(noisy_spatial)
    packed_cond = Flux2Pipeline._pack_latents(cond_spatial)

    enc = torch.randn(b, l_txt, joint_dim, device=device, dtype=dtype)
    txt_ids = torch.zeros(b, l_txt, 4, device=device, dtype=torch.long)
    for bi in range(b):
        for t in range(l_txt):
            txt_ids[bi, t, 0] = t

    ids_main = Flux2Pipeline._prepare_latent_ids(noisy_spatial).to(device)
    ids_sub = prepare_subject_latent_ids(cond_spatial, scale=10).to(device)
    img_ids = torch.cat([ids_main, ids_sub], dim=1)

    n_main = packed_main.shape[1]
    timestep = torch.full((b,), 0.5, device=device, dtype=dtype)
    guidance = torch.ones(b, device=device, dtype=dtype)
    return packed_main, packed_cond, enc, txt_ids, img_ids, timestep, guidance, n_main, noisy_spatial


@pytest.mark.parametrize("device_name", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_conditional_transformer_forward_and_backward(device_name: str):
    device = torch.device(device_name)
    dtype = torch.float32

    m = _tiny_model(device, dtype)
    m.train()

    B, l_txt, h, w = 1, 3, 2, 4
    packed_main, packed_cond, enc, txt_ids, img_ids, ts, gu, n_main, noisy_spatial = _training_like_tensors(
        device=device, dtype=dtype, b=B, l_txt=l_txt, h_lat=h, w_lat=w, joint_dim=256
    )

    out = m(
        hidden_states=packed_main,
        cond_hidden_states=packed_cond,
        encoder_hidden_states=enc,
        txt_ids=txt_ids,
        img_ids=img_ids,
        timestep=ts,
        guidance=gu,
        return_dict=False,
    )[0]

    assert out.shape[0] == B
    assert out.shape[1] == n_main, "Output sequence must be main image tokens only"
    assert out.shape[2] == 128
    assert torch.isfinite(out).all()

    main_ids = Flux2Pipeline._prepare_latent_ids(noisy_spatial).to(out.device)
    spatial_pred = unpack_main_latents(out, main_ids)
    assert spatial_pred.shape == (B, 128, h, w)

    noise = torch.randn_like(noisy_spatial)
    target = noise - noisy_spatial
    loss = (spatial_pred.float() - target.float()).pow(2).mean()
    loss.backward()
    n_grad = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grad > 0, "Expected some gradients after backward"


def test_forward_without_cond_still_runs():
    """No cond path: cond_hidden_states=None (sanity)."""
    device = torch.device("cpu")
    dtype = torch.float32
    m = _tiny_model(device, dtype).eval()

    B, l_txt, h, w = 1, 2, 2, 2
    noisy_spatial = torch.randn(B, 128, h, w, dtype=dtype)
    packed_main = Flux2Pipeline._pack_latents(noisy_spatial)
    enc = torch.randn(B, l_txt, 256, dtype=dtype)
    txt_ids = torch.zeros(B, l_txt, 4, dtype=torch.long)
    ids_main = Flux2Pipeline._prepare_latent_ids(noisy_spatial)
    ts = torch.full((B,), 0.3, dtype=dtype)
    gu = torch.ones(B, dtype=dtype)

    with torch.no_grad():
        out = m(
            hidden_states=packed_main,
            cond_hidden_states=None,
            encoder_hidden_states=enc,
            txt_ids=txt_ids,
            img_ids=ids_main,
            timestep=ts,
            guidance=gu,
            return_dict=False,
        )[0]
    assert out.shape[1] == packed_main.shape[1]
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    not os.environ.get("FLUX2_FORWARD_FULL_WEIGHTS"),
    reason="Set FLUX2_FORWARD_FULL_WEIGHTS=1 and FLUX2_MODEL_DIR to run full DiT forward (large, needs GPU RAM).",
)
def test_full_checkpoint_forward_optional():
    """Optional: load real FLUX.2 transformer weights and run one forward (matches train stack)."""
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel as Flux2Base
    from src.transformer_flux import FluxTransformer2DModel

    root = os.environ.get("FLUX2_MODEL_DIR", "").strip()
    if not root or not Path(root).is_dir():
        pytest.skip("FLUX2_MODEL_DIR invalid")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    base = Flux2Base.from_pretrained(root, subfolder="transformer", torch_dtype=dtype)
    model = FluxTransformer2DModel.from_config(base.config)
    model.load_state_dict(base.state_dict(), strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    del base

    # Small spatial grid to save memory
    B, l_txt, h, w = 1, 8, 4, 4
    packed_main, packed_cond, enc, txt_ids, img_ids, ts, gu, n_main, _ = _training_like_tensors(
        device=device, dtype=dtype, b=B, l_txt=l_txt, h_lat=h, w_lat=w, joint_dim=model.config.joint_attention_dim
    )

    with torch.no_grad():
        out = model(
            hidden_states=packed_main,
            cond_hidden_states=packed_cond,
            encoder_hidden_states=enc,
            txt_ids=txt_ids,
            img_ids=img_ids,
            timestep=ts,
            guidance=gu,
            return_dict=False,
        )[0]

    assert out.shape[1] == n_main
    assert torch.isfinite(out.float()).all()
