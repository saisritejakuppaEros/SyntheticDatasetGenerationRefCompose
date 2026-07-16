#!/usr/bin/env python3
"""
Flux2 LoRA training — documented data flow + shape tests
==========================================================

This file explains ``train.py`` (one training step) and locks the geometry with
unit tests. Shapes below match a typical run like your terminal example::

    Target latent after encode_flux2_latents (main branch):   (1, 128, 68, 120)
    Canvas latent after encode_flux2_latents (subject branch): (1, 128, 68, 120)

Run::

    cd lora_training/v2/train && pytest testing/test_training_flow_documented.py -v

This file also runs **section 5** — one batch, full training-step loss + ``backward()`` on a tiny
random-init ``Flux2Transformer2DModelCond`` (small latent grid for speed; same pack/ids/unpack/loss
as ``train.py``).

End-to-end **transformer** forward (cond on/off), backward, and unpack — including CUDA::

    pytest testing/test_forward_pass.py -v

Optional full-weight DiT forward (needs ``FLUX2_MODEL_DIR``)::

    FLUX2_FORWARD_FULL_WEIGHTS=1 FLUX2_MODEL_DIR=/path/to/FLUX.2-dev pytest testing/test_forward_pass.py::test_full_checkpoint_forward_optional -v

---------------------------------------------------------------------------
1) What is frozen vs trained
---------------------------------------------------------------------------

- **VAE**: inference only; encodes RGB (normalized [-1,1]) → patchified latents.
- **Text encoder (Mistral3) + tokenizer (Pixtral)**: inference only; produces
  ``prompt_embeds`` and ``text_ids``.
- **Flux2 transformer**: base weights can stay in bf16/fp16; **only LoRA
  parameters** (low-rank matrices in attention processors) receive gradients
  from the loss. Optimizer is built over ``p.requires_grad`` on the transformer
  (see ``train.py``).

---------------------------------------------------------------------------
2) Per-step pipeline (batch size 1, with subject/canvas conditioning)
---------------------------------------------------------------------------

**A — Text**

- ``prompts`` → ``encode_prompts_flux2`` → ``encoder_hidden_states`` (prompt
  tokens) + ``text_ids`` (4D coords for text positions in the joint sequence).

**B — Images → latents (clean)**

- ``pixel_values`` (B,3,H,W) target scene → ``encode_flux2_latents(vae, …)``
  → ``model_input`` shape **(B, C, Hm, Wm)** e.g. (1, 128, 68, 120) when using unified 1920×1088 RGB.
- ``subject_pixel_values`` (B,3,H,W) conditioning canvas (same H,W as target when
  using default unified resolution) → ``subject_latents`` **(B, C, H, W)** matching
  ``model_input``.

``C=128`` here is **patchified latent channels** after VAE + ``_patchify_latents``
+ BN normalize (see ``src/flux2_train_helpers.py``).

**C — Main branch noise (flow-match scheduling)**

- Sample ``noise`` same shape as ``model_input``.
- Sample timestep index → ``sigmas`` from ``noise_scheduler_copy``.
- ``noisy_model_input = (1 - sigmas) * model_input + sigmas * noise``.
- **Only the main** latents are noised; canvas latents stay **clean**.

**D — Pack for transformer (sequence layout)**

- ``packed_noisy = _pack_latents(noisy)`` → **(B, Hm*Wm, C)** e.g. (1, 8160, 128).
- ``packed_cond = _pack_latents(subject_latents)`` → same sequence length when Hm=Wm and Hc=Wc.

**E — Position IDs (4D coords: time / height / width / layer)**

- ``latent_image_ids = _prepare_latent_ids(model_input)`` for the main image.
- ``prepare_subject_latent_ids(subject_latents)`` assigns a **different time
  slice** (T) so the transformer knows tokens belong to the conditioning image.
- ``img_ids = cat([main_ids, subject_ids], dim=1)`` → one long image token
  sequence for RoPE / dispatch.

**F — Transformer forward**

- ``transformer( hidden_states=packed_noisy, cond_hidden_states=packed_cond, …,
  encoder_hidden_states=prompt_embeds, txt_ids=…, img_ids=img_ids, … )``
- Double-stream blocks mix text + image streams; single-stream blocks continue
  the fused sequence. Conditioning tokens attend under masks defined in
  ``src/layers_flux2.py`` (canvas tokens isolated + LoRA applied on Q/K/V/proj
  with a **token mask** so updates target the right spans).

**G — Output slice + loss**

- ``model_pred`` is unpacked to the **main** latent grid only via
  ``unpack_main_latents`` + ``main_latent_ids``.
- **Target** (flow matching): ``target = noise - model_input``.
- **Loss**: weighted MSE ``mean((weighting * (model_pred - target)^2))`` over
  the main latent dimensions (see ``compute_loss_weighting_for_sd3``).

---------------------------------------------------------------------------
3) What LoRA does here (``src/layers_flux2.py``)
---------------------------------------------------------------------------

- For each selected **double-stream** and **single-stream** attention block,
  the default processor is replaced with ``MultiDoubleStreamBlockFlux2LoraProcessor``
  / ``MultiSingleStreamBlockFlux2LoraProcessor``.
- Each LoRA is a **low-rank update** to linear projections: e.g.
  ``query += weight * LoRA_down_up(hidden_states)`` (and K, V, output proj).
- ``LoRALinearLayerFlux2.forward`` applies a **learned mask** on the token
  dimension so the low-rank path only sees the **conditioning (canvas) token
  block**; its span uses ``cond_width``/``cond_height`` (unified RGB size in
  default canvas mode, or ``cond_size``² in random-resolution mode) via
  ``cond_width//8 * cond_height//8 * 16//64`` tokens per LoRA slot.
- Attention **masks** separate: main image + text can attend broadly; canvas
  tokens attend **within** their own canvas patch grid (see ``mask[...]`` in
  ``MultiDoubleStreamBlockFlux2LoraProcessor.__call__``).

So: **LoRA learns residual edits to attention (and related paths) conditioned
on your layout**, without full fine-tune of the base transformer.

---------------------------------------------------------------------------
4) Matching latent grids (default canvas path)
---------------------------------------------------------------------------

By default the canvas dataset **does not** use random ``noise_size`` for the
target anymore. Target and canvas are both resized to the same pixel box
``(unified_train_width, unified_train_height)`` (defaults **1920×1080**, snapped
to /16 so e.g. 1080→**1088** for FLUX.2 VAE + patchify). Default
``canvas_unified_resize`` is **contain** (letterbox), so the bbox composite at
**1920×1080** and a 16:9 full frame stay **paired** — thin black bars, no
center-crop skew.

The **on-disk composite** in ``canvas_bbox_compose`` / ``stage4_canva`` uses
**uniform letterboxing** into 1920×1080 (same scale/offset for background and
bbox corners), so overlays match the source image aspect — not independent
``sx``/``sy`` stretch.

**Main and cond share one latent H×W** (e.g. 68×120 patches for 1088×1920 RGB
→ **8160** tokens per stream). Use ``cover`` if you want scale-to-fill + crop
instead of letterbox for the unified tensor step.

Opt in to the old behaviour with ``--canvas_random_target_resolution`` (random
main side up to ``noise_size``, canvas at ``cond_size``²) — then use
**train_batch_size=1** unless every sample happens to share the same target
resolution.

**Legacy note:** ``cond_size`` (e.g. 512) still controls internal letterboxing
for *random-resolution* mode and LoRA geometry when ``canvas_random_target_resolution``
is set; in unified mode, LoRA ``cond_width`` / ``cond_height`` follow
``unified_train_width`` / ``unified_train_height`` (see ``train.py``).

---------------------------------------------------------------------------
5) Single-batch backprop (smoke test in this file)
---------------------------------------------------------------------------

``test_single_batch_backprop_like_train_py`` builds a **tiny** 1-double + 1-single-block
conditional DiT (random weights), runs **batch size 1** with the same tensor wiring as
``train.py`` (noised main latents, clean cond, ``img_ids`` concat, forward, ``unpack_main_latents``,
flow-matching target ``noise - model_input``, ``compute_loss_weighting_for_sd3``, MSE reduction),
then ``loss.backward()`` and checks for finite gradients. Latent spatial size is **small** (e.g.
2×4 patches) so the test stays fast; shape-only tests above use the full **8160**-token unified grid.

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
from diffusers.training_utils import compute_loss_weighting_for_sd3

from src.flux2_transformer_cond import Flux2Transformer2DModelCond
from src.flux2_train_helpers import prepare_subject_latent_ids, unpack_main_latents


def _tiny_cond_transformer(device: torch.device, dtype: torch.dtype) -> Flux2Transformer2DModelCond:
    """Minimal DiT for smoke tests; ``attention_head_dim=128`` matches default RoPE axes sum."""
    return Flux2Transformer2DModelCond(
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=128,
        joint_attention_dim=256,
        in_channels=128,
        out_channels=128,
        timestep_guidance_channels=256,
        mlp_ratio=2.0,
    ).to(device=device, dtype=dtype)


def _one_batch_training_tensors(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    l_txt: int,
    h_lat: int,
    w_lat: int,
):
    """``model_input`` / noise / pack / ids / timestep — mirrors ``train.py`` (cond path)."""
    c = 128
    model_input = torch.randn(batch_size, c, h_lat, w_lat, device=device, dtype=dtype)
    cond_spatial = torch.randn(batch_size, c, h_lat, w_lat, device=device, dtype=dtype)
    noise = torch.randn_like(model_input)
    sigma_scalar = torch.tensor(0.35, device=device, dtype=dtype)
    sigmas = sigma_scalar.view(1, 1, 1, 1).expand_as(model_input)
    noisy_spatial = (1.0 - sigmas) * model_input + sigmas * noise

    packed_main = Flux2Pipeline._pack_latents(noisy_spatial)
    packed_cond = Flux2Pipeline._pack_latents(cond_spatial)
    enc = torch.randn(batch_size, l_txt, 256, device=device, dtype=dtype)
    txt_ids = torch.zeros(batch_size, l_txt, 4, device=device, dtype=torch.long)
    for bi in range(batch_size):
        for t in range(l_txt):
            txt_ids[bi, t, 0] = t

    ids_main = Flux2Pipeline._prepare_latent_ids(noisy_spatial).to(device)
    ids_sub = prepare_subject_latent_ids(cond_spatial, scale=10).to(device)
    img_ids = torch.cat([ids_main, ids_sub], dim=1)

    timestep = torch.full((batch_size,), 0.42, device=device, dtype=dtype)
    guidance = torch.ones(batch_size, device=device, dtype=dtype)
    target = noise - model_input
    return (
        packed_main,
        packed_cond,
        enc,
        txt_ids,
        img_ids,
        timestep,
        guidance,
        model_input,
        target,
        sigmas,
        noisy_spatial,
    )


# ---------------------------------------------------------------------------
# Example from user terminal (adjust if your run differs)
# ---------------------------------------------------------------------------
EXAMPLE_B = 1
EXAMPLE_C = 128
# Unified 1920×1088 RGB → latent spatial (H/16 × W/16) after VAE+patchify
EXAMPLE_MAIN_H = 68
EXAMPLE_MAIN_W = 120
EXAMPLE_COND_H = 68
EXAMPLE_COND_W = 120

EXAMPLE_MAIN_SEQ = EXAMPLE_MAIN_H * EXAMPLE_MAIN_W  # 8160
EXAMPLE_COND_SEQ = EXAMPLE_COND_H * EXAMPLE_COND_W  # 8160


def test_documented_main_and_cond_latent_shapes():
    """After VAE + patchify, latents are 4D (B, C, H, W); unified canvas path uses same H×W for both."""
    main = torch.zeros(EXAMPLE_B, EXAMPLE_C, EXAMPLE_MAIN_H, EXAMPLE_MAIN_W)
    cond = torch.zeros(EXAMPLE_B, EXAMPLE_C, EXAMPLE_COND_H, EXAMPLE_COND_W)
    assert main.shape == (1, 128, 68, 120)
    assert cond.shape == (1, 128, 68, 120)


def test_pack_latents_flattens_spatial_to_sequence():
    x = torch.randn(EXAMPLE_B, EXAMPLE_C, EXAMPLE_MAIN_H, EXAMPLE_MAIN_W)
    packed = Flux2Pipeline._pack_latents(x)
    assert packed.shape == (EXAMPLE_B, EXAMPLE_MAIN_SEQ, EXAMPLE_C)
    assert packed.shape == (1, 8160, 128)


def test_prepare_latent_ids_shape_matches_main_grid():
    model_input = torch.zeros(EXAMPLE_B, EXAMPLE_C, EXAMPLE_MAIN_H, EXAMPLE_MAIN_W)
    ids = Flux2Pipeline._prepare_latent_ids(model_input)
    assert ids.shape == (EXAMPLE_B, EXAMPLE_MAIN_SEQ, 4)


def test_subject_latent_ids_shape_matches_cond_grid():
    subject = torch.zeros(EXAMPLE_B, EXAMPLE_C, EXAMPLE_COND_H, EXAMPLE_COND_W)
    s_ids = prepare_subject_latent_ids(subject, scale=10)
    assert s_ids.shape[0] == EXAMPLE_B
    assert s_ids.shape[1] == EXAMPLE_COND_SEQ
    assert s_ids.shape[2] == 4


def test_concatenated_img_ids_sequence_length():
    main = torch.zeros(EXAMPLE_B, EXAMPLE_C, EXAMPLE_MAIN_H, EXAMPLE_MAIN_W)
    sub = torch.zeros(EXAMPLE_B, EXAMPLE_C, EXAMPLE_COND_H, EXAMPLE_COND_W)
    main_ids = Flux2Pipeline._prepare_latent_ids(main)
    sub_ids = prepare_subject_latent_ids(sub, scale=10)
    merged = torch.cat([main_ids, sub_ids], dim=1)
    assert merged.shape == (EXAMPLE_B, EXAMPLE_MAIN_SEQ + EXAMPLE_COND_SEQ, 4)
    assert merged.shape[1] == 8160 + 8160


def test_flow_match_noisy_latent_same_shape_as_clean():
    model_input = torch.randn(EXAMPLE_B, EXAMPLE_C, EXAMPLE_MAIN_H, EXAMPLE_MAIN_W)
    noise = torch.randn_like(model_input)
    sigmas = torch.tensor(0.3, dtype=model_input.dtype, device=model_input.device)
    while sigmas.ndim < model_input.ndim:
        sigmas = sigmas.unsqueeze(-1)
    noisy = (1.0 - sigmas) * model_input + sigmas * noise
    assert noisy.shape == model_input.shape


def test_training_target_is_noise_minus_clean():
    model_input = torch.randn(EXAMPLE_B, EXAMPLE_C, EXAMPLE_MAIN_H, EXAMPLE_MAIN_W)
    noise = torch.randn_like(model_input)
    target = noise - model_input
    assert target.shape == model_input.shape


def test_mse_loss_reduces_over_spatial_and_channel_dims():
    """Matches train.py: reshape to (B, -1) then mean over batch after per-sample mean."""
    model_pred = torch.randn(EXAMPLE_B, EXAMPLE_C, EXAMPLE_MAIN_H, EXAMPLE_MAIN_W)
    target = torch.randn_like(model_pred)
    weighting = torch.ones_like(model_pred)
    per_sample = (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
        target.shape[0], -1
    ).mean(1)
    loss = per_sample.mean()
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_unpack_main_latents_roundtrip_geometry():
    """Packed prediction (B, main_seq, C) unwrapped to (B, C, Hm, Wm) using main ids only."""
    packed_pred = torch.randn(EXAMPLE_B, EXAMPLE_MAIN_SEQ, EXAMPLE_C)
    model_input = torch.zeros(EXAMPLE_B, EXAMPLE_C, EXAMPLE_MAIN_H, EXAMPLE_MAIN_W)
    main_latent_ids = Flux2Pipeline._prepare_latent_ids(model_input)
    out = unpack_main_latents(packed_pred, main_latent_ids)
    assert out.shape == (EXAMPLE_B, EXAMPLE_C, EXAMPLE_MAIN_H, EXAMPLE_MAIN_W)


def test_lora_cond_token_count_matches_unified_resolution():
    """Processor cond token count = (W//8)*(H//8)*16//64 — here unified 1920×1088 → 8160."""
    cond_width, cond_height = 1920, 1088
    cond_tokens = cond_width // 8 * cond_height // 8 * 16 // 64
    assert cond_tokens == 8160
    assert cond_tokens == EXAMPLE_COND_SEQ


@pytest.mark.parametrize("device_name", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_single_batch_backprop_like_train_py(device_name: str):
    """
    One optimization step geometry: forward → unpack → weighted MSE vs flow-match target → backward.

    Uses a compact latent grid (2×4) for speed; logic matches ``train.py`` (weighting_scheme=none).
    """
    device = torch.device(device_name)
    dtype = torch.float32

    m = _tiny_cond_transformer(device, dtype)
    m.train()

    B, l_txt, h, w = 1, 3, 2, 4
    (
        packed_main,
        packed_cond,
        enc,
        txt_ids,
        img_ids,
        timestep,
        guidance,
        model_input,
        target,
        sigmas,
        noisy_spatial,
    ) = _one_batch_training_tensors(device=device, dtype=dtype, batch_size=B, l_txt=l_txt, h_lat=h, w_lat=w)

    m.zero_grad(set_to_none=True)
    out = m(
        hidden_states=packed_main,
        cond_hidden_states=packed_cond,
        encoder_hidden_states=enc,
        txt_ids=txt_ids,
        img_ids=img_ids,
        timestep=timestep,
        guidance=guidance,
        return_dict=False,
    )[0]

    main_ids = Flux2Pipeline._prepare_latent_ids(noisy_spatial).to(out.device)
    model_pred = unpack_main_latents(out, main_ids)

    weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
    loss = torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        dim=1,
    ).mean()

    assert torch.isfinite(loss)
    loss.backward()

    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "expected gradients after backward"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.skipif(
    not os.environ.get("FLUX2_TRAINING_TEST_VAE"),
    reason="Set FLUX2_TRAINING_TEST_VAE=1 and FLUX2_MODEL_DIR to run VAE shape check on disk.",
)
def test_encode_flux2_latents_shapes_optional_vae():
    """Optional: real VAE encode for one folder (slow, needs GPU RAM)."""
    import torch
    from diffusers import AutoencoderKLFlux2

    root = os.environ.get("FLUX2_MODEL_DIR", "")
    if not root or not Path(root).is_dir():
        pytest.skip("FLUX2_MODEL_DIR invalid")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    vae = AutoencoderKLFlux2.from_pretrained(root, subfolder="vae")
    vae.eval().to(device, dtype=dtype)

    from src.flux2_train_helpers import encode_flux2_latents

    # Unified canvas path: same H×W for target and conditioning (1088×1920 = snapped 1080p, /16)
    h_px, w_px = 1088, 1920
    main_px = torch.randn(1, 3, h_px, w_px, device=device, dtype=torch.float32)
    z_main = encode_flux2_latents(vae, main_px, dtype)
    assert z_main.ndim == 4

    cond_px = torch.randn(1, 3, h_px, w_px, device=device, dtype=torch.float32)
    z_cond = encode_flux2_latents(vae, cond_px, dtype)
    assert z_cond.shape == z_main.shape, "cond latent grid should match main when RGB H×W match (training default)"
