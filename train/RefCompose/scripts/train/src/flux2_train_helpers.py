from typing import List, Optional, Tuple

import torch

from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline, retrieve_latents


def encode_flux2_latents(
    vae,
    pixel_values: torch.Tensor,
    weight_dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Image batch (B,3,H,W) normalized [-1,1] -> patchified latents (B, C', H', W')."""
    # Match VAE weight dtype (bf16 under mixed precision); avoids conv bias/input mismatch without autocast.
    p0 = next(vae.parameters())
    pixel_values = pixel_values.to(device=p0.device, dtype=p0.dtype)
    enc = vae.encode(pixel_values)
    latents = retrieve_latents(enc, generator=generator, sample_mode="sample")
    latents = latents.float()
    latents = Flux2Pipeline._patchify_latents(latents)
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    latents = (latents - latents_bn_mean) / latents_bn_std
    return latents.to(weight_dtype)


def prepare_subject_latent_ids(latents_bchw: torch.Tensor, scale: int = 10) -> torch.Tensor:
    """Per-batch 4D ids for one subject/canvas latent map (B, C, H, W)."""
    b = latents_bchw.shape[0]
    out = []
    for i in range(b):
        ids = Flux2Pipeline._prepare_image_ids([latents_bchw[i : i + 1]], scale=scale)
        out.append(ids)
    return torch.cat(out, dim=0)


def unpack_main_latents(packed_pred: torch.Tensor, main_latent_ids: torch.Tensor) -> torch.Tensor:
    """Unpack packed transformer output (B, seq, C) using main-image position ids only."""
    main_latent_ids = main_latent_ids.to(device=packed_pred.device)
    parts = []
    for b in range(packed_pred.shape[0]):
        u = Flux2Pipeline._unpack_latents_with_ids(packed_pred[b : b + 1], main_latent_ids[b : b + 1])
        parts.append(u)
    return torch.cat(parts, dim=0)
