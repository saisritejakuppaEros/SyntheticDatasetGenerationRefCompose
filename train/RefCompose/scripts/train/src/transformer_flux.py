"""FLUX.2 DiT with optional canvas/subject conditioning. Re-exports the conditional model."""

from .flux2_transformer_cond import Flux2Transformer2DModelCond

# Alias expected by older imports
FluxTransformer2DModel = Flux2Transformer2DModelCond

__all__ = ["Flux2Transformer2DModelCond", "FluxTransformer2DModel"]
