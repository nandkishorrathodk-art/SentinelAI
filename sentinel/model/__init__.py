"""
SentinelAI Model Package
========================
Provides:
  - SentinelConfig: dataclass configuration
  - SentinelTransformer: full model assembly
  - SentinelMoE: adaptive mixture of experts
  - CausalSelfAttention: grouped-query attention
  - TokenEmbedding, RotaryPositionalEncoding: embedding & RoPE
  - SynapticPlasticity: living weights engine
"""

from sentinel.model.config import SentinelConfig

__all__ = ["SentinelConfig"]

# Lazy-load torch-dependent components
def get_model_components():
    from sentinel.model.embedding import TokenEmbedding, RotaryPositionalEncoding
    from sentinel.model.attention import CausalSelfAttention
    from sentinel.model.moe import SentinelMoE, SwiGLUExpert
    from sentinel.model.transformer import SentinelTransformer, TransformerBlock, RMSNorm
    from sentinel.model.plasticity import SynapticPlasticity
    return {
        "TokenEmbedding": TokenEmbedding,
        "RotaryPositionalEncoding": RotaryPositionalEncoding,
        "CausalSelfAttention": CausalSelfAttention,
        "SentinelMoE": SentinelMoE,
        "SwiGLUExpert": SwiGLUExpert,
        "SentinelTransformer": SentinelTransformer,
        "TransformerBlock": TransformerBlock,
        "RMSNorm": RMSNorm,
        "SynapticPlasticity": SynapticPlasticity,
    }
