"""Model components: MoE, vision encoder, transformer backbone, and SA-MoE."""
from forge.model.sa_config import SentinelConfig

try:
    from forge.model.moe import MoELayer, get_balance_loss
    from forge.model.vision import PatchVisionEncoder
    from forge.model.transformer import ForgeLM
    from forge.model.sa_transformer import SentinelTransformer
    from forge.model.sa_moe import SentinelMoE
    from forge.model.plasticity import SynapticPlasticity
    __all__ = [
        "MoELayer",
        "get_balance_loss",
        "PatchVisionEncoder",
        "ForgeLM",
        "SentinelConfig",
        "SentinelTransformer",
        "SentinelMoE",
        "SynapticPlasticity",
    ]
except OSError:
    __all__ = ["SentinelConfig"]