import torch
import torch.nn as nn

class MultimodalProjector(nn.Module):
    """
    Projects visual embeddings to the language model's embedding space.
    
    Uses a 2-layer MLP (LLaVA/Qwen style) with GELU activation
    to align the vision encoder's output with the text embedding dimension,
    allowing seamless desktop screenshot and GUI perception tokens inside the Transformer.
    """
    def __init__(self, vision_dim: int, text_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_tokens: Tensor of shape (B, N, vision_dim)
        Returns:
            Tensor of shape (B, N, text_dim) ready to be concatenated with text tokens
        """
        return self.proj(visual_tokens)
