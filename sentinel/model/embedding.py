import torch
import torch.nn as nn
from typing import Tuple

class TokenEmbedding(nn.Module):
    """
    Standard lookup table for token embeddings.
    """
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initializes the TokenEmbedding.

        Args:
            vocab_size: The size of the vocabulary.
            d_model: The embedding dimension.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to lookup token embeddings.

        Args:
            x: Input tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            Tensor of embeddings with shape (batch_size, seq_len, d_model).
        """
        return self.embedding(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) implementation.
    Encodes positional information by rotating pairs of embedding dimensions.
    """
    def __init__(self, d_model: int, theta: float = 10000.0):
        """
        Initializes RoPE.

        Args:
            d_model: The embedding dimension. Must be even.
            theta: The base value for the exponential scaling of frequencies.
        """
        super().__init__()
        self.d_model = d_model
        self.theta = theta
        
        # Calculate inverse frequencies for the rotary embeddings
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the sine and cosine matrices for the given sequence length.

        Args:
            x: Input tensor (used to check device and dtype).
            seq_len: The length of the sequence to generate frequencies for.

        Returns:
            A tuple of (cos, sin) tensors of shape (seq_len, d_model).
        """
        if seq_len > self.seq_len_cached or self.cos_cached is None or self.cos_cached.device != x.device or self.cos_cached.dtype != x.dtype:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Duplicate frequencies for both halves
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos().to(dtype=x.dtype)
            self.sin_cached = emb.sin().to(dtype=x.dtype)
            
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor, typically of shape (batch, heads, seq_len, head_dim).
            k: Key tensor, typically of shape (batch, heads, seq_len, head_dim).
            cos: Cosine tensor returned from forward, shape (seq_len, head_dim).
            sin: Sine tensor returned from forward, shape (seq_len, head_dim).

        Returns:
            Tuple of rotated (q, k) tensors.
        """
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
