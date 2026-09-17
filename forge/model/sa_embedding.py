from typing import Tuple
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            seq_len > self.seq_len_cached
            or self.cos_cached is None
            or self.cos_cached.device != x.device
            or self.cos_cached.dtype != x.dtype
        ):
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos().to(dtype=x.dtype)
            self.sin_cached = emb.sin().to(dtype=x.dtype)
            
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def apply_rotary(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

