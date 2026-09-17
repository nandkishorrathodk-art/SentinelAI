import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    Grouped-Query Causal Self-Attention (GQA) module.
    
    This module implements autoregressive causal self-attention with Grouped-Query Attention (GQA),
    where multiple Query heads share a single Key/Value head. It also supports KV-caching for 
    efficient inference generation and Rotary Position Embeddings (RoPE).
    
    Args:
        d_model (int): The hidden dimension size.
        n_heads (int): The number of query heads.
        n_kv_heads (int): The number of key/value heads. Must divide n_heads.
        max_seq_len (int): Maximum sequence length for the KV cache.
        dropout (float): Dropout probability.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        n_kv_heads: int, 
        max_seq_len: int, 
        dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.max_seq_len = max_seq_len
        
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        
        # KV Cache for generation
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None
        self.cache_len = 0
        
    def reset_cache(self):
        """Clears the KV cache between generations."""
        self.cache_k = None
        self.cache_v = None
        self.cache_len = 0
        
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Applies Rotary Position Embeddings (RoPE) to the input tensor."""
        # x: (batch, seq_len, heads, head_dim)
        x_rot = torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
        return (x * cos) + (x_rot * sin)
        
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeats key/value heads for GQA."""
        # x: (batch, seq_len, n_kv_heads, head_dim)
        if n_rep == 1:
            return x
        bs, seqlen, n_kv_heads, head_dim = x.shape
        x = x[:, :, :, None, :].expand(bs, seqlen, n_kv_heads, n_rep, head_dim)
        return x.reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        cos: Optional[torch.Tensor] = None, 
        sin: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for causal self attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model)
            cos (torch.Tensor, optional): Cosine component of RoPE, shape (batch, seq_len, 1, head_dim)
            sin (torch.Tensor, optional): Sine component of RoPE, shape (batch, seq_len, 1, head_dim)
            mask (torch.Tensor, optional): Boolean attention mask
            use_cache (bool): Whether to use/update KV cache (for inference)
            
        Returns:
            torch.Tensor: Output of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE if provided
        if cos is not None and sin is not None:
            xq = self._apply_rope(xq, cos, sin)
            xk = self._apply_rope(xk, cos, sin)
            
        # KV Cache logic
        if use_cache:
            if self.cache_k is None or self.cache_len == 0:
                self.cache_k = xk
                self.cache_v = xv
                self.cache_len = seq_len
            else:
                self.cache_k = torch.cat([self.cache_k, xk], dim=1)
                self.cache_v = torch.cat([self.cache_v, xv], dim=1)
                self.cache_len += seq_len
                xk = self.cache_k
                xv = self.cache_v
                
        # Repeat KV heads for GQA
        xk = self._repeat_kv(xk, self.n_rep)
        xv = self._repeat_kv(xv, self.n_rep)
        
        # Transpose for SDPA: (batch, heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Determine whether to use causal mask intrinsically in SDPA
        is_causal = mask is None and seq_len > 1
        
        # PyTorch 2.0+ Flash Attention / Memory Efficient Attention
        attn_output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )
        
        # Reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.wo(attn_output)
