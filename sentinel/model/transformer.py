import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

from sentinel.model.config import SentinelConfig
from sentinel.model.embedding import TokenEmbedding, RotaryPositionalEncoding
from sentinel.model.attention import CausalSelfAttention
from sentinel.model.moe import SentinelMoE


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Faster and more stable alternative to LayerNorm that scales activations by their RMS.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS along the feature dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class TransformerBlock(nn.Module):
    """
    A single Transformer layer combining Attention and MoE with RMSNorm and residuals.
    Uses pre-norm formulation for stable training.
    """
    def __init__(self, config: SentinelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.moe = SentinelMoE(config)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            freqs_cis: RoPE frequencies
            use_cache: Whether to use KV cache for generation
            
        Returns:
            Tuple of (output_tensor, moe_aux_loss)
        """
        # Self-Attention path
        h = self.norm1(x)
        attn_out = self.attention(h, freqs_cis, use_cache=use_cache)
        x = x + attn_out
        
        # MoE path
        h = self.norm2(x)
        moe_out, aux_loss = self.moe(h)
        x = x + moe_out
        
        return x, aux_loss


class SentinelTransformer(nn.Module):
    """
    The main SentinelAI Transformer Model.
    Autoregressive language model with MoE and Living Weights.
    """
    def __init__(self, config: SentinelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.tok_embeddings = TokenEmbedding(config)
        self.rope = RotaryPositionalEncoding(config)
        
        # Transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        # Final layer norm & LM Head
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier uniform for linear layers
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def reset_kv_cache(self):
        """Clears the KV cache in all attention layers."""
        for layer in self.layers:
            layer.attention.reset_cache()
            
    def forward(
        self, 
        input_ids: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass for the Sentinel Transformer.
        
        Args:
            input_ids: (batch, seq_len)
            targets: (batch, seq_len) for computing loss
            
        Returns:
            Tuple of (logits, loss, total_moe_aux_loss)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.tok_embeddings(input_ids)
        
        # RoPE setup
        freqs_cis = self.rope(seq_len, device=x.device)
        
        total_aux_loss = 0.0
        
        for layer in self.layers:
            x, aux_loss = layer(x, freqs_cis, use_cache=False)
            total_aux_loss += aux_loss
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss, total_aux_loss

    @torch.no_grad()
    def generate(
        self, 
        prompt_ids: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None, 
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV-cache.
        
        Args:
            prompt_ids: (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Softmax temperature
            top_k: Top-K sampling cutoff
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated token ids (batch, seq_len + max_new_tokens)
        """
        self.eval()
        self.reset_kv_cache()
        device = prompt_ids.device
        
        # Initial forward pass to fill cache
        x = self.tok_embeddings(prompt_ids)
        seq_len = prompt_ids.size(1)
        freqs_cis = self.rope(seq_len + max_new_tokens, device=device)
        
        curr_freqs = freqs_cis[:seq_len]
        for layer in self.layers:
            x, _ = layer(x, curr_freqs, use_cache=True)
            
        x = self.norm(x)
        logits = self.lm_head(x[:, -1, :]) # Only care about last token
        
        out_ids = [prompt_ids]
        curr_id = self._sample(logits, temperature, top_k, top_p)
        out_ids.append(curr_id)
        
        # Autoregressive generation
        for i in range(1, max_new_tokens):
            x = self.tok_embeddings(curr_id)
            curr_freq = freqs_cis[seq_len + i - 1 : seq_len + i]
            
            for layer in self.layers:
                x, _ = layer(x, curr_freq, use_cache=True)
                
            x = self.norm(x)
            logits = self.lm_head(x.squeeze(1))
            
            curr_id = self._sample(logits, temperature, top_k, top_p)
            out_ids.append(curr_id)
            
        return torch.cat(out_ids, dim=1)

    def _sample(self, logits: torch.Tensor, temperature: float, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
        """Helper for top-k and top-p sampling."""
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
            
        logits = logits / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            for b in range(logits.size(0)):
                indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                logits[b, indices_to_remove] = -float('Inf')
                
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
