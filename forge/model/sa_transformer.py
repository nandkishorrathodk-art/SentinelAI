import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.model.sa_config import SentinelConfig
from forge.model.sa_embedding import RotaryPositionalEncoding, TokenEmbedding
from forge.model.sa_attention import CausalSelfAttention
from forge.model.sa_moe import SentinelMoE


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class TransformerBlock(nn.Module):
    def __init__(self, config: SentinelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.moe = SentinelMoE(config)

    def forward(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor,
        sin: torch.Tensor,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        attn_out = self.attention(h, cos, sin, use_cache=use_cache)
        x = x + attn_out
        
        h = self.norm2(x)
        moe_out, aux_loss = self.moe(h)
        x = x + moe_out
        return x, aux_loss


class SentinelTransformer(nn.Module):
    """
    The Sentinel-Prime 46.66M SA-MoE Transformer Engine.
    Combines GQA, RoPE, RMSNorm, SwiGLU, and dynamic expert routing.
    """
    def __init__(self, config: Optional[SentinelConfig] = None):
        super().__init__()
        self.config = config or SentinelConfig()
        
        self.tok_embeddings = TokenEmbedding(self.config.vocab_size, self.config.d_model)
        self.rope = RotaryPositionalEncoding(self.config.head_dim, self.config.rope_theta)
        
        self.layers = nn.ModuleList([
            TransformerBlock(self.config) for _ in range(self.config.n_layers)
        ])
        
        self.norm = RMSNorm(self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        x = self.tok_embeddings(input_ids)
        cos, sin = self.rope(x, seq_len)
        
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, aux_loss = layer(x, cos, sin, use_cache=use_cache)
            total_aux_loss = total_aux_loss + aux_loss
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss, total_aux_loss

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.attention.reset_cache()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_id: int = 2
    ) -> List[int]:
        self.eval()
        self.reset_kv_cache()
        curr_ids = torch.tensor([prompt_ids], dtype=torch.long, device=next(self.parameters()).device)
        
        for _ in range(max_new_tokens):
            if curr_ids.shape[1] > self.config.max_seq_len:
                break
            logits, _, _ = self(curr_ids, use_cache=True)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_val = next_token.item()
            curr_ids = torch.cat((curr_ids, next_token), dim=1)
            if token_val == eos_id:
                break
                
        self.reset_kv_cache()
        return curr_ids[0].tolist()

