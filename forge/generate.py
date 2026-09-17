"""
Sentinel Sovereign Neural LLM — Inference & Text Generation Runner
==================================================================
Loads trained Sentinel checkpoint weights (.pt) and generates tokens
autoregressively using pure neural matrix multiplication and GQA KV-cache.

ZERO agent scaffolding, ZERO puppet heuristics, ZERO external APIs.
Every token produced comes purely from the model's own neural weights.
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Byte-Pair Encoding Tokenizer (BPE)
# ==============================================================================

class BytePairTokenizer:
    """Pure Python Byte-Pair Encoding tokenizer with word-boundary isolation."""

    def __init__(self, vocab: Optional[Dict[int, bytes]] = None, merges: Optional[Dict[Tuple[int, int], int]] = None):
        self.special_tokens = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3, "<MASK>": 4}
        self.inv_special_tokens = {v: k for k, v in self.special_tokens.items()}
        if vocab is not None and merges is not None:
            self.vocab = vocab
            self.merges = merges
            self._next_id = max(self.vocab.keys()) + 1 if self.vocab else 261
        else:
            self.vocab = {i + 5: bytes([i]) for i in range(256)}
            self.merges = {}
            self._next_id = 261

    @property
    def vocab_size(self) -> int:
        return self._next_id

    def __len__(self) -> int:
        return self._next_id

    def encode(self, text: str) -> List[int]:
        words = re.findall(r"\s+|\w+|[^\w\s]", text)
        out = []
        for w in words:
            ids = [b + 5 for b in w.encode("utf-8")]
            while len(ids) >= 2:
                pairs = list(zip(ids, ids[1:]))
                pair_to_merge = None
                lowest_id = float("inf")
                for p in pairs:
                    if p in self.merges and self.merges[p] < lowest_id:
                        lowest_id = self.merges[p]
                        pair_to_merge = p
                if pair_to_merge is None:
                    break
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair_to_merge:
                        new_ids.append(self.merges[pair_to_merge])
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids
            out.extend(ids)
        return out

    def decode(self, ids: List[int]) -> str:
        b = bytearray()
        for i in ids:
            if i in self.inv_special_tokens:
                pass
            elif i in self.vocab:
                b.extend(self.vocab[i])
        return b.decode("utf-8", errors="replace")


# ==============================================================================
# 2. Sentinel Neural Architecture (SA-MoE + GQA + RoPE)
# ==============================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return (x * cos) + (rotated * sin)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class CausalGQAAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, n_kv_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.num_queries_per_kv = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos = cos[:T, :].unsqueeze(0).unsqueeze(0)
        sin = sin[:T, :].unsqueeze(0).unsqueeze(0)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn)


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SentinelAdaptiveMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 6, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.shared_expert = SwiGLUExpert(d_model, d_ff)
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, d_ff) for _ in range(n_experts)])
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.cross_residual = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        flat_x = x.view(-1, C)
        router_logits = self.router(flat_x)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        out = self.shared_expert(flat_x)
        for e_idx, expert in enumerate(self.experts):
            expert_out = expert(flat_x)
            for i in range(self.top_k):
                mask = (topk_indices[:, i:i+1] == e_idx).to(flat_x.dtype)
                out = out + mask * topk_weights[:, i:i+1] * expert_out

        density = router_probs.mean(dim=0)
        aux_loss = self.n_experts * torch.sum(density * router_probs.mean(dim=0))
        final_out = out.view(B, T, C) + self.cross_residual(x)
        return final_out, aux_loss


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, n_experts: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalGQAAttention(d_model, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.moe = SentinelAdaptiveMoE(d_model, d_ff, n_experts=n_experts, top_k=2)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attn(self.norm1(x), cos, sin)
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, aux_loss


class SentinelTransformer(nn.Module):
    """
    Sentinel SA-MoE Neural Transformer.
    Direct sovereign neural weights.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        d_ff: int = 1024,
        n_experts: int = 6
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(d_model // n_heads)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ff, n_experts) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = x.shape
        h = self.tok_embed(x)
        cos, sin = self.rope(T, x.device)
        total_aux = 0.0
        for layer in self.layers:
            h, aux = layer(h, cos, sin)
            total_aux = total_aux + aux
        h = self.final_norm(h)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss, total_aux

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 40
    ) -> torch.Tensor:
        """Autoregressively sample tokens from the model's logits distribution."""
        self.eval()
        for _ in range(max_new_tokens):
            x_cond = prompt_ids[:, -512:]
            logits, _, _ = self(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            prompt_ids = torch.cat((prompt_ids, next_tok), dim=1)
            if next_tok.item() == 2:  # <EOS>
                break
        return prompt_ids


# ==============================================================================
# 3. Model Loader & Generator Runner
# ==============================================================================

def load_sentinel_model(checkpoint_path: str, device: str = "cpu") -> Tuple[SentinelTransformer, BytePairTokenizer]:
    """
    Loads trained Sentinel checkpoint (.pt) and reconstructs the tokenizer and model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"[*] Loading Sentinel weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct Tokenizer from learned vocab and merges in checkpoint
    vocab = checkpoint.get("vocab")
    merges = checkpoint.get("merges")
    tokenizer = BytePairTokenizer(vocab=vocab, merges=merges)
    vocab_size = len(tokenizer)
    print(f"[*] Tokenizer loaded: {vocab_size} vocabulary tokens.")

    # Reconstruct Model architecture
    d_model = checkpoint.get("d_model", 256)
    n_layers = checkpoint.get("n_layers", 8)
    n_heads = checkpoint.get("n_heads", 8)
    n_experts = checkpoint.get("n_experts", 6)

    model = SentinelTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=4,
        d_ff=1024,
        n_experts=n_experts
    )

    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Sentinel Neural Model instantiated: {total_params:,} parameters ({total_params/1e6:.2f}M)")
    print(f"[+] Device: {device}")
    return model, tokenizer


def generate_text(
    model: SentinelTransformer,
    tokenizer: BytePairTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    device: str = "cpu"
) -> str:
    """Encodes prompt, generates tokens via neural weights, and decodes result."""
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    prompt_len = input_ids.shape[1]

    t0 = time.time()
    gen_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
    t1 = time.time()

    full_output = tokenizer.decode(gen_ids[0].tolist())
    generated_tokens = gen_ids.shape[1] - prompt_len
    tokens_per_sec = generated_tokens / max(t1 - t0, 1e-4)

    return full_output, tokens_per_sec


def main():
    parser = argparse.ArgumentParser(description="Sentinel Sovereign Neural LLM Generator")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_final.pt",
        help="Path to trained sentinel_final.pt checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Question: What defines Artificial Superintelligence (ASI)?\n<think>",
        help="Input prompt for generation"
    )
    parser.add_argument("--max_tokens", type=int, default=85, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=30, help="Top-K sampling")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    model, tokenizer = load_sentinel_model(args.checkpoint, device=args.device)

    print("\n" + "=" * 60)
    print("SENTINEL SOVEREIGN NEURAL GENERATION")
    print("=" * 60)
    print(f"> PROMPT:\n{args.prompt}\n")

    output_text, speed = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )

    print(f"> GENERATION OUTPUT:\n{output_text}")
    print(f"\n[Generation speed: {speed:.1f} tokens/second]")
    print("=" * 60)


if __name__ == "__main__":
    main()

