"""
SentinelAI — Core Smoke Tests
==============================
Validates that all core components build, run forward passes, and produce
correct shapes. These tests run on CPU and complete in seconds.
"""

from __future__ import annotations

import pytest
import torch

from sentinel.model.config import SentinelConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> SentinelConfig:
    """Small config for fast CPU testing."""
    return SentinelConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
        n_experts=4,
        top_k_experts=2,
        shared_expert=True,
        dropout=0.0,
        plasticity_enabled=False,
        vision_enabled=False,
    )


@pytest.fixture
def batch():
    """Tiny batch: [batch=2, seq_len=16]."""
    torch.manual_seed(42)
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    return x, y


# ---------------------------------------------------------------------------
# Tokenizer Tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_train_and_encode_decode_roundtrip(self):
        from sentinel.tokenizer.bpe import BytePairTokenizer

        tok = BytePairTokenizer()
        corpus = "the cat sat on the mat. the cat ate the rat."
        tok.train(corpus, vocab_size=280)

        encoded = tok.encode("the cat")
        assert isinstance(encoded, list)
        assert all(isinstance(i, int) for i in encoded)
        assert len(encoded) > 0

        decoded = tok.decode(encoded)
        assert isinstance(decoded, str)
        assert "cat" in decoded

    def test_special_tokens_exist(self):
        from sentinel.tokenizer.bpe import BytePairTokenizer

        tok = BytePairTokenizer()
        assert tok.special_tokens["<PAD>"] == 0
        assert tok.special_tokens["<BOS>"] == 1
        assert tok.special_tokens["<EOS>"] == 2

    def test_vocab_size_property(self):
        from sentinel.tokenizer.bpe import BytePairTokenizer

        tok = BytePairTokenizer()
        corpus = "hello world " * 100
        tok.train(corpus, vocab_size=300)
        assert len(tok) <= 300


# ---------------------------------------------------------------------------
# Embedding Tests
# ---------------------------------------------------------------------------

class TestEmbedding:
    def test_token_embedding_shape(self, config):
        from sentinel.model.embedding import TokenEmbedding

        emb = TokenEmbedding(config.vocab_size, config.d_model)
        x = torch.randint(0, config.vocab_size, (2, 16))
        out = emb(x)
        assert out.shape == (2, 16, config.d_model)

    def test_rope_produces_cos_sin(self, config):
        from sentinel.model.embedding import RotaryPositionalEncoding

        rope = RotaryPositionalEncoding(config.head_dim, config.rope_theta)
        cos, sin = rope(torch.zeros(1), seq_len=16)
        assert cos.shape[-1] == config.head_dim
        assert sin.shape[-1] == config.head_dim


# ---------------------------------------------------------------------------
# Attention Tests
# ---------------------------------------------------------------------------

class TestAttention:
    def test_causal_self_attention_output_shape(self, config):
        from sentinel.model.attention import CausalSelfAttention
        from sentinel.model.embedding import RotaryPositionalEncoding

        attn = CausalSelfAttention(
            config.d_model, config.n_heads, config.n_kv_heads,
            config.max_seq_len, config.dropout,
        )
        rope = RotaryPositionalEncoding(config.head_dim, config.rope_theta)

        x = torch.randn(2, 16, config.d_model)
        cos, sin = rope(x, seq_len=16)
        out = attn(x, cos, sin)
        assert out.shape == (2, 16, config.d_model)


# ---------------------------------------------------------------------------
# MoE Tests
# ---------------------------------------------------------------------------

class TestMoE:
    def test_moe_output_shape_and_aux_loss(self, config):
        from sentinel.model.moe import SentinelMoE

        moe = SentinelMoE(
            config.d_model, config.d_ff, config.n_experts,
            config.top_k_experts, shared_expert=True,
        )
        x = torch.randn(2, 16, config.d_model)
        out, aux_loss = moe(x)
        assert out.shape == (2, 16, config.d_model)
        assert isinstance(aux_loss, (torch.Tensor, float))
        if isinstance(aux_loss, torch.Tensor):
            assert aux_loss.ndim == 0  # scalar

    def test_router_is_learned_not_hardcoded(self, config):
        from sentinel.model.moe import SentinelMoE

        moe = SentinelMoE(
            config.d_model, config.d_ff, config.n_experts,
            config.top_k_experts,
        )
        # Router must have learnable parameters
        router_params = [p for n, p in moe.named_parameters() if "router" in n.lower() or "gate" in n.lower()]
        assert len(router_params) > 0, "Router must have learnable parameters (no puppet/hardcoded routing)"


# ---------------------------------------------------------------------------
# Full Transformer Tests
# ---------------------------------------------------------------------------

class TestTransformer:
    def test_forward_pass_shapes(self, config, batch):
        from sentinel.model.transformer import SentinelTransformer

        model = SentinelTransformer(config)
        x, y = batch

        logits, loss, moe_aux = model(x, targets=y)
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is not None
        assert loss.ndim == 0  # scalar

    def test_generate_produces_tokens(self, config):
        from sentinel.model.transformer import SentinelTransformer

        model = SentinelTransformer(config)
        model.eval()

        prompt = torch.randint(0, config.vocab_size, (1, 4))
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=10)

        assert generated.shape[0] == 1
        assert generated.shape[1] >= 4 + 1  # at least 1 new token

    def test_parameter_count_is_reasonable(self, config):
        from sentinel.model.transformer import SentinelTransformer

        model = SentinelTransformer(config)
        n_params = sum(p.numel() for p in model.parameters())
        # With tiny config, should be roughly 100K-5M params
        assert 10_000 < n_params < 50_000_000

    def test_loss_decreases_over_steps(self, config, batch):
        """Core sanity check: loss must go down with training."""
        from sentinel.model.transformer import SentinelTransformer

        model = SentinelTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x, y = batch

        losses = []
        for _ in range(20):
            logits, loss, moe_aux = model(x, targets=y)
            total_loss = loss + 0.01 * moe_aux
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss at step 20 should be lower than step 1
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Dataset Tests
# ---------------------------------------------------------------------------

class TestDataset:
    def test_text_chunk_dataset(self):
        from sentinel.training.dataset import TextChunkDataset

        tokens = list(range(100))
        ds = TextChunkDataset(tokens, seq_len=10)
        assert len(ds) > 0

        x, y = ds[0]
        assert x.shape == (10,)
        assert y.shape == (10,)
        # Target is input shifted by 1
        assert (x[1:] == y[:-1]).all()


# ---------------------------------------------------------------------------
# Integration Test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_end_to_end_train_generate(self):
        """Full pipeline: build model → train 5 steps → generate text."""
        from sentinel.model.config import SentinelConfig
        from sentinel.model.transformer import SentinelTransformer
        from sentinel.training.dataset import TextChunkDataset

        cfg = SentinelConfig(
            vocab_size=128, d_model=32, n_heads=2, n_kv_heads=1,
            n_layers=1, d_ff=64, max_seq_len=16,
            n_experts=2, top_k_experts=1, shared_expert=True,
            dropout=0.0, plasticity_enabled=False, vision_enabled=False,
        )

        model = SentinelTransformer(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Tiny dataset
        tokens = torch.randint(0, 128, (200,)).tolist()
        ds = TextChunkDataset(tokens, seq_len=16)
        x, y = ds[0]
        x, y = x.unsqueeze(0), y.unsqueeze(0)

        # Train 5 steps
        for _ in range(5):
            logits, loss, aux = model(x, targets=y)
            (loss + 0.01 * aux).backward()
            optimizer.step()
            optimizer.zero_grad()

        # Generate
        model.eval()
        prompt = torch.randint(0, 128, (1, 4))
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=5, temperature=0.8)
        assert out.shape[1] >= 5

