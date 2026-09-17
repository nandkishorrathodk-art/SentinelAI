# SENTINEL_RULE.md — Core Invariants & Mandate: Sentinel Neural LLM Model

> **NON-NEGOTIABLE CORE MANDATE**:  
> **WE ARE BUILDING A SOVEREIGN NEURAL LLM (LARGE LANGUAGE MODEL) FROM SCRATCH.**  
> Sentinel is a pure neural network architecture. It is NOT an agent wrapper, NOT an external API client, and NOT a scripted puppet framework.

---

## 1. Strict Core Prohibitions (Aisa Karna Sakht Mana Hai)

1. **Zero External API Delegation as the "Brain"**:
   - Never replace or delegate Sentinel's thinking to third-party APIs (OpenAI, Groq, NVIDIA NIM, Anthropic). 
   - Sentinel's intelligence lives strictly in its **OWN weights** (`.pt` / `.bin` / `.xml`). Third-party APIs may only be used for offline dataset generation/distillation if explicitly requested by the user, never as the runtime brain.

2. **Zero Agent Scaffolding & Toolsmith Theater**:
   - No `Toolsmith`, no fake `BeingLifeLoop`, no simulated agent loops, and no hardcoded template strings (e.g. `f"print({cycle_count} * 137)"`).
   - Never write Python code that pretends the AI authored it. Every character the model outputs must be decoded token IDs from `model.generate()`.

3. **Zero Fake Outputs & Mock Demos**:
   - Never print hardcoded `<think>` tags or fake thoughts in demos.
   - Every inference demo must pass actual token IDs into the neural model's forward pass and decode the model's own logits.

4. **Zero Narrow Task Stereotyping**:
   - Sentinel is an **Open-Ended Universal Foundation LLM**: Reasoning, Mathematics, Epistemology, Logic, Algorithms, Language, and Code.
   - Never restrict its identity to a narrow single-purpose label (e.g. "just a cybersecurity tool").

5. **Zero Success by Decree (Empirical Verification Only)**:
   - Claims of model capability are backed strictly by real mathematical and machine receipts:
     - Real cross-entropy loss values.
     - Real token-per-second throughput.
     - Real autoregressive token generation.

---

## 2. The 5 Architectural Pillars of Sentinel LLM

Every component in the repository must belong directly to the neural language model pipeline:

```
Raw Text ──▶ 1. Tokenizer (BPE) ──▶ 2. Embedding + RoPE ──▶ 3. Transformer Blocks (GQA + SA-MoE) ──▶ 4. RMSNorm + LM Head ──▶ 5. Logits / generate()
```

### Pillar 1: Word-Bounded BPE Tokenizer (`tokenizer/bpe.py`)
- Lossless Byte-Pair Encoding implemented from scratch (Zero HuggingFace / tiktoken dependencies).
- Word-boundary regex isolation to prevent cross-word token collapse.
- Exact byte-level UTF-8 fallback for zero `<UNK>` on arbitrary bytes.
- Standard special tokens: `<PAD>=0`, `<BOS>=1`, `<EOS>=2`, `<UNK>=3`, `<MASK>=4`.
- Standard vocab target: 32,000 subword tokens.

### Pillar 2: Embedding & Positional Geometry (`model/embedding.py`)
- Standard lookup table: `nn.Embedding(vocab_size, d_model)`.
- **Rotary Position Embeddings (RoPE)**:
  - Sin/Cos rotation matrices based on $\theta = 10000.0$.
  - Applied directly to Query and Key representations in 2D pairs.
  - Enables length generalization beyond the training context window.

### Pillar 3: Grouped-Query Attention with KV Cache (`model/attention.py`)
- **Grouped-Query Attention (GQA)**:
  - $N_{heads}$ Query heads share $N_{kv\_heads}$ Key/Value heads ($N_{heads} > N_{kv\_heads}$).
  - Reduces KV cache VRAM footprint by up to 50-75% during autoregressive generation.
- Causal triangular attention mask + PyTorch 2.0+ Flash Attention (`F.scaled_dot_product_attention`).
- Dynamic KV Cache (`cache_k`, `cache_v`) for $O(1)$ token-generation complexity.

### Pillar 4: Sentinel Adaptive MoE (SA-MoE) (`model/moe.py`)
- **SwiGLU Activation**: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- **Always-On Shared Expert**: 1 dedicated expert always processes every token to capture universal syntax and grammar.
- **Learned Neural Router**: `nn.Linear(d_model, n_routed)` with Softmax Top-$K$ selection (typically Top-2 of 5 routed experts).
- **Expert Vitality Tracking**: Exponential Moving Average (EMA) of selection frequency per expert.
- **Expert Birth/Death Recycling**: Reinitializes weights of dead experts ($\text{vitality} < 0.01$) to prevent expert collapse.
- **Load Balancing Auxiliary Loss**: Penalizes router imbalance during training:
  $$L_{balance} = N_{routed} \sum_{i=1}^{N_{routed}} f_i \cdot P_i$$

### Pillar 5: Living Weights & Inference Engine (`model/plasticity.py` & `inference/`)
- Pre-norm **RMSNorm** (Root Mean Square Normalization).
- Linear LM Head mapping $d_{model} \to \text{vocab\_size}$ without bias.
- **Synaptic Plasticity (Hebbian Living Weights)**: Test-time micro-updates to attention projection weights based on prediction confidence entropy.
- **Autoregressive Generation**: Nucleus (Top-p) sampling, Top-k filtering, temperature scaling, repetition penalty.

---

## 3. Training & Deployment Philosophy

1. **Dual-Tier Compute Paradigm**:
   - **Cloud Multi-GPU Tier (Kaggle Dual-T4 / A100)**: Used for heavy pretraining, multi-epoch optimization, and loss convergence down to $\le 0.01$.
   - **Local Inference Tier (Acer Swift Neo Intel NPU / CPU)**: Quantized INT8 / OpenVINO IR export targeting the 11 TOPS Intel AI Boost NPU for instant, offline local generation.

2. **Clean Project Layout**:
   ```
   sentinel/
   ├── tokenizer/      # BPE Tokenizer from scratch
   ├── model/          # RoPE, GQA Attention, SwiGLU, SA-MoE, Transformer
   ├── training/       # Loss functions, Dataset loaders, Kaggle/Multi-GPU trainers
   └── inference/      # Autoregressive generation, KV-cache, OpenVINO NPU export
   ```

