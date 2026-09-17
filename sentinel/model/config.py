from dataclasses import dataclass, field

@dataclass
class SentinelConfig:
    # Tokenizer
    vocab_size: int = 32000
    
    # Architecture
    d_model: int = 256          # embedding dimension
    n_heads: int = 8            # query attention heads
    n_kv_heads: int = 4         # key/value heads (GQA: fewer than n_heads)
    n_layers: int = 8           # transformer blocks
    d_ff: int = 1024            # feedforward hidden dim (4x d_model)
    max_seq_len: int = 512      # context window
    dropout: float = 0.1
    
    # MoE
    n_experts: int = 6          # total experts per layer (including 1 shared)
    top_k_experts: int = 2      # active routed experts per token
    shared_expert: bool = True  # always-on shared expert
    expert_recycle_threshold: float = 0.01  # vitality below this = dead
    moe_aux_loss_weight: float = 0.01       # load balancing loss weight
    
    # RoPE
    rope_theta: float = 10000.0
    
    # Living Weights (Synaptic Plasticity)
    plasticity_enabled: bool = True
    plasticity_lr: float = 1e-5
    plasticity_decay: float = 0.999   # weight decay for plasticity updates
    
    # Vision
    vision_enabled: bool = True
    patch_size: int = 16        # ViT patch size
    image_size: int = 224       # input image resolution
    vision_d_model: int = 256   # vision encoder hidden dim
    
    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 10000
    grad_clip: float = 1.0
    
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
    
    @property
    def n_vision_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2
    
    @property
    def total_params_estimate(self) -> str:
        # Rough estimate
        embed = self.vocab_size * self.d_model
        attn_per_layer = 4 * self.d_model * self.d_model  # Q,K,V,O
        ffn_per_expert = 3 * self.d_model * self.d_ff  # gate, up, down
        ffn_per_layer = ffn_per_expert * self.n_experts
        total = embed + self.n_layers * (attn_per_layer + ffn_per_layer)
        if total > 1e9:
            return f"{total/1e9:.1f}B"
        return f"{total/1e6:.1f}M"
