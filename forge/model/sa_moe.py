from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.model.sa_config import SentinelConfig


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SentinelMoE(nn.Module):
    """
    Sentinel Adaptive Mixture-of-Experts (SA-MoE).
    1. Shared Expert: Always active for universal features.
    2. Learned Neural Router: Routes to top-k dynamic experts.
    3. Expert Vitality Tracking & Birth/Death Recycling.
    4. Load Balancing Loss + Cross-Expert Residual.
    """
    def __init__(self, config: SentinelConfig):
        super().__init__()
        d_model = config.d_model
        d_ff = config.d_ff
        n_experts = config.n_experts
        top_k = config.top_k_experts
        shared_expert = config.shared_expert

        self.d_model = d_model
        self.n_experts = n_experts
        self.n_routed_experts = n_experts - 1 if shared_expert else n_experts
        self.top_k = min(top_k, self.n_routed_experts)
        self.shared_expert = shared_expert
        
        if self.shared_expert:
            self.shared_ffn = SwiGLUExpert(d_model, d_ff)
            
        self.experts = nn.ModuleList([
            SwiGLUExpert(d_model, d_ff) for _ in range(self.n_routed_experts)
        ])
        
        self.router = nn.Linear(d_model, self.n_routed_experts, bias=False)
        self.cross_expert_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.register_buffer("vitality_score", torch.ones(self.n_routed_experts) / self.n_routed_experts)
        self.vitality_momentum = 0.99
        
    def recycle_dead_experts(self, threshold: float = 0.01):
        with torch.no_grad():
            dead_mask = self.vitality_score < threshold
            if not dead_mask.any():
                return
                
            for i, is_dead in enumerate(dead_mask):
                if is_dead:
                    expert = self.experts[i]
                    nn.init.normal_(expert.gate_proj.weight, mean=0, std=0.02)
                    nn.init.normal_(expert.up_proj.weight, mean=0, std=0.02)
                    nn.init.normal_(expert.down_proj.weight, mean=0, std=0.02)
                    self.vitality_score[i] = 1.0 / self.n_routed_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)
        
        router_logits = self.router(flat_x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        combined_output = torch.zeros_like(flat_x)
        expert_mask = torch.zeros((flat_x.size(0), self.n_routed_experts), device=x.device)
        expert_mask.scatter_(1, topk_indices, 1.0)
        
        for i, expert in enumerate(self.experts):
            idx, = torch.where(expert_mask[:, i] > 0)
            if idx.numel() > 0:
                expert_input = flat_x[idx]
                expert_out = expert(expert_input)
                _, topk_pos = torch.where(topk_indices[idx] == i)
                routing_weights = topk_probs[idx, topk_pos].unsqueeze(-1)
                combined_output[idx] += expert_out * routing_weights

        if self.shared_expert:
            shared_out = self.shared_ffn(flat_x)
            combined_output = combined_output + shared_out
            
        output = combined_output + self.cross_expert_proj(combined_output)
        output = output.view(batch_size, seq_len, d_model)
        
        if self.training:
            f_i = expert_mask.mean(dim=0)
            p_i = router_probs.mean(dim=0)
            aux_loss = self.n_routed_experts * torch.sum(f_i * p_i)
            with torch.no_grad():
                self.vitality_score = self.vitality_momentum * self.vitality_score + (1 - self.vitality_momentum) * f_i
        else:
            aux_loss = torch.tensor(0.0, device=x.device)
            
        return output, aux_loss

