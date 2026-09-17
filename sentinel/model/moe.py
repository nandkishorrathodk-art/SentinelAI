import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SwiGLUExpert(nn.Module):
    """
    Single expert Feed-Forward Network using SwiGLU activation.
    
    Architecture: down_proj(silu(gate_proj(x)) * up_proj(x))
    
    Args:
        d_model (int): Hidden dimension size.
        d_ff (int): Intermediate feed-forward dimension.
    """
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
    
    An innovative MoE architecture that features:
    1. A Shared Expert: One expert is always active to capture common patterns.
    2. Learned Neural Router: Routes to top-k remaining experts dynamically.
    3. Expert Vitality Tracking: Tracks expert selection frequency.
    4. Expert Recycling: Periodically reinitializes dead experts to prevent collapse.
    5. Load Balancing Loss: Auxiliary loss to encourage even expert utilization.
    6. Cross-Expert Residual: Combines expert outputs with the residual.
    
    Args:
        d_model (int): Hidden dimension size.
        d_ff (int): Intermediate feed-forward dimension for each expert.
        n_experts (int): Total number of experts (including the shared expert if enabled).
        top_k (int): Number of active routed experts per token.
        shared_expert (bool): Whether to reserve one expert as universally shared.
    """
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        n_experts: int, 
        top_k: int, 
        shared_expert: bool = True
    ):
        super().__init__()
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
        
        # Neural Router (zero puppet behavior, fully learned)
        self.router = nn.Linear(d_model, self.n_routed_experts, bias=False)
        
        # Cross-expert residual projection
        self.cross_expert_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Vitality tracking (Exponential Moving Average)
        self.register_buffer("vitality_score", torch.ones(self.n_routed_experts) / self.n_routed_experts)
        self.vitality_momentum = 0.99
        
    def recycle_dead_experts(self, threshold: float = 0.01):
        """
        Reinitializes weights of dead experts (vitality < threshold) with small random values.
        This prevents expert collapse and ensures all capacity is utilized over time.
        """
        with torch.no_grad():
            dead_mask = self.vitality_score < threshold
            if not dead_mask.any():
                return
                
            for i, is_dead in enumerate(dead_mask):
                if is_dead:
                    expert = self.experts[i]
                    # Reinitialize with small variance
                    nn.init.normal_(expert.gate_proj.weight, mean=0, std=0.02)
                    nn.init.normal_(expert.up_proj.weight, mean=0, std=0.02)
                    nn.init.normal_(expert.down_proj.weight, mean=0, std=0.02)
                    
                    # Reset vitality to uniform distribution level
                    self.vitality_score[i] = 1.0 / self.n_routed_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of SA-MoE.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Output tensor, Load balancing loss scalar)
        """
        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model) # (batch * seq_len, d_model)
        
        # 1. Routing
        router_logits = self.router(flat_x) # (batch * seq_len, n_routed_experts)
        router_probs = F.softmax(router_logits, dim=-1) # (batch * seq_len, n_routed_experts)
        
        # Select Top-K experts
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # 2. Expert Computation
        combined_output = torch.zeros_like(flat_x)
        
        # For efficiency in a simple implementation, we loop over experts.
        # In a highly optimized CUDA implementation, this would be a fused kernel or Triton.
        expert_mask = torch.zeros((flat_x.size(0), self.n_routed_experts), device=x.device)
        expert_mask.scatter_(1, topk_indices, 1.0) # 1 if routed to expert i
        
        for i, expert in enumerate(self.experts):
            idx, = torch.where(expert_mask[:, i] > 0)
            if idx.numel() > 0:
                expert_input = flat_x[idx]
                expert_out = expert(expert_input)
                
                # Weight by routing probability
                _, topk_pos = torch.where(topk_indices[idx] == i)
                routing_weights = topk_probs[idx, topk_pos].unsqueeze(-1)
                
                combined_output[idx] += expert_out * routing_weights

        # 3. Shared Expert Computation
        if self.shared_expert:
            shared_out = self.shared_ffn(flat_x)
            combined_output = combined_output + shared_out
            
        # 4. Cross-Expert Residual
        # Allow implicit knowledge sharing
        output = combined_output + self.cross_expert_proj(combined_output)
        output = output.view(batch_size, seq_len, d_model)
        
        # 5. Load Balancing Loss
        if self.training:
            # f_i: fraction of tokens routed to expert i
            f_i = expert_mask.mean(dim=0)
            # P_i: mean router probability for expert i
            p_i = router_probs.mean(dim=0)
            
            # L_balance = n_experts * sum(f_i * P_i)
            aux_loss = self.n_routed_experts * torch.sum(f_i * p_i)
            
            # 6. Vitality Tracking (EMA of token fraction)
            with torch.no_grad():
                self.vitality_score = self.vitality_momentum * self.vitality_score + (1 - self.vitality_momentum) * f_i
        else:
            aux_loss = torch.tensor(0.0, device=x.device)
            
        return output, aux_loss
