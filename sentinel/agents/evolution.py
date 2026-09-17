"""
Architecture Evolution Agent.
Tracks MoE expert vitality and router distribution.
Triggers birth of new specialized experts when task complexity rises, and re-allocates underperforming experts.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List

class EvolutionAgent(nn.Module):
    def __init__(self, num_experts: int, vitality_threshold: float = 0.05):
        """
        Initializes the Evolution Agent.
        
        Args:
            num_experts (int): Initial number of experts in the MoE layers.
            vitality_threshold (float): Threshold below which an expert is considered dead/underperforming.
        """
        super().__init__()
        self.num_experts = num_experts
        self.vitality_threshold = vitality_threshold
        # Tracks the exponentially smoothed usage of each expert
        self.register_buffer("expert_vitality", torch.ones(num_experts) / num_experts)
        
    def update_vitality(self, router_probs: torch.Tensor, decay: float = 0.99):
        """
        Updates expert vitality based on router probabilities.
        
        Args:
            router_probs (torch.Tensor): (B, num_experts) probabilities from the MoE router.
            decay (float): Exponential moving average decay factor.
        """
        batch_usage = router_probs.mean(dim=0)
        self.expert_vitality = decay * self.expert_vitality + (1 - decay) * batch_usage

    @torch.no_grad()
    def evaluate_architecture(self) -> Dict[str, Any]:
        """
        Evaluates the current architecture and proposes evolutionary actions based on learned usage statistics.
        
        Returns:
            dict: Proposed actions ('reallocate', 'split') and target expert indices.
        """
        dead_experts = (self.expert_vitality < self.vitality_threshold).nonzero(as_tuple=True)[0]
        # Overloaded expert threshold (e.g. taking more than 3x its fair share of load)
        overloaded_threshold = 3.0 / self.num_experts
        overloaded_experts = (self.expert_vitality > overloaded_threshold).nonzero(as_tuple=True)[0]
        
        actions = []
        for e in dead_experts:
            actions.append({"action": "reallocate", "expert_idx": e.item()})
            
        for e in overloaded_experts:
            actions.append({"action": "split", "expert_idx": e.item()})
            
        return {
            "actions": actions, 
            "vitality": self.expert_vitality.tolist()
        }
