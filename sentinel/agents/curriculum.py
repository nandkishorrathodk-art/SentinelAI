"""
Curriculum Designer Agent.
Measures training loss curves across domains (coding, hacking, reasoning, vision).
Dynamically reweights and schedules data batches from easy fundamentals to advanced tasks.
"""
import torch
import torch.nn as nn
from typing import Dict

class CurriculumDesigner(nn.Module):
    def __init__(self, num_domains: int):
        """
        Initializes the Curriculum Designer.
        
        Args:
            num_domains (int): Number of domains to track.
        """
        super().__init__()
        self.num_domains = num_domains
        self.register_buffer("domain_weights", torch.ones(num_domains) / num_domains)
        self.register_buffer("loss_history", torch.zeros(num_domains, 100)) # Keep last 100 losses
        self.history_idx = 0
        
        # Meta-learner to predict optimal weights based on loss gradients
        self.weight_predictor = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=0)
        )

    def update_loss(self, domain_idx: int, loss: float):
        """Updates the loss history for a specific domain."""
        self.loss_history[domain_idx, self.history_idx % 100] = loss
        self.history_idx += 1

    @torch.no_grad()
    def reweight_domains(self) -> torch.Tensor:
        """
        Dynamically calculates the optimal sampling weights for data domains.
        Prioritizes domains with high loss plateaus (needing more focus) or steep drops (fast learning).
        """
        # Normalize loss history along the time dimension
        std = self.loss_history.std(dim=1, keepdim=True) + 1e-6
        norm_history = (self.loss_history - self.loss_history.mean(dim=1, keepdim=True)) / std
        
        new_weights = self.weight_predictor(norm_history).squeeze(-1)
        self.domain_weights.copy_(new_weights)
        return self.domain_weights

    def get_batch_schedule(self) -> Dict[int, float]:
        """Returns the current schedule/weights for batch sampling."""
        return {i: self.domain_weights[i].item() for i in range(self.num_domains)}
