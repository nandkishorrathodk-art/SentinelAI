"""
Adversarial Quality Gate.
Consensus voting across teacher outputs. Rejects low-confidence or hallucinated data before feeding to the trainer.
"""
import torch
import torch.nn as nn
from typing import Tuple

class QualityGate(nn.Module):
    def __init__(self, hidden_dim: int, num_teachers: int):
        """
        Initializes the Adversarial Quality Gate.
        
        Args:
            hidden_dim (int): Dimensionality of the representations.
            num_teachers (int): Number of teacher models providing labels (e.g., GPT-4o, Claude).
        """
        super().__init__()
        self.num_teachers = num_teachers
        
        # Learned consensus module to weight teachers dynamically based on feature context
        self.consensus_network = nn.Sequential(
            nn.Linear(hidden_dim * num_teachers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_teachers),
            nn.Softmax(dim=-1)
        )
        
        # Quality thresholding score predictor
        self.quality_scorer = nn.Linear(hidden_dim, 1)

    def forward(self, teacher_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates a batch of teacher outputs to establish a consensus representation and a quality score.
        
        Args:
            teacher_embeddings (torch.Tensor): Embeddings from teachers (B, num_teachers, hidden_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Consensus representation (B, hidden_dim)
                - Quality scores (B,)
        """
        B, T, D = teacher_embeddings.shape
        flat_embeds = teacher_embeddings.view(B, T * D)
        
        # Dynamic teacher weights (B, num_teachers, 1)
        weights = self.consensus_network(flat_embeds).unsqueeze(-1) 
        
        # Weighted sum for consensus: (B, hidden_dim)
        consensus_embed = (teacher_embeddings * weights).sum(dim=1)
        
        # Predict quality of consensus: (B,)
        quality_score = torch.sigmoid(self.quality_scorer(consensus_embed)).squeeze(-1)
        
        return consensus_embed, quality_score

    @torch.no_grad()
    def filter_batch(self, teacher_embeddings: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
        """
        Rejects data below a quality threshold. Returns a boolean mask of valid samples.
        """
        _, scores = self(teacher_embeddings)
        return scores > threshold
