"""
Memory Architect Agent.
Interfaces with 3-tier memory; tags high-value experiences for permanent retention; prevents catastrophic forgetting.
"""
import torch
import torch.nn as nn

class MemoryArchitect(nn.Module):
    def __init__(self, feature_dim: int):
        """
        Initializes the Memory Architect.
        
        Args:
            feature_dim (int): Dimensionality of experience representations.
        """
        super().__init__()
        # Evaluates the 'surprise' or 'value' of an experience to determine if it should be kept
        self.value_network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def evaluate_experience(self, experience_embedding: torch.Tensor) -> torch.Tensor:
        """
        Returns a value score between 0 and 1 for an experience.
        """
        return self.value_network(experience_embedding)

    @torch.no_grad()
    def tag_for_retention(self, experiences: torch.Tensor, threshold: float = 0.8) -> torch.Tensor:
        """
        Tags experiences for permanent retention (L3 memory) if their value exceeds a threshold.
        
        Args:
            experiences (torch.Tensor): Batch of experience embeddings (B, feature_dim)
            threshold (float): Threshold above which to retain.
            
        Returns:
            torch.Tensor: Boolean mask of shape (B,) indicating which experiences to retain.
        """
        values = self.evaluate_experience(experiences).squeeze(-1)
        return values > threshold

    def consolidate_memory(self, short_term_buffer: torch.Tensor) -> torch.Tensor:
        """
        Processes short term buffer (L1/L2) and decides what gets pushed to L3 permanent memory.
        Returns embeddings to be permanently stored.
        """
        retention_mask = self.tag_for_retention(short_term_buffer)
        return short_term_buffer[retention_mask]
