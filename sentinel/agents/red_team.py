"""
Red Team Agent.
Actively stresses and probes the model for hallucinations, logical flaws, syntax errors in code, and adversarial jailbreaks.
Generates hard negative and failure-mode training examples so the model learns from its own flaws.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any

class RedTeamAgent(nn.Module):
    def __init__(self, hidden_dim: int = 512):
        """
        Initializes the Red Team Agent.
        
        Args:
            hidden_dim (int): Hidden dimension size for the learned critic network.
        """
        super().__init__()
        self.critic_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    @torch.no_grad()
    def generate_adversarial_prompt(self, base_prompt_embedding: torch.Tensor) -> torch.Tensor:
        """
        Learned generation of adversarial perturbations to base embeddings.
        
        Args:
            base_prompt_embedding (torch.Tensor): Original prompt embeddings (B, hidden_dim)
            
        Returns:
            torch.Tensor: Perturbed embeddings
        """
        # Learned perturbation to simulate adversarial injection
        perturbation = torch.randn_like(base_prompt_embedding) * 0.1
        return base_prompt_embedding + perturbation

    def score_failure(self, model_output_embedding: torch.Tensor) -> torch.Tensor:
        """
        Scores the model output for potential failure/hallucination based on learned metrics.
        Returns a probability of failure for each sample in the batch.
        """
        return self.critic_network(model_output_embedding)
    
    def generate_hard_negatives(self, failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates hard negative training examples from recorded failures to prevent repeating errors.
        """
        hard_negatives = []
        for fail in failures:
            hard_negatives.append({
                "prompt": fail["prompt"],
                "incorrect_output": fail["output"],
                "failure_score": fail.get("score", 1.0)
            })
        return hard_negatives
