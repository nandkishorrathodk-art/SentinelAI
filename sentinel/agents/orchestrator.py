"""
Sentinel Orchestrator.
Coordinates all sub-agents concurrently. Exposes clean APIs for running harvesting batches, 
quality filtering, curriculum scheduling, and continuous self-evolution.
"""
import torch
import torch.nn as nn
from typing import Dict, Any

from .red_team import RedTeamAgent
from .curriculum import CurriculumDesigner
from .memory_arch import MemoryArchitect
from .quality_gate import QualityGate
from .evolution import EvolutionAgent

class SentinelOrchestrator(nn.Module):
    def __init__(self, hidden_dim: int, num_domains: int, num_teachers: int, num_experts: int):
        """
        Initializes the Sentinel Orchestrator and all Meta-Cognitive Sub-Agents.
        
        Args:
            hidden_dim (int): Model hidden dimension.
            num_domains (int): Number of curriculum domains.
            num_teachers (int): Number of teacher models in the ensemble.
            num_experts (int): Number of MoE experts.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Instantiate pure PyTorch sub-agents
        self.red_team = RedTeamAgent(hidden_dim=hidden_dim)
        self.curriculum = CurriculumDesigner(num_domains=num_domains)
        self.memory_arch = MemoryArchitect(feature_dim=hidden_dim)
        self.quality_gate = QualityGate(hidden_dim=hidden_dim, num_teachers=num_teachers)
        self.evolution = EvolutionAgent(num_experts=num_experts)

    def process_training_step(
        self, 
        raw_teacher_embeddings: torch.Tensor, 
        model_output_embeddings: torch.Tensor,
        router_probs: torch.Tensor,
        domain_idx: int,
        loss: float
    ) -> Dict[str, Any]:
        """
        Orchestrates a full meta-cognitive training step across all sub-agents.
        This provides a unified API for the trainer to invoke continuous improvement loops.
        """
        stats = {}
        
        # 1. Curriculum Update & Scheduling
        self.curriculum.update_loss(domain_idx, loss)
        stats["curriculum_schedule"] = self.curriculum.get_batch_schedule()
        
        # 2. Quality Filtering (Reject hallucinations from teachers)
        valid_mask = self.quality_gate.filter_batch(raw_teacher_embeddings)
        stats["rejection_rate"] = 1.0 - (valid_mask.sum().float() / valid_mask.numel()).item()
        
        # 3. Red Team Evaluation (Stress test model outputs)
        failure_probs = self.red_team.score_failure(model_output_embeddings)
        stats["mean_failure_prob"] = failure_probs.mean().item()
        
        # 4. Memory Architecture (Retain high-value experiences)
        retention_mask = self.memory_arch.tag_for_retention(model_output_embeddings)
        stats["retention_rate"] = (retention_mask.sum().float() / (retention_mask.numel() + 1e-6)).item()
        
        # 5. Architecture Evolution (Track MoE vitality)
        self.evolution.update_vitality(router_probs)
        stats["evolution_actions"] = self.evolution.evaluate_architecture()["actions"]
        
        return stats

    def trigger_evolution(self) -> Dict[str, Any]:
        """
        API to retrieve and apply architecture evolution proposed by the Evolution Agent.
        """
        return self.evolution.evaluate_architecture()
