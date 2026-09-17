"""
Sentinel Training & Evolution Sub-Agent Engine
=============================================
Multi-agent synthetic reasoning curriculum, quality gates,
red-team failure probe, and PyTorch training orchestration for Sentinel LLM.
"""

from forge.training.dataset_agents import (
    MathReasoningAgent,
    CodeLogicAgent,
    ScienceConceptsAgent,
    DatasetSynthesizer,
)
from forge.training.quality_gate import QualityGate
from forge.training.red_team_critic import RedTeamCritic
from forge.training.trainer import SentinelTrainer

__all__ = [
    "MathReasoningAgent",
    "CodeLogicAgent",
    "ScienceConceptsAgent",
    "DatasetSynthesizer",
    "QualityGate",
    "RedTeamCritic",
    "SentinelTrainer",
]