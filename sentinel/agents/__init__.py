"""
Meta-Cognitive Sub-Agents and Orchestrator for SentinelAI.
Exposes fully learned, self-evolving PyTorch agent modules.
"""
from .red_team import RedTeamAgent
from .curriculum import CurriculumDesigner
from .memory_arch import MemoryArchitect
from .quality_gate import QualityGate
from .evolution import EvolutionAgent
from .orchestrator import SentinelOrchestrator

__all__ = [
    "RedTeamAgent",
    "CurriculumDesigner",
    "MemoryArchitect",
    "QualityGate",
    "EvolutionAgent",
    "SentinelOrchestrator"
]
