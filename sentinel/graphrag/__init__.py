"""
Sentinel GraphRAG (Knowledge Graph + Retrieval-Augmented Generation) Engine
===========================================================================
Teaches Sentinel how concepts, mathematics, computer systems, and algorithms
fundamentally connect through multi-hop relational graph traversals.
"""

from sentinel.graphrag.graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from sentinel.graphrag.seed_knowledge import populate_universal_knowledge_graph
from sentinel.graphrag.teacher_agents import (
    GraphMathTeacher,
    GraphSystemsTeacher,
    GraphScienceTeacher,
    GraphDialogueTeacher,
    GraphRAGTeacherSwarm,
)

__all__ = [
    "KnowledgeGraph",
    "KnowledgeNode",
    "KnowledgeEdge",
    "populate_universal_knowledge_graph",
    "GraphMathTeacher",
    "GraphSystemsTeacher",
    "GraphScienceTeacher",
    "GraphDialogueTeacher",
    "GraphRAGTeacherSwarm",
]

