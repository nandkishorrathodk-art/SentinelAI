"""
Dream Consolidator.
Scans L2, extracts lessons, updates L3, prunes L2.
"""
from .episodic import EpisodicMemory
from .semantic import SemanticMemory, KnowledgeNode
import time
import uuid

class DreamConsolidator:
    """
    Sleep mode process that transforms episodic memories into semantic knowledge.
    """
    def __init__(self, episodic_memory: EpisodicMemory, semantic_memory: SemanticMemory):
        self.episodic = episodic_memory
        self.semantic = semantic_memory

    def run_consolidation(self) -> None:
        """
        Performs memory consolidation.
        """
        recent_episodes = self.episodic.get_recent(limit=100)
        
        for ep in recent_episodes:
            if len(ep.content) > 10:
                node_id = f"node_{uuid.uuid4().hex[:8]}"
                node = KnowledgeNode(
                    id=node_id,
                    content=f"Consolidated insight from {ep.id}",
                    metadata={"source_episode": ep.id, "timestamp": time.time()},
                    connections=[]
                )
                self.semantic.add_node(node)
                self.episodic.remove_episode(ep.id)
