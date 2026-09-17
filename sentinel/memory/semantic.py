"""
L3 Permanent Semantic Memory.
Compressed knowledge crystals, fact assertions, verified exploits, CVE metadata.
"""
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class KnowledgeNode:
    id: str
    content: str
    metadata: Dict[str, Any]
    connections: List[str]

class SemanticMemory:
    """
    Permanent semantic knowledge graph.
    """
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}

    def add_node(self, node: KnowledgeNode) -> None:
        """Adds a knowledge node to the semantic graph."""
        self.nodes[node.id] = node

    def query_concept(self, query: str) -> List[KnowledgeNode]:
        """
        Queries the semantic graph for concepts.
        """
        results = []
        for node in self.nodes.values():
            if query.lower() in node.content.lower():
                results.append(node)
        return results

    def add_connection(self, source_id: str, target_id: str) -> None:
        """Creates a directed connection between two knowledge nodes."""
        if source_id in self.nodes and target_id in self.nodes:
            if target_id not in self.nodes[source_id].connections:
                self.nodes[source_id].connections.append(target_id)
