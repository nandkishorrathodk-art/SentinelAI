"""
Sentinel Knowledge Graph Core
=============================
In-memory and serializable Knowledge Graph structure for multi-hop
semantic reasoning, causal chain extraction, and grounded pedagogical synthesis.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class KnowledgeNode:
    """Represents a core concept, entity, law, or algorithmic construct."""
    id: str
    label: str
    domain: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "domain": self.domain,
            "description": self.description,
            "properties": self.properties,
        }


@dataclass
class KnowledgeEdge:
    """Directed relational link between two concepts."""
    source: str
    target: str
    relation: str  # DEPENDS_ON, DERIVES, IMPLEMENTS, PART_OF, EXPLAINS, CONTRASTS
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "explanation": self.explanation,
        }


class KnowledgeGraph:
    """Core Knowledge Graph engine supporting multi-hop traversal and sub-graph extraction."""

    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self._adj: Dict[str, List[KnowledgeEdge]] = {}
        self._rev_adj: Dict[str, List[KnowledgeEdge]] = {}

    def add_node(self, node: KnowledgeNode) -> None:
        self.nodes[node.id] = node
        if node.id not in self._adj:
            self._adj[node.id] = []
        if node.id not in self._rev_adj:
            self._rev_adj[node.id] = []

    def add_edge(self, edge: KnowledgeEdge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise KeyError(f"Both nodes must exist before adding edge: {edge.source} -> {edge.target}")
        self.edges.append(edge)
        self._adj[edge.source].append(edge)
        self._rev_adj[edge.target].append(edge)

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        return self.nodes.get(node_id)

    def get_outgoing_edges(self, node_id: str) -> List[KnowledgeEdge]:
        return self._adj.get(node_id, [])

    def get_incoming_edges(self, node_id: str) -> List[KnowledgeEdge]:
        return self._rev_adj.get(node_id, [])

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 4
    ) -> List[List[KnowledgeEdge]]:
        """Finds all directed causal or structural paths from source to target within max_depth."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return []

        paths: List[List[KnowledgeEdge]] = []

        def dfs(current: str, target: str, current_path: List[KnowledgeEdge], visited: Set[str], depth: int):
            if depth > max_depth:
                return
            if current == target:
                paths.append(list(current_path))
                return

            for edge in self._adj.get(current, []):
                if edge.target not in visited:
                    visited.add(edge.target)
                    current_path.append(edge)
                    dfs(edge.target, target, current_path, visited, depth + 1)
                    current_path.pop()
                    visited.remove(edge.target)

        dfs(source_id, target_id, [], {source_id}, 0)
        return paths

    def get_subgraph(self, center_node_id: str, depth: int = 1) -> Tuple[List[KnowledgeNode], List[KnowledgeEdge]]:
        """Extracts the ego-network / neighborhood around a node up to specified depth."""
        if center_node_id not in self.nodes:
            return [], []

        visited_nodes: Set[str] = {center_node_id}
        collected_edges: List[KnowledgeEdge] = []
        queue = deque([(center_node_id, 0)])

        while queue:
            curr_id, curr_depth = queue.popleft()
            if curr_depth >= depth:
                continue

            # Check outgoing
            for edge in self._adj.get(curr_id, []):
                collected_edges.append(edge)
                if edge.target not in visited_nodes:
                    visited_nodes.add(edge.target)
                    queue.append((edge.target, curr_depth + 1))

            # Check incoming
            for edge in self._rev_adj.get(curr_id, []):
                collected_edges.append(edge)
                if edge.source not in visited_nodes:
                    visited_nodes.add(edge.source)
                    queue.append((edge.source, curr_depth + 1))

        nodes = [self.nodes[nid] for nid in visited_nodes]
        return nodes, list(set(collected_edges))

    def get_nodes_by_domain(self, domain: str) -> List[KnowledgeNode]:
        return [n for n in self.nodes.values() if n.domain.lower() == domain.lower()]

    def stats(self) -> Dict[str, Any]:
        domains = set(n.domain for n in self.nodes.values())
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "domains": list(domains),
        }

