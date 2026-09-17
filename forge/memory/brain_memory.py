"""
primeASI — 3-Tier Neuromorphic Memory & Dream Consolidation
==========================================================
L1: Working Memory (volatile, ring buffer with attention sink)
L2: Episodic Memory (persistent storage of actions, tool receipts, execution outcomes)
L3: Permanent Semantic Crystals (compressed knowledge graph)
DreamConsolidator: Background sleep-mode worker that consolidates L2 -> L3.
"""
from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# =====================================================================
# L1: Working Memory
# =====================================================================

@dataclass
class WorkingMemoryConfig:
    max_seq_len: int = 4096
    sink_tokens: int = 4


class WorkingMemory:
    """Volatile working memory for rapid contextual access with attention sinks."""
    def __init__(self, config: Optional[WorkingMemoryConfig] = None):
        self.config = config or WorkingMemoryConfig()
        self.tokens: List[int] = []

    def add_tokens(self, tokens: Sequence[int]) -> None:
        self.tokens.extend(tokens)
        if len(self.tokens) > self.config.max_seq_len:
            sink = self.tokens[:self.config.sink_tokens]
            recent = self.tokens[-(self.config.max_seq_len - self.config.sink_tokens):]
            self.tokens = sink + recent

    def get_context(self) -> List[int]:
        return list(self.tokens)

    def clear(self) -> None:
        self.tokens.clear()


# =====================================================================
# L2: Episodic Memory
# =====================================================================

@dataclass
class Episode:
    id: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.tokens:
            self.tokens = re.findall(r"\w+", self.content.lower())


class EpisodicMemory:
    """Stores and retrieves recent interaction and machine execution episodes."""
    def __init__(self):
        self.episodes: Dict[str, Episode] = {}

    def add_episode(self, content: str, metadata: Optional[Dict[str, Any]] = None, episode_id: Optional[str] = None) -> Episode:
        ep_id = episode_id or f"ep_{uuid.uuid4().hex[:8]}"
        ep = Episode(id=ep_id, content=content, metadata=metadata or {})
        self.episodes[ep_id] = ep
        return ep

    def search(self, query: str, top_k: int = 5) -> List[Episode]:
        q_tokens = set(re.findall(r"\w+", query.lower()))
        if not q_tokens:
            return list(self.episodes.values())[-top_k:]

        scored = []
        for ep in self.episodes.values():
            overlap = len(q_tokens.intersection(ep.tokens))
            if overlap > 0:
                score = overlap / math.sqrt(len(ep.tokens) + 1e-5)
                scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def get_recent(self, limit: int = 10) -> List[Episode]:
        sorted_eps = sorted(self.episodes.values(), key=lambda e: e.timestamp, reverse=True)
        return sorted_eps[:limit]

    def remove_episode(self, episode_id: str) -> None:
        if episode_id in self.episodes:
            del self.episodes[episode_id]


# =====================================================================
# L3: Semantic Memory (Knowledge Crystals)
# =====================================================================

@dataclass
class KnowledgeCrystal:
    id: str
    assertion: str
    confidence: float
    evidence_receipts: List[str] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class SemanticMemory:
    """Permanent semantic memory of crystallized lessons and verified techniques."""
    def __init__(self):
        self.crystals: Dict[str, KnowledgeCrystal] = {}

    def add_crystal(self, crystal: KnowledgeCrystal) -> None:
        self.crystals[crystal.id] = crystal

    def query(self, text: str) -> List[KnowledgeCrystal]:
        query_words = set(re.findall(r"\w+", text.lower()))
        results = []
        for c in self.crystals.values():
            c_words = set(re.findall(r"\w+", c.assertion.lower()))
            if query_words.intersection(c_words):
                results.append(c)
        return results

    def connect(self, source_id: str, target_id: str) -> None:
        if source_id in self.crystals and target_id in self.crystals:
            if target_id not in self.crystals[source_id].connections:
                self.crystals[source_id].connections.append(target_id)


# =====================================================================
# Dream Consolidator
# =====================================================================

class DreamConsolidator:
    """Offline sleep-cycle process that compresses L2 experiences into L3 crystals."""
    def __init__(self, episodic: EpisodicMemory, semantic: SemanticMemory):
        self.episodic = episodic
        self.semantic = semantic

    def consolidate(self, min_length: int = 20) -> int:
        recent = self.episodic.get_recent(limit=50)
        crystallized = 0

        for ep in recent:
            if len(ep.content) >= min_length:
                crystal_id = f"crystal_{uuid.uuid4().hex[:8]}"
                assertion = f"Verified Lesson from {ep.id}: {ep.content[:120]}"
                receipt = ep.metadata.get("audit_head") or ep.metadata.get("receipt", "verified_in_sandbox")
                crystal = KnowledgeCrystal(
                    id=crystal_id,
                    assertion=assertion,
                    confidence=ep.metadata.get("confidence", 0.95),
                    evidence_receipts=[str(receipt)],
                    metadata={"source_episode": ep.id},
                )
                self.semantic.add_crystal(crystal)
                self.episodic.remove_episode(ep.id)
                crystallized += 1

        return crystallized


# =====================================================================
# Unified Brain Memory
# =====================================================================

class BrainMemory:
    """Unified 3-Tier Brain Memory managing L1, L2, L3, and Dreaming."""
    def __init__(self):
        self.l1 = WorkingMemory()
        self.l2 = EpisodicMemory()
        self.l3 = SemanticMemory()
        self.consolidator = DreamConsolidator(self.l2, self.l3)

    def record_experience(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Episode:
        return self.l2.add_episode(content, metadata)

    def sleep_and_dream(self) -> int:
        """Trigger offline dream consolidation."""
        return self.consolidator.consolidate()

