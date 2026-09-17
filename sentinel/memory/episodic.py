"""
L2 Episodic Memory.
Persistent storage of recent episodes. Vector/hash indexing for top-k retrieval.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Episode:
    id: str
    content: str
    timestamp: float
    vector_representation: Optional[List[float]] = None

class EpisodicMemory:
    """
    Stores and retrieves recent conversation and action episodes.
    """
    def __init__(self):
        self.episodes: Dict[str, Episode] = {}

    def add_episode(self, episode: Episode) -> None:
        """Stores a new episode."""
        self.episodes[episode.id] = episode

    def retrieve_top_k(self, query_vector: List[float], k: int = 5) -> List[Episode]:
        """
        Retrieves top-k most relevant episodes based on vector similarity.
        Placeholder implementation.
        """
        return list(self.episodes.values())[:k]

    def get_recent(self, limit: int = 10) -> List[Episode]:
        """Returns the most recent episodes."""
        sorted_eps = sorted(self.episodes.values(), key=lambda e: e.timestamp, reverse=True)
        return sorted_eps[:limit]
        
    def remove_episode(self, episode_id: str) -> None:
        """Removes an episode by ID."""
        if episode_id in self.episodes:
            del self.episodes[episode_id]
