"""
L1 Working Memory (Context).
Volatile, fast token ring-buffer with sliding window and attention sink support.
"""
from typing import List
from dataclasses import dataclass

@dataclass
class WorkingMemoryConfig:
    max_seq_len: int = 4096
    sink_tokens: int = 4

class WorkingMemory:
    """
    Volatile working memory for rapid contextual access.
    Implements a sliding window with attention sinks.
    """
    def __init__(self, config: WorkingMemoryConfig):
        self.config = config
        self.tokens: List[int] = []
        
    def add_tokens(self, tokens: List[int]) -> None:
        """Adds new tokens to the working memory, maintaining the sliding window."""
        self.tokens.extend(tokens)
        if len(self.tokens) > self.config.max_seq_len:
            # Retain sink tokens and the most recent tokens
            sink = self.tokens[:self.config.sink_tokens]
            recent = self.tokens[-(self.config.max_seq_len - self.config.sink_tokens):]
            self.tokens = sink + recent

    def get_context(self) -> List[int]:
        """Returns the current token context."""
        return self.tokens.copy()
        
    def clear(self) -> None:
        """Clears the working memory."""
        self.tokens.clear()
