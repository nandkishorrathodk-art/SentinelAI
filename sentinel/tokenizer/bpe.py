import json
import collections
from typing import Dict, List, Tuple

class BytePairTokenizer:
    """
    A Byte-Pair Encoding (BPE) tokenizer implemented from scratch.
    It operates on byte sequences to avoid out-of-vocabulary issues and uses a merge table to build up subwords.
    """
    def __init__(self) -> None:
        self.special_tokens: Dict[str, int] = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
            "<MASK>": 4
        }
        self.inv_special_tokens: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        
        # Vocab starts at 5. Bytes 0-255 map to IDs 5-260.
        self.vocab: Dict[int, bytes] = {}
        for i in range(256):
            self.vocab[i + 5] = bytes([i])
            
        self.merges: Dict[Tuple[int, int], int] = {}
        self._next_id: int = 261

    @property
    def vocab_size(self) -> int:
        """Returns the current size of the vocabulary."""
        return self._next_id

    def __len__(self) -> int:
        """Returns the current size of the vocabulary."""
        return self.vocab_size

    def train(self, corpus: str, vocab_size: int = 32000) -> None:
        """
        Trains the BPE tokenizer on the given corpus.

        Args:
            corpus: The raw text string to train on.
            vocab_size: The target vocabulary size.
        """
        num_merges = vocab_size - self.vocab_size
        if num_merges <= 0:
            return

        byte_stream = corpus.encode('utf-8')
        ids = [b + 5 for b in byte_stream]

        for _ in range(num_merges):
            if len(ids) < 2:
                break
                
            pairs = zip(ids, ids[1:])
            counts = collections.Counter(pairs)
            
            if not counts:
                break
                
            best_pair = counts.most_common(1)[0][0]
            
            new_id = self._next_id
            self._next_id += 1
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

    def encode(self, text: str) -> List[int]:
        """
        Encodes a text string into a list of token IDs.

        Args:
            text: The input text.

        Returns:
            A list of token IDs.
        """
        byte_stream = text.encode('utf-8')
        ids = [b + 5 for b in byte_stream]
        
        while len(ids) >= 2:
            pairs = list(zip(ids, ids[1:]))
            pair_to_merge = None
            lowest_id = float('inf')
            
            for p in pairs:
                if p in self.merges and self.merges[p] < lowest_id:
                    lowest_id = self.merges[p]
                    pair_to_merge = p
                    
            if pair_to_merge is None:
                break 
                
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair_to_merge:
                    new_ids.append(self.merges[pair_to_merge])
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a list of token IDs back into a text string.

        Args:
            ids: The list of token IDs.

        Returns:
            The decoded string.
        """
        b = bytearray()
        for i in ids:
            if i in self.inv_special_tokens:
                pass
            elif i in self.vocab:
                b.extend(self.vocab[i])
        return b.decode('utf-8', errors='replace')

    def save(self, path: str) -> None:
        """Saves the tokenizer vocabulary to a JSON file."""
        str_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        data = {
            "special_tokens": self.special_tokens,
            "merges": str_merges,
            "next_id": self._next_id
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Loads the tokenizer vocabulary from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.special_tokens = data["special_tokens"]
        self.inv_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self._next_id = data["next_id"]
        
        str_merges = data["merges"]
        self.merges = {}
        for k, v in str_merges.items():
            p1, p2 = map(int, k.split(','))
            self.merges[(p1, p2)] = v
            
        self.vocab = {}
        for i in range(256):
            self.vocab[i + 5] = bytes([i])
            
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for (p1, p2), new_id in sorted_merges:
            self.vocab[new_id] = self.vocab[p1] + self.vocab[p2]
