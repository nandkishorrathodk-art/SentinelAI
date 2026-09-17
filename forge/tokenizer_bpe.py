"""
Forge / primeASI — Lossless Word-Bounded Byte-Pair Encoding (BPE) Tokenizer
==========================================================================
BPE implementation supporting arbitrary vocabulary scales, byte-level fallback,
and special tokens (<PAD>, <BOS>, <EOS>, <UNK>, <MASK>).
"""
from __future__ import annotations

import collections
import json
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class BytePairTokenizer:
    """A Byte-Pair Encoding (BPE) tokenizer implemented from scratch.
    Operates on byte sequences with word-boundary isolation to prevent collapse.
    """

    def __init__(self) -> None:
        self.special_tokens: Dict[str, int] = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
            "<MASK>": 4,
        }
        self.inv_special_tokens: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}

        # Base vocab: IDs 5-260 map to raw bytes 0-255
        self.vocab: Dict[int, bytes] = {i + 5: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int, int], int] = {}
        self._next_id: int = 261

    @property
    def vocab_size(self) -> int:
        """Returns the current size of the vocabulary."""
        return self._next_id

    def __len__(self) -> int:
        return self._next_id

    def train(self, corpus: str, vocab_size: int = 2048) -> None:
        """Trains the BPE tokenizer on the given corpus with word-boundary isolation."""
        num_merges = vocab_size - self._next_id
        if num_merges <= 0:
            return

        words = re.findall(r"\s+|\w+|[^\w\s]", corpus)
        word_token_lists = [[b + 5 for b in w.encode("utf-8")] for w in words]

        for _ in range(num_merges):
            pairs = collections.Counter()
            for w in word_token_lists:
                for i in range(len(w) - 1):
                    pairs[(w[i], w[i + 1])] += 1

            if not pairs:
                break

            best_pair, freq = pairs.most_common(1)[0]
            if freq < 2:
                break

            new_id = self._next_id
            self._next_id += 1
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            new_lists = []
            for w in word_token_lists:
                i = 0
                nw = []
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == best_pair:
                        nw.append(new_id)
                        i += 2
                    else:
                        nw.append(w[i])
                        i += 1
                new_lists.append(nw)
            word_token_lists = new_lists

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encodes a text string into a list of token IDs."""
        words = re.findall(r"\s+|\w+|[^\w\s]", text)
        out: List[int] = []
        if add_bos:
            out.append(self.special_tokens["<BOS>"])

        for w in words:
            ids = [b + 5 for b in w.encode("utf-8")]
            while len(ids) >= 2:
                pairs = list(zip(ids, ids[1:]))
                pair_to_merge = None
                lowest_id = float("inf")
                for p in pairs:
                    if p in self.merges and self.merges[p] < lowest_id:
                        lowest_id = self.merges[p]
                        pair_to_merge = p
                if pair_to_merge is None:
                    break
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair_to_merge:
                        new_ids.append(self.merges[pair_to_merge])
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids
            out.extend(ids)

        if add_eos:
            out.append(self.special_tokens["<EOS>"])
        return out

    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
        """Decodes a list of token IDs back into a text string."""
        b = bytearray()
        for i in ids:
            if i in self.inv_special_tokens:
                if not skip_specials:
                    b.extend(self.inv_special_tokens[i].encode("utf-8"))
            elif i in self.vocab:
                b.extend(self.vocab[i])
            else:
                b.extend(b"?")
        return b.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """Saves tokenizer state (merges and vocabulary) to a JSON file."""
        serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        serializable_vocab = {str(k): list(v) for k, v in self.vocab.items()}
        data = {
            "special_tokens": self.special_tokens,
            "merges": serializable_merges,
            "vocab": serializable_vocab,
            "next_id": self._next_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Loads tokenizer state from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.special_tokens = data["special_tokens"]
        self.inv_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.merges = {
            (int(k.split(",")[0]), int(k.split(",")[1])): v
            for k, v in data["merges"].items()
        }
        self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        self._next_id = data["next_id"]

