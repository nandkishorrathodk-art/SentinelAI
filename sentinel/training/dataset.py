"""
SentinelAI — Training Dataset
=============================
Loads raw text or JSONL instruction data, tokenizes it, and produces
fixed-length (input, target) pairs for next-token-prediction training.

Supports:
  - Plain text files (pre-training)
  - JSONL instruction pairs (fine-tuning / distillation)
  - Cybersecurity-focused data: CVE descriptions, exploit code, bash commands
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, IterableDataset


class TextChunkDataset(Dataset):
    """Fixed-length token-chunk dataset for autoregressive pre-training.

    Takes a flat token-ID tensor and slices it into overlapping (input, target)
    windows where target is the input shifted right by one position.

    Example:
        tokens = [10, 20, 30, 40, 50, 60]
        seq_len = 4
        chunk 0: input=[10,20,30,40]  target=[20,30,40,50]
        chunk 1: input=[20,30,40,50]  target=[30,40,50,60]
    """

    def __init__(
        self,
        token_ids: list[int] | torch.Tensor,
        seq_len: int = 512,
        stride: int | None = None,
    ):
        if isinstance(token_ids, list):
            self.tokens = torch.tensor(token_ids, dtype=torch.long)
        else:
            self.tokens = token_ids.long()

        self.seq_len = seq_len
        self.stride = stride or seq_len  # non-overlapping by default

        # Number of complete chunks
        self.n_chunks = max(0, (len(self.tokens) - seq_len - 1) // self.stride + 1)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


class InstructionDataset(Dataset):
    """JSONL instruction-response dataset for supervised fine-tuning.

    Each line in the JSONL file should have:
        {"prompt": "...", "response": "..."}
    or:
        {"instruction": "...", "output": "..."}

    The tokenizer encodes: <BOS> prompt <SEP> response <EOS>
    and creates (input, target) pairs with the prompt portion masked
    (loss = -100 on prompt tokens so only response is trained).
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,  # BytePairTokenizer
        max_seq_len: int = 512,
        mask_prompt: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_prompt = mask_prompt
        self.samples: list[dict[str, str]] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt = item.get("prompt") or item.get("instruction", "")
                response = item.get("response") or item.get("output", "")
                if prompt and response:
                    self.samples.append({"prompt": prompt, "response": response})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        bos = self.tokenizer.special_tokens["<BOS>"]
        eos = self.tokenizer.special_tokens["<EOS>"]
        pad = self.tokenizer.special_tokens["<PAD>"]

        prompt_ids = self.tokenizer.encode(sample["prompt"])
        response_ids = self.tokenizer.encode(sample["response"])

        # Build sequence: <BOS> prompt response <EOS>
        full_ids = [bos] + prompt_ids + response_ids + [eos]

        # Truncate to max_seq_len
        if len(full_ids) > self.max_seq_len + 1:
            full_ids = full_ids[: self.max_seq_len + 1]

        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]

        # Mask prompt tokens in target (so loss only applies to response)
        if self.mask_prompt:
            prompt_len = 1 + len(prompt_ids)  # <BOS> + prompt
            prompt_len = min(prompt_len, len(target_ids))
            for i in range(prompt_len):
                target_ids[i] = -100  # PyTorch CrossEntropyLoss ignores -100

        # Pad to fixed length
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [pad] * pad_len
            target_ids = target_ids + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


class StreamingTextDataset(IterableDataset):
    """Memory-efficient streaming dataset for large text files.

    Reads text files line-by-line, tokenizes on the fly, and yields
    fixed-length chunks without loading the entire corpus into memory.
    Useful for multi-GB cybersecurity corpora.
    """

    def __init__(
        self,
        file_paths: list[str | Path],
        tokenizer,
        seq_len: int = 512,
        shuffle_files: bool = True,
    ):
        self.file_paths = [Path(p) for p in file_paths]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_files = shuffle_files

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        buffer: list[int] = []
        files = list(self.file_paths)

        if self.shuffle_files:
            import random
            random.shuffle(files)

        for file_path in files:
            if not file_path.exists():
                continue
            with open(file_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = self.tokenizer.encode(line)
                    buffer.extend(tokens)

                    # Yield complete chunks from buffer
                    while len(buffer) >= self.seq_len + 1:
                        chunk = buffer[: self.seq_len + 1]
                        x = torch.tensor(chunk[:-1], dtype=torch.long)
                        y = torch.tensor(chunk[1:], dtype=torch.long)
                        yield x, y
                        buffer = buffer[self.seq_len:]


def load_text_corpus(
    path: str | Path,
    tokenizer,
    seq_len: int = 512,
) -> TextChunkDataset:
    """Convenience: load a single text file into a TextChunkDataset."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    token_ids = tokenizer.encode(text)
    return TextChunkDataset(token_ids, seq_len=seq_len)


def create_cyber_pretraining_mix(
    data_dir: str | Path,
    tokenizer,
    seq_len: int = 512,
) -> TextChunkDataset:
    """Load and mix multiple cybersecurity text sources into one dataset.

    Expected directory structure:
        data_dir/
            exploits/       # CVE descriptions, exploit code
            commands/        # bash/powershell command logs
            writeups/        # CTF writeups, pentest reports
            code/            # Python/C/Rust security tools
            general/         # general text for language grounding
    """
    data_dir = Path(data_dir)
    all_tokens: list[int] = []

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for txt_file in sorted(subdir.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            # Add EOS between files
            all_tokens.append(tokenizer.special_tokens.get("<EOS>", 2))

    if not all_tokens:
        raise ValueError(f"No .txt files found under {data_dir}")

    return TextChunkDataset(all_tokens, seq_len=seq_len)

