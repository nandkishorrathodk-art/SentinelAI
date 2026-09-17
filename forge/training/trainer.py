"""
Sentinel Neural Model Trainer
=============================
Autonomous PyTorch training loop for Sentinel SA-MoE Transformer.
Supports training from scratch or fine-tuning from existing checkpoints.
Calculates cross-entropy loss + MoE load-balancing auxiliary loss.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from forge.generate import SentinelTransformer, BytePairTokenizer, load_sentinel_model


class CognitiveDataset(Dataset):
    """Chunks raw tokenized streams into fixed context window sequences for training."""

    def __init__(self, token_ids: List[int], seq_len: int = 128):
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.n_chunks = max(0, (len(self.tokens) - seq_len - 1) // seq_len + 1)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y


class SentinelTrainer:
    """Trains and evolves Sentinel's weights on curated cognitive datasets."""

    def __init__(
        self,
        model: SentinelTransformer,
        tokenizer: BytePairTokenizer,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            logits, loss, aux_loss = self.model(x, targets=y)
            combined_loss = loss + 0.01 * aux_loss

            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        elapsed = time.time() - t0
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch:02d}/{total_epochs:02d} | Avg Loss: {avg_loss:.4f} | Elapsed: {elapsed:.1f}s", flush=True)
        return avg_loss

    def train(
        self,
        corpus: str,
        epochs: int = 5,
        batch_size: int = 8,
        seq_len: int = 128,
        output_checkpoint: str = "sentinel_v2.pt"
    ) -> List[float]:
        """Encodes corpus, builds dataloader, and runs full training cycle."""
        print(f"[*] Tokenizing training corpus...")
        encoded = self.tokenizer.encode(corpus)
        print(f"[+] Corpus encoded into {len(encoded):,} tokens.")

        dataset = CognitiveDataset(encoded, seq_len=seq_len)
        print(f"[+] Dataset chunks created: {len(dataset)} sequences (Batch size: {batch_size})")

        if len(dataset) == 0:
            raise ValueError("Corpus too small for the specified sequence length.")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"[*] Starting Sentinel Neural Training for {epochs} epochs on {self.device}...")
        loss_history = []
        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(dataloader, epoch, epochs)
            loss_history.append(avg_loss)

        # Save evolved checkpoint
        os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint)), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "vocab": self.tokenizer.vocab,
            "merges": self.tokenizer.merges,
            "d_model": self.model.d_model,
            "n_layers": len(self.model.layers),
            "n_heads": self.model.layers[0].attn.n_heads,
            "n_experts": self.model.layers[0].moe.n_experts,
            "loss_history": loss_history,
        }, output_checkpoint)

        print(f"[SAVED] Upgraded Sentinel weights saved to: {output_checkpoint}")
        return loss_history