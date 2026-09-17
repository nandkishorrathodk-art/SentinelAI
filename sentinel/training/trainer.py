"""
SentinelAI — Training Engine
=============================
Complete training loop with:
  - AdamW optimizer with cosine annealing + linear warmup
  - Mixed-precision training (torch.amp) for T4 GPU efficiency
  - Gradient clipping to prevent explosion
  - MoE auxiliary load-balancing loss integration
  - Checkpointing: save/resume training state
  - Dual-GPU support via DataParallel (Kaggle 2x T4)
  - Comprehensive logging: loss, tokens/sec, LR, expert utilization
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sentinel.model.config import SentinelConfig


@dataclass
class TrainingMetrics:
    """Snapshot of training state for logging."""
    step: int
    loss: float
    moe_aux_loss: float
    learning_rate: float
    tokens_per_sec: float
    grad_norm: float
    elapsed_seconds: float
    gpu_memory_mb: float = 0.0


class CosineWarmupScheduler:
    """Linear warmup followed by cosine decay to min_lr.

    Learning rate schedule:
        warmup:  lr = base_lr * (step / warmup_steps)
        cosine:  lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        base_lr: float = 3e-4,
        min_lr: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self._step = 0

    def step(self) -> float:
        """Advance one step and update optimizer LR. Returns current LR."""
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _get_lr(self) -> float:
        if self._step < self.warmup_steps:
            return self.base_lr * (self._step / max(1, self.warmup_steps))
        if self._step >= self.max_steps:
            return self.min_lr
        progress = (self._step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    @property
    def current_lr(self) -> float:
        return self._get_lr()


class SentinelTrainer:
    """Full training engine for SentinelAI.

    Handles the complete training lifecycle:
      1. Forward pass through model
      2. Cross-entropy loss + MoE auxiliary loss
      3. Backward pass with gradient scaling (AMP)
      4. Gradient clipping
      5. Optimizer step + LR schedule
      6. Logging and checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        config: SentinelConfig,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        log_interval: int = 10,
        checkpoint_interval: int = 500,
        device: str | torch.device | None = None,
    ):
        # Device selection
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.config = config
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Multi-GPU: wrap with DataParallel if multiple GPUs available
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.n_gpus > 1:
            print(f"[SentinelAI] Using {self.n_gpus} GPUs via DataParallel")
            self.model = nn.DataParallel(model).to(self.device)
        else:
            self.model = model.to(self.device)

        self.raw_model = model  # unwrapped reference for saving

        # Optimizer: AdamW with decoupled weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
            fused=torch.cuda.is_available(),  # fused AdamW on CUDA
        )

        # LR Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            base_lr=config.learning_rate,
        )

        # Mixed precision scaler (for FP16 on GPU)
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
            )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.training_log: list[TrainingMetrics] = []

    def train(self, max_steps: int | None = None) -> list[TrainingMetrics]:
        """Run the full training loop.

        Args:
            max_steps: Override config.max_steps if provided.

        Returns:
            List of TrainingMetrics snapshots.
        """
        max_steps = max_steps or self.config.max_steps
        self.model.train()

        total_tokens = 0
        epoch = 0
        t_start = time.perf_counter()

        print(f"\n{'='*60}")
        print(f"  SentinelAI Training Started")
        print(f"  Device: {self.device} | GPUs: {max(1, self.n_gpus)}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Max Steps: {max_steps} | Batch Size: {self.config.batch_size}")
        print(f"  AMP: {self.use_amp}")
        print(f"{'='*60}\n")

        while self.global_step < max_steps:
            epoch += 1
            for batch_x, batch_y in self.train_loader:
                if self.global_step >= max_steps:
                    break

                metrics = self._train_step(batch_x, batch_y)
                total_tokens += batch_x.numel()

                # Logging
                if self.global_step % self.log_interval == 0:
                    elapsed = time.perf_counter() - t_start
                    metrics.tokens_per_sec = total_tokens / max(elapsed, 1e-6)
                    metrics.elapsed_seconds = elapsed
                    if torch.cuda.is_available():
                        metrics.gpu_memory_mb = (
                            torch.cuda.max_memory_allocated() / 1024 / 1024
                        )

                    self.training_log.append(metrics)
                    self._log(metrics)

                # Checkpointing
                if self.global_step % self.checkpoint_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

                # Validation
                if (
                    self.val_loader is not None
                    and self.global_step % (self.checkpoint_interval * 2) == 0
                ):
                    val_loss = self.validate()
                    print(f"  [VAL] Step {self.global_step} | Val Loss: {val_loss:.4f}")
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best")

        # Final save
        self.save_checkpoint("final")
        print(f"\n{'='*60}")
        print(f"  Training Complete! Final Loss: {self.training_log[-1].loss:.4f}")
        print(f"{'='*60}\n")

        return self.training_log

    def _train_step(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> TrainingMetrics:
        """Execute a single training step."""
        self.global_step += 1
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            logits, ce_loss, moe_aux = self.model(batch_x, targets=batch_y)

            # Total loss = cross-entropy + weighted MoE load balancing
            loss = ce_loss + self.config.moe_aux_loss_weight * moe_aux

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        # Gradient clipping (unscale first for accurate norm)
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # LR schedule
        lr = self.scheduler.step()

        return TrainingMetrics(
            step=self.global_step,
            loss=loss.item(),
            moe_aux_loss=moe_aux.item() if isinstance(moe_aux, torch.Tensor) else moe_aux,
            learning_rate=lr,
            tokens_per_sec=0.0,  # filled in by caller
            grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            elapsed_seconds=0.0,
        )

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average loss."""
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in self.val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits, ce_loss, _ = self.model(batch_x, targets=batch_y)

            total_loss += ce_loss.item()
            n_batches += 1

        self.model.train()
        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, name: str) -> Path:
        """Save model + optimizer + scheduler state."""
        ckpt_path = self.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "model_state_dict": self.raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_step": self.scheduler._step,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            ckpt_path,
        )
        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler._step = ckpt.get("scheduler_step", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[SentinelAI] Resumed from step {self.global_step}")

    def _log(self, m: TrainingMetrics) -> None:
        """Print training metrics."""
        gpu_info = f" | GPU: {m.gpu_memory_mb:.0f}MB" if m.gpu_memory_mb > 0 else ""
        print(
            f"  Step {m.step:>6d} | "
            f"Loss: {m.loss:.4f} | "
            f"MoE Aux: {m.moe_aux_loss:.4f} | "
            f"LR: {m.learning_rate:.2e} | "
            f"Tok/s: {m.tokens_per_sec:,.0f} | "
            f"Grad: {m.grad_norm:.2f}"
            f"{gpu_info}"
        )

