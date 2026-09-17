import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math

class SynapticPlasticity:
    """
    Living Weights - Synaptic Plasticity Module.
    Allows specific layers (attention V projections and expert down projections) 
    to dynamically adapt at inference time based on prediction confidence, simulating Hebbian learning.
    """
    def __init__(self, lr: float = 1e-5, decay_rate: float = 0.999, top_k_updates: int = 128):
        """
        Args:
            lr: Learning rate for synaptic updates
            decay_rate: Weight decay applied to modified weights to prevent drift
            top_k_updates: Number of top active synapses to update per step (efficiency)
        """
        self.lr = lr
        self.decay_rate = decay_rate
        self.top_k_updates = top_k_updates
        self._enabled = True
        
        # Track registered layers and their activations
        self.registered_layers: List[nn.Linear] = []
        self.activations: Dict[nn.Linear, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
    def enable(self):
        """Enable plasticity updates."""
        self._enabled = True
        
    def disable(self):
        """Disable plasticity updates."""
        self._enabled = False
        
    def register_layer(self, layer: nn.Linear):
        """
        Register a layer to undergo synaptic plasticity.
        Usually attention V projections or expert down projections.
        """
        self.registered_layers.append(layer)
        
        def forward_hook(module, input, output):
            if self._enabled:
                # Save input activation (pre-synaptic)
                # Store detached tensor to avoid memory leaks
                self.activations[module] = input[0].detach()
                
        handle = layer.register_forward_hook(forward_hook)
        self.hooks.append(handle)
        
    def compute_reward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute reward based on prediction confidence.
        Reward = -Entropy(Softmax(logits))
        Higher confidence (lower entropy) -> Higher reward
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        # Average over batch and seq length
        avg_entropy = entropy.mean()
        # Scale and invert so low entropy -> positive reward
        reward = -avg_entropy
        return reward
        
    @torch.no_grad()
    def update(self, logits: torch.Tensor):
        """
        Apply Hebbian update to registered layers based on recent activations and confidence reward.
        ΔW = η * reward * (activation_post @ activation_pre.T)
        """
        if not self._enabled or not self.registered_layers:
            return
            
        reward = self.compute_reward(logits)
        
        for layer in self.registered_layers:
            if layer not in self.activations:
                continue
                
            # Pre-synaptic activation: x
            x = self.activations[layer] # (batch, seq_len, in_features)
            batch_size, seq_len, in_features = x.shape
            
            # Post-synaptic activation: y = Wx (+ b)
            # Recompute post-synaptic activation for the batch
            # We flatten batch and seq_len for outer product approximation
            x_flat = x.view(-1, in_features) # (N, in_features)
            y_flat = layer(x).view(-1, layer.out_features) # (N, out_features)
            
            # Compute Hebbian update (simplified outer product, averaged over N)
            # We want roughly y @ x.T
            # Shape of W is (out_features, in_features)
            hebbian_update = torch.matmul(y_flat.T, x_flat) / (batch_size * seq_len)
            
            # Scale by reward and learning rate
            delta_w = self.lr * reward * hebbian_update
            
            # Sparsity: Only update the top-k most active synapses to keep it fast
            if self.top_k_updates > 0 and self.top_k_updates < delta_w.numel():
                # Get threshold for top-k magnitude
                # Flatten, find top-k, mask others to zero
                vals, _ = torch.topk(torch.abs(delta_w).flatten(), self.top_k_updates)
                threshold = vals[-1]
                delta_w = torch.where(torch.abs(delta_w) >= threshold, delta_w, torch.zeros_like(delta_w))
                
            # Apply update
            layer.weight.data += delta_w
            
        # Clear activations
        self.activations.clear()
        
    @torch.no_grad()
    def decay(self):
        """
        Apply weight decay to registered layers to prevent weights from drifting infinitely.
        """
        if not self._enabled:
            return
            
        for layer in self.registered_layers:
            # Simple exponential decay
            layer.weight.data *= self.decay_rate
            
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.registered_layers.clear()
        self.activations.clear()
