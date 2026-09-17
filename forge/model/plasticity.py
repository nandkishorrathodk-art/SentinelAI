import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SynapticPlasticity:
    """
    Living Weights - Synaptic Plasticity Module.
    Allows specific layers (attention V projections and expert down projections) 
    to dynamically adapt at inference time based on prediction confidence, simulating Hebbian learning.
    """
    def __init__(self, lr: float = 1e-5, decay_rate: float = 0.999, top_k_updates: int = 128):
        self.lr = lr
        self.decay_rate = decay_rate
        self.top_k_updates = top_k_updates
        self._enabled = True
        self.registered_layers: List[nn.Linear] = []
        self.activations: Dict[nn.Linear, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
    def enable(self):
        self._enabled = True
        
    def disable(self):
        self._enabled = False
        
    def register_layer(self, layer: nn.Linear):
        self.registered_layers.append(layer)
        
        def forward_hook(module, input, output):
            if self._enabled:
                self.activations[module] = input[0].detach()
                
        handle = layer.register_forward_hook(forward_hook)
        self.hooks.append(handle)
        
    def compute_reward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        avg_entropy = entropy.mean()
        reward = -avg_entropy
        return reward
        
    @torch.no_grad()
    def update(self, logits: torch.Tensor):
        if not self._enabled or not self.registered_layers:
            return
            
        reward = self.compute_reward(logits)
        
        for layer in self.registered_layers:
            if layer not in self.activations:
                continue
                
            x = self.activations[layer]
            batch_size, seq_len, in_features = x.shape
            x_flat = x.view(-1, in_features)
            y_flat = layer(x).view(-1, layer.out_features)
            hebbian_update = torch.matmul(y_flat.T, x_flat) / (batch_size * seq_len)
            delta_w = self.lr * reward * hebbian_update
            
            if self.top_k_updates > 0 and self.top_k_updates < delta_w.numel():
                vals, _ = torch.topk(torch.abs(delta_w).flatten(), self.top_k_updates)
                threshold = vals[-1]
                delta_w = torch.where(torch.abs(delta_w) >= threshold, delta_w, torch.zeros_like(delta_w))
                
            layer.weight.data += delta_w
            
        self.activations.clear()
        
    @torch.no_grad()
    def decay(self):
        if not self._enabled:
            return
        for layer in self.registered_layers:
            layer.weight.data *= self.decay_rate
            
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.registered_layers.clear()
        self.activations.clear()

