import torch
import torch.nn as nn
from typing import Optional, List, Any, Generator

class SentinelGenerator:
    """
    Production inference engine for SentinelAI.
    Features streaming generation with KV-Cache, advanced nucleus sampling, and living weights integration.
    """
    def __init__(self, model: nn.Module, tokenizer: Any):
        """
        Initializes the generator with a trained model and tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None,
        apply_living_weights: bool = False
    ) -> Generator[str, None, None]:
        """
        Generates text iteratively, yielding decoded tokens.
        """
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        generated_tokens = list(input_ids)
        past_key_values = None
        
        stop_tokens_set = set(stop_tokens) if stop_tokens else set()
        if hasattr(self.tokenizer, 'eos_token_id'):
            stop_tokens_set.add(self.tokenizer.eos_token_id)
            
        for _ in range(max_new_tokens):
            # Support for model forward with cache
            outputs = self.model(
                input_ids=input_tensor,
                use_cache=True,
                past_key_values=past_key_values
            )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = getattr(outputs, 'past_key_values', None)
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for token in set(generated_tokens):
                    logits[0, token] /= repetition_penalty
                    
            # Temperature
            if temperature != 1.0:
                logits = logits / temperature
                
            probs = torch.softmax(logits, dim=-1)
            
            # Top-K
            if top_k > 0:
                top_k_vals, _ = torch.topk(probs, top_k)
                min_top_k = top_k_vals[..., -1, None]
                probs = torch.where(probs < min_top_k, torch.zeros_like(probs), probs)
                
            # Top-P
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs = torch.where(indices_to_remove, torch.zeros_like(probs), probs)
                
            # Min-P
            if min_p > 0.0:
                max_prob = torch.max(probs, dim=-1, keepdim=True)[0]
                probs = torch.where(probs < (min_p * max_prob), torch.zeros_like(probs), probs)
                
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)[0, 0].item()
            
            generated_tokens.append(next_token)
            yield self.tokenizer.decode([next_token])
            
            if next_token in stop_tokens_set:
                break
                
            input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            
        if apply_living_weights and hasattr(self.model, 'apply_micro_plasticity'):
            self.model.apply_micro_plasticity(generated_tokens)

    def generate(self, *args, **kwargs) -> str:
        """
        Generates text synchronously, returning the full string.
        """
        result = []
        for token_str in self.generate_stream(*args, **kwargs):
            result.append(token_str)
        return "".join(result)
