"""
Attention Layer Decryption Hook (Component D).

Hooks into a Hugging Face transformer's attention layer to dynamically
modify the attention mask based on provided relation subkeys.
Tokens corresponding to relations that lack a valid subkey are masked
out (assigned -inf), preventing the LLM from 'seeing' those unauthorized
structural graph connections.
"""

import torch
import torch.nn as nn
from typing import Dict, Set, Callable, Optional


class AttentionDecryptionHook:
    """
    Manages PyTorch forward hooks for in-model structural decryption.
    """
    
    def __init__(self):
        self.handles = []
        self.active_subkeys: Set[bytes] = set()
        # Maps token sequence index to its required relation subkey
        self.token_requirements: Dict[int, bytes] = {}

    def set_context(self, authorized_subkeys: list[bytes], token_requirements: Dict[int, bytes]):
        """
        Configure the hook with the keys for the current generation request.
        
        Args:
            authorized_subkeys: List of valid 32-byte subkeys from the IntentClassifier.
            token_requirements: Map of token_index -> required 32-byte subkey to unmask.
        """
        self.active_subkeys = set(authorized_subkeys)
        self.token_requirements = token_requirements

    def clear_context(self):
        """Clear context after generation."""
        self.active_subkeys.clear()
        self.token_requirements.clear()

    def _attention_forward_pre_hook(self, module: nn.Module, args: tuple, kwargs: dict):
        """
        Pre-hook for Hugging Face Attention layers (e.g., LlamaAttention, Qwen2Attention).
        Modifies the attention_mask in kwargs before the forward pass computes QK^T.
        """
        if "attention_mask" not in kwargs or kwargs["attention_mask"] is None:
            # If no mask is provided, we can't easily modify it without knowing
            # the exact tensor shapes expected by the specific model architecture.
            # In a robust implementation, we'd construct one.
            return args, kwargs
            
        attn_mask = kwargs["attention_mask"]
        
        # attn_mask is typically [batch_size, 1, seq_len, seq_len] or [batch_size, 1, q_len, kv_len]
        # We need to modify the mask such that queries cannot attend to keys
        # at indices where token_requirements are not met.
        
        # Determine sequence length from the mask shape
        seq_len = attn_mask.size(-1)
        
        # Create a modification mask
        # 0 = allowed, -inf = blocked
        mod_mask = torch.zeros(seq_len, dtype=attn_mask.dtype, device=attn_mask.device)
        
        for token_idx, required_key in self.token_requirements.items():
            if token_idx < seq_len:
                if required_key not in self.active_subkeys:
                    # Key missing or invalid -> mask out this token
                    # PyTorch uses highly negative values for masked positions
                    mod_mask[token_idx] = torch.finfo(attn_mask.dtype).min
                    
        # Apply the modification to the key dimension (last dim)
        # Broadcasting will apply this across batch and query dimensions
        modified_mask = attn_mask + mod_mask.view(1, 1, 1, seq_len)
        
        kwargs["attention_mask"] = modified_mask
        
        return args, kwargs

    def register(self, model: nn.Module) -> None:
        """
        Register the hook on all attention layers of the model.
        """
        self.remove() # clear existing
        
        # Find all attention layers. This heuristic matches most HF models
        # (e.g., LlamaAttention, Qwen2Attention, MistralAttention).
        for name, module in model.named_modules():
            if "attn" in name.lower() or "attention" in name.lower():
                if hasattr(module, "q_proj") or hasattr(module, "c_attn"):
                    # It's likely a self-attention module
                    handle = module.register_forward_pre_hook(
                        self._attention_forward_pre_hook, 
                        with_kwargs=True
                    )
                    self.handles.append(handle)
                    
    def remove(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
