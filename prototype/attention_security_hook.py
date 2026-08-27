"""
Attention-Layer Cryptographic Anti-Poisoning Hook (ACW)
Attaches to PyTorch MultiheadAttention modules to verify KDC token signatures
and inject -inf attention masks to mathematically erase poisoned or forged context tokens.
"""

import hmac
import hashlib
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional


class AttentionSecurityHook:
    """
    Cryptographic PyTorch pre-hook for transformer self-attention layers.
    Verifies context tokens and suppresses unauthorized or poisoned tokens via -inf masking.
    """
    def __init__(self, kdc_root_key: bytes):
        self.kdc_root_key = kdc_root_key
        self.hook_handle = None
        self.active_token_signatures: List[str] = []
        self.active_token_payloads: List[str] = []

    def sign_context_token(self, doc_id: str, token_content: str) -> str:
        """
        KDC generates an HMAC-SHA256 signature for authorized context tokens.
        """
        payload = f"{doc_id}::{token_content}".encode("utf-8")
        sig = hmac.new(self.kdc_root_key, payload, hashlib.sha256).hexdigest()
        return sig

    def verify_token_signature(self, doc_id: str, token_content: str, signature: str) -> bool:
        """
        Verify signature authenticity against KDC root key.
        """
        expected_sig = self.sign_context_token(doc_id, token_content)
        return hmac.compare_digest(expected_sig, signature)

    def set_batch_context(self, token_payloads: List[Tuple[str, str]], signatures: List[str]):
        """
        Set context payload and signatures for the upcoming forward pass.
        token_payloads: List of (doc_id, token_content)
        """
        self.active_token_payloads = token_payloads
        self.active_token_signatures = signatures

    def create_security_mask(self, seq_len: int, num_context_tokens: int) -> torch.Tensor:
        """
        Generate additive attention mask matrix M:
        M[i, j] = 0.0 for verified tokens; M[i, j] = -inf for forged/poisoned tokens.
        """
        mask = torch.zeros((seq_len, seq_len), dtype=torch.float32)
        
        # Check signature of each context token (placed at the end of prompt sequence)
        context_start = seq_len - num_context_tokens
        
        for idx in range(num_context_tokens):
            token_pos = context_start + idx
            doc_id, content = self.active_token_payloads[idx]
            sig = self.active_token_signatures[idx]
            
            is_valid = self.verify_token_signature(doc_id, content, sig)
            if not is_valid:
                # Mask out this poisoned context token from influencing all query positions
                mask[:, token_pos] = -float("inf")
                
        return mask

    def attach_to_module(self, attention_module: nn.Module):
        """
        Attach forward_pre_hook to PyTorch multi-head attention module.
        """
        def pre_hook(module, args):
            # args: (query, key, value, key_padding_mask, need_weights, attn_mask, ...)
            # We inject the computed security mask into attn_mask argument
            query, key, value = args[0], args[1], args[2]
            seq_len = key.shape[0] if key.dim() == 3 else key.shape[1]
            num_ctx = len(self.active_token_signatures)
            
            if num_ctx > 0:
                sec_mask = self.create_security_mask(seq_len, num_ctx).to(key.device)
                # Return modified args tuple
                new_args = list(args)
                if len(new_args) >= 6:
                    existing_mask = new_args[5]
                    if existing_mask is not None:
                        new_args[5] = existing_mask + sec_mask
                    else:
                        new_args[5] = sec_mask
                return tuple(new_args)
            return args

        self.hook_handle = attention_module.register_forward_pre_hook(pre_hook)

    def remove(self):
        """Detach hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
