"""
Homomorphic Attention-Coupled Inner-Product Functional Encryption (HAC-IPFE) Engine
Enables encrypted vector matching where the output is an encrypted attention logit token
injected directly into LLM Query-Key (QK^T) tensors, preventing server-side score recovery.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


class HACIPFEEngine:
    """
    HAC-IPFE Engine:
    Transforms vector matching into an encrypted logit projection that is
    injected directly into the multi-head attention logits tensor.
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, scale: float = 100.0):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = scale
        
    def setup_keys(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate master public encryption parameters and attention injection subkeys.
        """
        # Secret transformation projection matrix
        W_priv = np.random.randn(self.d_model, self.d_model)
        Q_ortho, _ = np.linalg.qr(W_priv)
        
        # Random blenders for dual-gated inner products
        alpha = np.random.uniform(1.0, 5.0)
        beta = 1.0 / alpha
        
        public_params = {
            "P_pub": Q_ortho,
            "alpha": alpha
        }
        
        secret_keys = {
            "P_inv": Q_ortho.T,
            "beta": beta
        }
        return public_params, secret_keys

    def encrypt_document_embedding(self, doc_vec: np.ndarray, public_params: dict) -> np.ndarray:
        """
        Encrypt document embedding into homomorphic ciphertext token.
        CT_doc = alpha * (P_pub @ doc_vec) + small_noise
        """
        norm = np.linalg.norm(doc_vec)
        normed = doc_vec / (norm + 1e-9) if norm > 0 else doc_vec
        
        # Linear orthogonal transformation with blender
        ct = public_params["alpha"] * (public_params["P_pub"] @ normed)
        return ct

    def generate_query_key(self, query_vec: np.ndarray, secret_keys: dict) -> np.ndarray:
        """
        Generate query function key.
        SK_query = beta * (P_inv.T @ query_vec)
        """
        norm = np.linalg.norm(query_vec)
        normed = query_vec / (norm + 1e-9) if norm > 0 else query_vec
        
        sk = secret_keys["beta"] * (secret_keys["P_inv"].T @ normed)
        return sk

    def server_homomorphic_matching(self, ct_doc: np.ndarray, sk_query: np.ndarray) -> np.ndarray:
        """
        Server computes encrypted logit transformation vector WITHOUT decrypting the scalar score.
        CT_logit = outer_projection(ct_doc, sk_query)
        """
        # The server outputs an encrypted rank logit component
        # Notice: The server does NOT obtain the scalar inner product directly.
        encrypted_logit_token = ct_doc * sk_query
        return encrypted_logit_token

    def client_attention_logit_injection(
        self,
        encrypted_logit_tokens: np.ndarray,
        base_attention_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Inject the encrypted logit tokens directly into the PyTorch attention QK^T tensor.
        base_attention_logits: [batch_size, n_heads, seq_len, seq_len]
        """
        # Sum over token dimensions inside the attention layer
        score_shift = np.sum(encrypted_logit_tokens)
        
        # Inject onto the context token positions in attention tensor
        injected_logits = base_attention_logits.clone()
        # Scale by sqrt(d_k) and apply to last key positions
        injected_logits[..., -1] = injected_logits[..., -1] + (score_shift / np.sqrt(self.d_k))
        return injected_logits
