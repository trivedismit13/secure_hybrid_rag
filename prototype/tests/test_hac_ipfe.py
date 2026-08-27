"""
Unit tests for Homomorphic Attention-Coupled IPFE (HAC-IPFE).
"""

import unittest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hac_ipfe_engine import HACIPFEEngine


class TestHACIPFEEngine(unittest.TestCase):

    def test_hac_ipfe_attention_injection(self):
        d_model = 64
        n_heads = 4
        engine = HACIPFEEngine(d_model=d_model, n_heads=n_heads)
        
        pub_params, sec_keys = engine.setup_keys()
        
        # Test query and documents
        query = np.random.randn(d_model)
        query = query / np.linalg.norm(query)
        
        # High relevance doc
        doc_relevant = query + np.random.normal(0, 0.05, size=d_model)
        doc_relevant = doc_relevant / np.linalg.norm(doc_relevant)
        
        # Low relevance doc
        doc_irrelevant = -query
        
        # Encrypt
        ct_rel = engine.encrypt_document_embedding(doc_relevant, pub_params)
        ct_irrel = engine.encrypt_document_embedding(doc_irrelevant, pub_params)
        
        sk_q = engine.generate_query_key(query, sec_keys)
        
        # Server produces encrypted logit tokens (server never sees true scalar score)
        logit_token_rel = engine.server_homomorphic_matching(ct_rel, sk_q)
        logit_token_irrel = engine.server_homomorphic_matching(ct_irrel, sk_q)
        
        # Simulated attention tensor [batch=1, heads=4, seq=8, seq=8]
        base_logits = torch.zeros((1, n_heads, 8, 8))
        
        # Inject into attention layer
        injected_rel = engine.client_attention_logit_injection(logit_token_rel, base_logits)
        injected_irrel = engine.client_attention_logit_injection(logit_token_irrel, base_logits)
        
        # Calculate softmax attention weights
        attn_rel = torch.softmax(injected_rel, dim=-1)
        attn_irrel = torch.softmax(injected_irrel, dim=-1)
        
        # Relevant doc should boost attention weight on context token (position -1)
        rel_attn_weight = attn_rel[0, 0, 0, -1].item()
        irrel_attn_weight = attn_irrel[0, 0, 0, -1].item()
        
        self.assertGreater(rel_attn_weight, irrel_attn_weight)
        print(f"\n[HAC-IPFE Verified] Relevant Attention: {rel_attn_weight:.4f} | Irrelevant Attention: {irrel_attn_weight:.4f}")


if __name__ == "__main__":
    unittest.main()
