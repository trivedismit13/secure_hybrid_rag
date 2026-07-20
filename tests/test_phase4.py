"""
Tests for Phase 4: Relation Subkeys & Attention Hook.
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from shield_rag.schema.ontology import RelationType, GraphNode
from shield_rag.crypto.relation_subkeys import RelationKeyManager
from shield_rag.decrypt_attn.hook import AttentionDecryptionHook
from shield_rag.generation.structured_prompt import StructuredPromptBuilder


class DummyAttention(nn.Module):
    """A dummy attention layer to test the pre-hook mask modification."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)  # satisfy the registration heuristic
        
    def forward(self, hidden_states, attention_mask=None):
        # Just return the mask so we can inspect what the hook did to it
        return attention_mask


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DummyAttention()
        
    def forward(self, hidden_states, attention_mask=None):
        return self.attn(hidden_states, attention_mask=attention_mask)


class TestComponentD:
    def test_relation_subkeys(self):
        master_key = b"A" * 32
        manager = RelationKeyManager(master_key)
        
        # Test derivation determinism
        key1 = manager.derive_subkey(RelationType.SATISFY)
        key2 = manager.derive_subkey(RelationType.SATISFY)
        assert key1 == key2
        
        # Test isolation
        key3 = manager.derive_subkey(RelationType.PART_OF)
        assert key1 != key3
        
        # Test verification
        assert manager.verify_token_tag(RelationType.SATISFY, key1) is True
        assert manager.verify_token_tag(RelationType.SATISFY, key3) is False
        
        # Test authorization filter
        auth_keys = manager.get_authorized_subkeys({RelationType.SATISFY})
        assert RelationType.SATISFY in auth_keys
        assert RelationType.PART_OF not in auth_keys

    def test_structured_prompt_builder(self):
        # Use a small fast tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        master_key = b"B" * 32
        manager = RelationKeyManager(master_key)
        builder = StructuredPromptBuilder(tokenizer, manager)
        
        nodes = [
            GraphNode("n1", None, "Pump A", []),
            GraphNode("n2", None, "Req B", []),
        ]
        triples = [
            ("n1", RelationType.SATISFY, "n2")
        ]
        
        prompt, reqs = builder.build_prompt("Is it safe?", nodes, triples)
        
        assert "Pump A" in prompt
        assert "Req B" in prompt
        assert RelationType.SATISFY.value in prompt
        
        # Verify that token requirements were generated
        assert len(reqs) > 0
        
        # All required tokens for this relation should map to the SATISFY subkey
        expected_key = manager.derive_subkey(RelationType.SATISFY)
        for idx, key in reqs.items():
            assert key == expected_key

    def test_attention_hook(self):
        model = DummyModel()
        hook = AttentionDecryptionHook()
        hook.register(model)
        
        # Simulate sequence length 10
        seq_len = 10
        batch_size = 1
        
        # Base attention mask (zeros mean allow, standard in HF before -inf addition)
        base_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        
        # Token 5 requires key A (authorized)
        # Token 6 requires key B (unauthorized)
        key_A = b"KeyA" * 8
        key_B = b"KeyB" * 8
        
        hook.set_context(
            authorized_subkeys=[key_A],
            token_requirements={5: key_A, 6: key_B}
        )
        
        # Forward pass
        hidden = torch.randn(batch_size, seq_len, 4)
        modified_mask = model(hidden, attention_mask=base_mask.clone())
        
        # Token 5 should be unchanged (0.0)
        assert modified_mask[0, 0, 0, 5].item() == 0.0
        
        # Token 6 should be masked out (highly negative)
        assert modified_mask[0, 0, 0, 6].item() < -1e4
        
        # Other tokens should be unchanged
        assert modified_mask[0, 0, 0, 4].item() == 0.0
        
        hook.remove()
