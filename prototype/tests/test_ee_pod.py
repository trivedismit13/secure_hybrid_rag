"""
Unit tests for Epistemic-Entangled Onion Decryption (EE-POD).
"""

import unittest
import torch
import hashlib
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ee_pod_engine import EEPODEngine


class TestEEPODEngine(unittest.TestCase):

    def setUp(self):
        self.engine = EEPODEngine(entropy_threshold=1.0, num_layers=3)
        self.keys = [
            hashlib.sha256(b"key_layer_0").digest(),
            hashlib.sha256(b"key_layer_1").digest(),
            hashlib.sha256(b"key_layer_2").digest()
        ]
        self.evidence_layers = [
            "Level 0: Standard pump operating pressure is 150 psi.",
            "Level 1: Transient overpressure limits up to 220 psi for 30 seconds.",
            "Level 2: Emergency shutdown valve triggers strictly at 250 psi."
        ]
        self.onion_pkg = self.engine.create_onion_ciphertext(self.evidence_layers, self.keys)
        self.available_keys = {0: self.keys[0], 1: self.keys[1], 2: self.keys[2]}

    def test_low_entropy_no_unmask(self):
        """Under low entropy (high confidence), deep onion layers remain masked."""
        # Simulated low entropy logits (sharp probability peak)
        logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
        hidden_state = torch.randn(1, 16)
        
        peeled, text, entropy = self.engine.evaluate_and_peel(
            current_layer=1,
            onion_package=self.onion_pkg,
            logits=logits,
            hidden_state=hidden_state,
            available_keys=self.available_keys
        )
        self.assertFalse(peeled)
        self.assertIsNone(text)
        self.assertLess(entropy, 1.0)

    def test_high_entropy_triggers_trapdoor_unmask(self):
        """Under high entropy (uncertainty spike), the next evidence layer is dynamically peeled."""
        # Simulated high entropy logits (uniform distribution across tokens)
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        hidden_state = torch.randn(1, 16)
        
        peeled, text, entropy = self.engine.evaluate_and_peel(
            current_layer=1,
            onion_package=self.onion_pkg,
            logits=logits,
            hidden_state=hidden_state,
            available_keys=self.available_keys
        )
        self.assertTrue(peeled)
        self.assertIn("Level 1:", text)
        self.assertGreater(entropy, 1.0)
        print(f"\n[EE-POD Verified] Entropy Spike ({entropy:.2f} > 1.0) triggered trapdoor unmasking: '{text[:45]}...'")


if __name__ == "__main__":
    unittest.main()
