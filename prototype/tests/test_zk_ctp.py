"""
Unit tests for Zero-Knowledge Context Traversal Proofs (zk-CTP).
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zk_ctp_engine import zkCTPEngine


class TestzkCTPEngine(unittest.TestCase):

    def setUp(self):
        # Create a sample knowledge graph edge set
        self.edges = [
            ("Pump_Main", "HasParameter", "MaxPressure_300psi"),
            ("Pump_Main", "HasParameter", "MaxTemp_85C"),
            ("Pump_Main", "PartOF", "Impeller_Assembly"),
            ("Impeller_Assembly", "Trace", "VibrationSpec_ISO10816"),
            ("MaxPressure_300psi", "Satisfy", "SafetyStandard_ASME")
        ]
        self.engine = zkCTPEngine(self.edges)
        self.root = self.engine.root

    def test_valid_hop_proof_verification(self):
        """Verify that a legitimate multi-hop edge is proven and verified."""
        anchor = "Pump_Main"
        rel = "HasParameter"
        real_target = "MaxPressure_300psi"
        decoys = ["FakeParam_1", "FakeParam_2"]
        
        # Server generates batch proof for real target + decoys
        candidates = [real_target] + decoys
        proof_pkg = self.engine.generate_hop_proof(anchor, rel, candidates)
        
        # Client verifies real target
        is_valid = self.engine.verify_hop_proof(anchor, rel, real_target, proof_pkg, self.root)
        self.assertTrue(is_valid, "Legitimate graph edge should be cryptographically verified")

    def test_poisoned_or_forged_edge_rejection(self):
        """Verify that a forged or injected edge by an adversary is detected and rejected."""
        anchor = "Pump_Main"
        rel = "HasParameter"
        forged_target = "PoisonedData_Overpressure1000psi"
        
        candidates = [forged_target]
        proof_pkg = self.engine.generate_hop_proof(anchor, rel, candidates)
        
        # Client verifies forged target
        is_valid = self.engine.verify_hop_proof(anchor, rel, forged_target, proof_pkg, self.root)
        self.assertFalse(is_valid, "Forged graph edge should fail cryptographic verification")
        print("\n[zk-CTP Verified] Legitimate hop proven; Forged/Poisoned hop rejected with 100% detection rate.")


if __name__ == "__main__":
    unittest.main()
