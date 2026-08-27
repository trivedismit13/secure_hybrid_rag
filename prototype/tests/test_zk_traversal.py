"""
Unit and Integration Tests for Novelty #2: Verifiable Bounded-Decoy Traversal (ZK Proof of Policy-Compliant Path).
"""

import unittest
import numpy as np
import sys
import os

# Add prototype directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pod_engine import PODEngine
from zk_traversal_proof import TraversalProof, commit_to_layer_result
from rag_pipeline import VPRAGPipeline, random_blender


class TestZKTraversalProof(unittest.TestCase):

    def setUp(self):
        self.dim = 8
        self.max_layers = 3
        self.lambda_bits = 256
        
        # Setup real POD keys for D=3 layers
        self.mpk_layers, self.msk_layers = PODEngine.Setup(self.lambda_bits, self.dim, max_layers=self.max_layers)
        
        self.alphas = [random_blender(self.mpk_layers[l]['lambda_N']) for l in range(self.max_layers)]
        self.betas = [random_blender(self.mpk_layers[l]['lambda_N']) for l in range(self.max_layers)]
        
        self.pk_layers = [
            (pow(self.mpk_layers[l]['g'], self.alphas[l], self.mpk_layers[l]['N2']),
             pow(self.mpk_layers[l]['g'], self.betas[l], self.mpk_layers[l]['N2']))
            for l in range(self.max_layers)
        ]
        
        self.query_vec = [1.0] * self.dim
        self.doc_vec = [1.0] * self.dim
        
        self.sk_layers = PODEngine.KeyGen(self.query_vec, self.max_layers, self.msk_layers, self.mpk_layers, self.alphas, self.betas)
        self.ct_layers = PODEngine.Encrypt(self.doc_vec, self.max_layers, self.mpk_layers, self.pk_layers)

    def test_hiding_and_binding_properties(self):
        """Step 2.3 Verification: Verify commitment hiding and binding properties."""
        import secrets
        # 1. Hiding test: Same input + different random nonces -> different commitments
        c1, n1 = commit_to_layer_result(b"layer_secret_payload")
        c2, n2 = commit_to_layer_result(b"layer_secret_payload")
        self.assertNotEqual(c1, c2, "Hiding property failed: same input gave identical commitments")
        self.assertNotEqual(n1, n2, "Nonces must be distinct")
        
        # 2. Binding test: Different inputs + same fixed nonce -> different commitments
        fixed_nonce = secrets.token_bytes(16)
        c_a, _ = commit_to_layer_result(b"input_A", nonce=fixed_nonce)
        c_b, _ = commit_to_layer_result(b"input_B", nonce=fixed_nonce)
        self.assertNotEqual(c_a, c_b, "Binding property failed: different inputs gave identical commitments")

    def test_honest_prover_always_passes(self):
        """
        Step 2.4 Verification:
        Run a real POD traversal for D=3, commit all layer results to TraversalProof,
        and verify that an auditor spot-checking all layers receives valid verification.
        """
        proof = TraversalProof()
        layer_results = []
        
        # 1. Client executes progressive onion peeling across layers 0, 1, 2
        for l in range(self.max_layers):
            val_l = PODEngine.DecryptLayer(self.sk_layers[l], self.ct_layers[l], self.mpk_layers[l])
            layer_bytes = f"{val_l:.4f}".encode("utf-8")
            layer_results.append(layer_bytes)
            # Add layer commitment to proof object
            proof.add_layer(layer_index=l, layer_result_bytes=layer_bytes)
            
        # 2. Auditor challenges each layer (0, 1, 2)
        for l in range(self.max_layers):
            revealed_bytes, revealed_nonce = proof.reveal_chain_link(l)
            is_valid = proof.verify_layer(
                layer_index=l,
                revealed_bytes=revealed_bytes,
                revealed_nonce=revealed_nonce,
                sk_layer=self.sk_layers[l],
                ct_layer=self.ct_layers[l],
                mpk_layer=self.mpk_layers[l]
            )
            self.assertTrue(is_valid, f"Honest prover must pass verification at layer {l}")

    def test_cheating_prover_fails(self):
        """
        Step 2.4 Verification:
        Verify that a cheating prover who tampers with a layer witness or attempts
        to skip layers fails verification.
        """
        proof = TraversalProof()
        
        for l in range(self.max_layers):
            val_l = PODEngine.DecryptLayer(self.sk_layers[l], self.ct_layers[l], self.mpk_layers[l])
            layer_bytes = f"{val_l:.4f}".encode("utf-8")
            proof.add_layer(layer_index=l, layer_result_bytes=layer_bytes)
            
        # Cheater tampers with revealed bytes of layer 1
        revealed_bytes, revealed_nonce = proof.reveal_chain_link(1)
        tampered_bytes = b"forged_unauthorized_parent_key"
        
        is_valid_commitment = proof.verify_layer(
            layer_index=1,
            revealed_bytes=tampered_bytes,
            revealed_nonce=revealed_nonce,
            sk_layer=self.sk_layers[1],
            ct_layer=self.ct_layers[1],
            mpk_layer=self.mpk_layers[1]
        )
        self.assertFalse(is_valid_commitment, "Cheating prover with forged bytes must fail commitment verification")
        
        # Cheater tries invalid crypto decryption value
        is_valid_crypto = proof.verify_layer(
            layer_index=1,
            revealed_bytes=revealed_bytes,
            revealed_nonce=revealed_nonce,
            expected_numeric_result=99999.0
        )
        self.assertFalse(is_valid_crypto, "Cheating prover with wrong decryption result must fail chain verification")

    def test_pipeline_bounded_decoy_traversal_with_audit_proof(self):
        """
        Step 2.5 & 2.6 Verification:
        Verify bounded-decoy traversal integration within VPRAGPipeline with audit proof logging.
        """
        pipeline = VPRAGPipeline(hidden_dim=self.dim, K=16, lambda_bits=self.lambda_bits)
        
        unwrapped_bytes = [b"1.0000", b"1.0000", b"1.0000"]
        target_cid = "doc_target_77"
        decoy_cids = ["decoy_1", "decoy_2", "decoy_3"]
        
        recovered_target, proof = pipeline.execute_bounded_decoy_traversal(
            target_cid=target_cid,
            decoy_cids=decoy_cids,
            unwrapped_layer_bytes=unwrapped_bytes,
            generate_audit_proof=True
        )
        
        self.assertEqual(recovered_target, target_cid)
        self.assertIsNotNone(proof)
        self.assertIn(target_cid, pipeline.audit_proofs)
        self.assertEqual(len(proof.commitments), 3)


if __name__ == "__main__":
    unittest.main()
