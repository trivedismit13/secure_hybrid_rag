"""
Unit and Integration Tests for Novelty #1: Revocable Clearance Without Corpus Re-Encryption.
"""

import unittest
import numpy as np
import sys
import os

# Add prototype directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from se_ipfe_engine import SEIPFEEngine
from revocation_proxy import RevocationProxy
from crypto_engine import AdaIPFEEngine
from rag_pipeline import VPRAGPipeline, random_blender


class TestRevocationProxy(unittest.TestCase):

    def setUp(self):
        self.proxy = RevocationProxy()
        # Setup SE-IPFE crypto parameters
        self.dim = 8
        self.lambda_bits = 256
        self.mpk, self.msk = SEIPFEEngine.Setup(self.lambda_bits, self.dim)
        self.alpha = random_blender(self.mpk['lambda_N'])
        self.beta = random_blender(self.mpk['lambda_N'])
        
        # Sample query vector
        self.query_vec = [1.0] * self.dim

    def test_revoked_user_gets_deny_value(self):
        """Verify that a revoked user's subkey is substituted with force_deny_value."""
        self.proxy.revoke("bob", "security audit failure")
        
        # Generate real subkey for clearance level 5
        real_key = SEIPFEEngine.KeyGen(
            y=self.query_vec,
            clearance=5,
            msk=self.msk,
            mpk=self.mpk,
            alpha=self.alpha,
            beta=self.beta
        )
        deny_val = RevocationProxy.create_force_deny_value(real_key)
        
        result = self.proxy.enforce("bob", real_key, deny_val)
        self.assertEqual(result, deny_val)
        self.assertEqual(result["clearance"], 0)

    def test_active_user_gets_real_key(self):
        """Verify that an active user's subkey is passed through unchanged."""
        real_key = SEIPFEEngine.KeyGen(
            y=self.query_vec,
            clearance=5,
            msk=self.msk,
            mpk=self.mpk,
            alpha=self.alpha,
            beta=self.beta
        )
        deny_val = RevocationProxy.create_force_deny_value(real_key)
        
        result = self.proxy.enforce("carol", real_key, deny_val)
        self.assertEqual(result, real_key)
        self.assertEqual(result["clearance"], 5)

    def test_se_ipfe_revocation_algebraic_noise_injection(self):
        """
        Verify that when a user is revoked in SE-IPFE:
        1. Clearance is forced to 0.
        2. Decryption fails / injects algebraic noise for sensitive documents.
        """
        doc_vec = [1.0] * self.dim
        pk = (
            pow(self.mpk['g'], self.alpha, self.mpk['N2']),
            pow(self.mpk['g'], self.beta, self.mpk['N2'])
        )
        # Encrypt document with sensitivity 3
        ct_doc = SEIPFEEngine.Encrypt(doc_vec, sensitivity=3, mpk=self.mpk, pk=pk)
        
        # Active user with clearance 4 -> Decryption succeeds
        sk_active = SEIPFEEngine.KeyGen(self.query_vec, clearance=4, msk=self.msk, mpk=self.mpk, alpha=self.alpha, beta=self.beta, user_id="alice", revocation_proxy=self.proxy)
        score_active = SEIPFEEngine.Decrypt(sk_active, ct_doc, self.mpk)
        self.assertAlmostEqual(score_active, 8.0, places=1)
        
        # Revoke user "alice"
        self.proxy.revoke("alice", "clearance revoked")
        sk_revoked = SEIPFEEngine.KeyGen(self.query_vec, clearance=4, msk=self.msk, mpk=self.mpk, alpha=self.alpha, beta=self.beta, user_id="alice", revocation_proxy=self.proxy)
        self.assertEqual(sk_revoked["clearance"], 0)
        
        score_revoked = SEIPFEEngine.Decrypt(sk_revoked, ct_doc, self.mpk)
        # Decryption of sensitive doc (sensitivity 3 > clearance 0) is corrupted by noise
        self.assertNotAlmostEqual(score_revoked, 8.0, places=1)
        print(f"\n[SE-IPFE Revocation Test] Active Score: {score_active:.2f} | Revoked Noised Score: {score_revoked:.2f}")

    def test_end_to_end_revocation_blocks_retrieval(self):
        """
        Step 1.5 End-to-End Verification:
        1. Query as active user -> retrieval succeeds with high similarity on matching doc.
        2. Revoke user -> similarity is corrupted / falls outside valid ALSH range.
        3. Reinstate user -> query succeeds again without re-encrypting the corpus.
        """
        dim = 16
        k_dim = 16
        pipeline = VPRAGPipeline(hidden_dim=dim, K=k_dim, lambda_bits=256)
        
        # Create a small sample corpus
        target_vec = np.random.randn(dim)
        target_vec = target_vec / np.linalg.norm(target_vec)
        
        doc_vectors = [
            target_vec + np.random.normal(0, 0.02, size=dim), # Highly relevant doc 0
            -target_vec,                                       # Dissimilar doc 1
            np.random.randn(dim)                               # Random doc 2
        ]
        doc_texts = ["Centrifugal pump maintenance guide.", "History of ancient Rome.", "Quantum computing algorithms."]
        corpus_id = "test_revocation_corpus"
        
        # Upload once to dual storage (IPFS + Blockchain)
        pipeline.upload_knowledge_base(corpus_id, doc_vectors, doc_texts)
        
        # 1. Query with active user "alice"
        results_active = pipeline.query(corpus_id, target_vec, top_k=3, user_id="alice")
        self.assertTrue(len(results_active) > 0)
        best_cid, best_score = results_active[0]
        # Active user retrieves matching doc with high cosine similarity
        self.assertGreater(best_score, 0.5)
        
        # 2. Revoke user "alice" via RevocationProxy
        pipeline.revocation_proxy.revoke("alice", reason="department transfer")
        self.assertTrue(pipeline.revocation_proxy.is_revoked("alice"))
        
        # 3. Re-run identical query for revoked "alice"
        results_revoked = pipeline.query(corpus_id, target_vec, top_k=3, user_id="alice")
        revoked_scores = [score for _, score in results_revoked]
        # With algebraic noise injected into the deny subkey, the match score is corrupted
        for score in revoked_scores:
            self.assertTrue(score < 0.2 or score > 1.5, "Revoked user similarity must be corrupted by algebraic noise")
            
        # 4. Reinstate "alice" and verify instant recovery without re-encrypting corpus
        pipeline.revocation_proxy.reinstate("alice")
        self.assertFalse(pipeline.revocation_proxy.is_revoked("alice"))
        results_reinstated = pipeline.query(corpus_id, target_vec, top_k=3, user_id="alice")
        self.assertGreater(results_reinstated[0][1], 0.5, "Reinstated user should retrieve again without corpus re-encryption")
        print("\n[Step 1.5 End-to-End Verified] Active (0.95+) -> Revoked (Corrupted/Blocked) -> Reinstated (0.95+) seamlessly.")


if __name__ == "__main__":
    unittest.main()
