"""
Unit tests for Lattice-Based Post-Quantum Functional Projection Engine (L-QDCS).
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lattice_engine import PolynomialRing, LatticeQDCSEngine


class TestLatticeQDCSEngine(unittest.TestCase):

    def test_polynomial_ring_arithmetic(self):
        ring = PolynomialRing(n=8, q=97)
        p1 = np.array([1, 2, 3, 4, 0, 0, 0, 0], dtype=np.int64)
        p2 = np.array([2, 0, 1, 0, 0, 0, 0, 0], dtype=np.int64)
        
        p_add = ring.add(p1, p2)
        self.assertEqual(p_add[0], 3)
        self.assertEqual(p_add[2], 4)
        
        p_sub = ring.subtract(p1, p2)
        self.assertEqual(p_sub[0], (1 - 2) % 97)
        self.assertEqual(p_sub[2], 2)
        
        p_mul = ring.multiply(p1, p2)
        self.assertEqual(len(p_mul), 8)

    def test_lattice_qdcs_matching_and_scope_filtering(self):
        dim = 64
        engine = LatticeQDCSEngine(dimension=dim, ring_n=256, ring_q=8380417)
        pk, sk = engine.keygen()
        
        # 16-dimensional category subspace S_q
        basis_vectors = []
        for i in range(16):
            b = np.zeros(dim)
            b[i] = 1.0
            basis_vectors.append(b)
        P_S = engine.generate_subspace_projection(basis_vectors)
        
        # Query generated within the authorized category subspace S_q
        raw_query = np.zeros(dim)
        raw_query[:16] = np.random.randn(16)
        target_vec = raw_query / np.linalg.norm(raw_query)
        
        # Similar document inside authorized scope
        similar_vec = target_vec + np.random.normal(0, 0.02, size=dim)
        similar_vec = similar_vec / np.linalg.norm(similar_vec)
        
        # Dissimilar document inside authorized scope
        dissimilar_vec = -target_vec
        
        # Document strictly in an OUT-OF-SCOPE domain (orthogonal to S_q)
        out_of_scope_raw = np.zeros(dim)
        out_of_scope_raw[16:] = np.random.randn(48)
        out_of_scope_vec = out_of_scope_raw / np.linalg.norm(out_of_scope_raw)
        
        # Encrypt documents with scope projection
        ct_similar = engine.encrypt_document(similar_vec, pk, projection_matrix=P_S)
        ct_dissimilar = engine.encrypt_document(dissimilar_vec, pk, projection_matrix=P_S)
        ct_out_of_scope = engine.encrypt_document(out_of_scope_vec, pk, projection_matrix=P_S)
        
        trapdoor = engine.generate_query_trapdoor(target_vec, sk)
        
        score_sim = engine.evaluate_similarity(ct_similar, trapdoor)
        score_dissim = engine.evaluate_similarity(ct_dissimilar, trapdoor)
        score_out_scope = engine.evaluate_similarity(ct_out_of_scope, trapdoor)
        
        self.assertGreater(score_sim, 0.85, "Similar in-scope document should have high positive cosine similarity")
        self.assertLess(score_dissim, -0.85, "Dissimilar in-scope document should have negative similarity")
        self.assertLess(abs(score_out_scope), 0.05, "Out-of-scope document should be suppressed to ~0.0")
        print(f"\n[L-QDCS Test Passed] Similar: {score_sim:.4f} | Dissimilar: {score_dissim:.4f} | Out-of-Scope: {score_out_scope:.4f}")


if __name__ == "__main__":
    unittest.main()
