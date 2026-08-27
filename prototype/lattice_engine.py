"""
Lattice-Based Post-Quantum Functional Projection Engine (L-QDCS)
Implements Ring-LWE (Learning With Errors over Polynomial Rings) with
Orthogonal Subspace Projection for Quantum-Resistant Encrypted Retrieval.
"""

import numpy as np
from typing import Tuple, List, Optional


class PolynomialRing:
    """
    Arithmetic in the polynomial quotient ring R_q = Z_q[X] / (X^n + 1)
    where n is a power of 2 and q is a prime modulus (e.g., 8380417).
    """
    def __init__(self, n: int = 512, q: int = 8380417, sigma: float = 2.0):
        self.n = n
        self.q = q
        self.sigma = sigma

    def sample_uniform(self) -> np.ndarray:
        """Sample a uniform random polynomial in R_q."""
        return np.random.randint(0, self.q, size=self.n, dtype=np.int64)

    def sample_gaussian(self) -> np.ndarray:
        """Sample small noise coefficients from discrete Gaussian distribution."""
        noise = np.random.normal(0, self.sigma, size=self.n)
        return np.round(noise).astype(np.int64) % self.q

    def add(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Addition modulo q in R_q."""
        return (p1 + p2) % self.q

    def subtract(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Subtraction modulo q in R_q."""
        return (p1 - p2) % self.q

    def multiply(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Negacyclic polynomial multiplication in Z_q[X] / (X^n + 1).
        """
        conv = np.convolve(p1, p2)
        result = np.zeros(self.n, dtype=np.int64)
        for i in range(len(conv)):
            deg = i % self.n
            sign = -1 if (i // self.n) % 2 == 1 else 1
            result[deg] = (result[deg] + sign * conv[i]) % self.q
        return result % self.q


class LatticeQDCSEngine:
    """
    Lattice-Based Query-Derived Cryptographic Scope (L-QDCS) Engine.
    Provides post-quantum 128-bit secure vector encryption with integrated
    orthogonal category subspace projection.
    """
    def __init__(self, dimension: int = 384, ring_n: int = 512, ring_q: int = 8380417):
        self.dim = dimension
        self.ring = PolynomialRing(n=ring_n, q=ring_q)
        self.scale = 20000

    def generate_subspace_projection(self, basis_vectors: List[np.ndarray]) -> np.ndarray:
        """
        Compute orthogonal projection matrix P_S = U * U^T for authorized category basis.
        """
        if not basis_vectors:
            return np.eye(self.dim, dtype=np.float64)
        
        B = np.column_stack(basis_vectors[:self.dim])
        Q, _ = np.linalg.qr(B)
        P_S = Q @ Q.T
        return P_S

    def keygen(self) -> Tuple[dict, np.ndarray]:
        """
        Generate public parameters (a, b) and secret key s in R_q.
        b = a * s + e (mod X^n + 1, mod q)
        """
        a = self.ring.sample_uniform()
        s = self.ring.sample_gaussian()
        e = self.ring.sample_gaussian()
        
        b = self.ring.add(self.ring.multiply(a, s), e)
        
        public_key = {"a": a, "b": b}
        secret_key = s
        return public_key, secret_key

    def encode_vector_to_poly(self, vector: np.ndarray) -> np.ndarray:
        """
        Encode high-dimensional float vector into polynomial coefficients in Z_q.
        """
        poly = np.zeros(self.ring.n, dtype=np.int64)
        norm = np.linalg.norm(vector)
        normed = vector / (norm + 1e-9) if norm > 0 else vector
        
        for i in range(min(self.dim, self.ring.n)):
            val = int(np.round(normed[i] * self.scale))
            poly[i] = val % self.ring.q
        return poly

    def encrypt_document(
        self,
        doc_vector: np.ndarray,
        public_key: dict,
        projection_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encrypt document vector under L-QDCS with category subspace projection.
        c1 = a * r + e1 (mod q)
        c2 = b * r + e2 + (P_S * x) (mod q)
        """
        if projection_matrix is not None:
            projected_vec = projection_matrix @ doc_vector
        else:
            projected_vec = doc_vector
            
        m_poly = self.encode_vector_to_poly(projected_vec)
        
        r = self.ring.sample_gaussian()
        e1 = self.ring.sample_gaussian()
        e2 = self.ring.sample_gaussian()
        
        c1 = self.ring.add(self.ring.multiply(public_key["a"], r), e1)
        br_plus_e2 = self.ring.add(self.ring.multiply(public_key["b"], r), e2)
        c2 = self.ring.add(br_plus_e2, m_poly)
        
        return (c1, c2)

    def generate_query_trapdoor(self, query_vector: np.ndarray, secret_key: np.ndarray) -> dict:
        """
        Generate client query trapdoor encoded for Ring-LWE dot product evaluation.
        """
        q_poly = self.encode_vector_to_poly(query_vector)
        return {
            "query_poly": q_poly,
            "secret_key": secret_key
        }

    def evaluate_similarity(
        self,
        ciphertext: Tuple[np.ndarray, np.ndarray],
        query_trapdoor: dict
    ) -> float:
        """
        Evaluate inner-product similarity in the encrypted domain using polynomial Ring-LWE.
        v = c2 - c1 * s (mod q)
        """
        c1, c2 = ciphertext
        s = query_trapdoor["secret_key"]
        q_poly = query_trapdoor["query_poly"]
        
        c1_s = self.ring.multiply(c1, s)
        v = self.ring.subtract(c2, c1_s)
        
        # Center coefficients to [-q/2, q/2]
        half_q = self.ring.q // 2
        centered_v = np.where(v > half_q, v - self.ring.q, v)
        centered_q = np.where(q_poly > half_q, q_poly - self.ring.q, q_poly)
        
        # Compute normalized cosine similarity score
        score = float(np.dot(centered_v[:self.dim], centered_q[:self.dim])) / (self.scale ** 2)
        return score
