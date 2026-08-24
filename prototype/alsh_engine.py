import numpy as np
from typing import Union
from config import ALSH_K, ALSH_M, ALSH_U

class ALSHEngine:
    def __init__(self, d: int, K: int = ALSH_K, m: int = ALSH_M, U: float = ALSH_U, seed: int = 42):
        """
        Asymmetric Locality-Sensitive Hashing (ALSH) Engine.
        d: original dimension of vectors.
        K: number of projection hyperplanes (signature length).
        m: power exponent (number of appended dimensions).
        U: scaling upper bound for L2 norm.
        """
        self.d = d
        self.K = K
        self.m = m
        self.U = U
        
        # Sample projection matrix R from normal distribution N(0, I)
        # Shape: K x (d + m)
        np.random.seed(seed)
        self.R = np.random.normal(0.0, 1.0, (self.K, self.d + self.m))

    def p_transform(self, v: np.ndarray) -> np.ndarray:
        """
        P-Transformation for knowledge vectors v.
        If L2-norm > U, rescale vector v.
        Appends ||v||_2^2, ||v||_2^4, ..., ||v||_2^{2^m} to v.
        Supports both 1D (single vector) and 2D (batch of vectors) inputs.
        """
        if v.ndim == 1:
            norm = np.linalg.norm(v)
            v_rescaled = v
            if norm > self.U:
                v_rescaled = v * (self.U / (norm + 1e-9))
                norm = self.U
            
            # Compute norm powers: ||v||^2, ||v||^4, ..., ||v||^{2^m}
            appended = [norm ** (2 ** i) for i in range(1, self.m + 1)]
            return np.concatenate([v_rescaled, appended])
        else:
            # Batch mode
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.U / (norms + 1e-9))
            v_rescaled = v * scale
            norms_rescaled = norms * scale
            
            appended = []
            for i in range(1, self.m + 1):
                appended.append(norms_rescaled ** (2 ** i))
            
            return np.hstack([v_rescaled] + appended)

    def q_transform(self, q: np.ndarray) -> np.ndarray:
        """
        Q-Transformation for query vector q.
        Appends 1/2 to q, m times.
        Supports both 1D and 2D inputs.
        """
        if q.ndim == 1:
            # Append [0.5, 0.5, ..., 0.5] (m times)
            appended = [0.5] * self.m
            return np.concatenate([q, appended])
        else:
            # Batch mode
            n_samples = q.shape[0]
            appended = np.full((n_samples, self.m), 0.5)
            return np.hstack([q, appended])

    def hash_vector(self, transformed_vec: np.ndarray) -> np.ndarray:
        """
        Projects transformed vector onto the random hyperplanes.
        H(x) = sign(R * x) where sign returns +1 for >= 0, and -1 for < 0.
        """
        # R shape: K x (d + m)
        # transformed_vec: (d + m) or Batch x (d + m)
        if transformed_vec.ndim == 1:
            projection = np.dot(self.R, transformed_vec)
            # return signature in {-1, 1}^K
            return np.where(projection >= 0.0, 1, -1)
        else:
            # Batch mode: projection is Batch x K
            projection = np.dot(transformed_vec, self.R.T)
            return np.where(projection >= 0.0, 1, -1)

    def compute_similarity(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """
        Computes similarity between two signature hashes.
        Since elements are in {-1, 1}, the dot product scaled by 1/K matches LSH collision logic.
        """
        return float(np.dot(h1, h2) / self.K)
