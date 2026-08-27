import random
import numpy as np
from typing import List, Tuple, Dict, Any, Union
from crypto_engine import AdaIPFEEngine

class QDCSEngine:
    @staticmethod
    def Setup(lambda_bits: int, n: int) -> Tuple[Dict[str, Any], List[int]]:
        """Setup matches standard Ada-IPFE parameters."""
        return AdaIPFEEngine.Setup(lambda_bits, n)

    @staticmethod
    def compute_projection_matrix(U: np.ndarray) -> np.ndarray:
        """
        Builds the orthogonal projection matrix P_S = U * U^T from
        an orthonormal basis matrix U (dimension d x k).
        """
        return U @ U.T

    @staticmethod
    def project_vector(x: Union[List[float], np.ndarray], U: np.ndarray) -> np.ndarray:
        """
        Projects vector x into the subspace spanned by orthonormal basis U:
        P_S(x) = U * (U^T * x)
        """
        x_vec = np.asarray(x, dtype=np.float64)
        return U @ (U.T @ x_vec)

    @staticmethod
    def KeyGen(y: List[float], allowed_domains: List[str], msk: List[int], mpk: Dict[str, Any], alpha: int, beta: int) -> Dict[str, Any]:
        """Generates query subkey associated with vector y and a list of authorized domains."""
        from rag_pipeline import keygen_with_blenders
        sk_hq = keygen_with_blenders(y, msk, mpk, alpha, beta)
        
        return {
            "sk_hq": sk_hq,
            "allowed_domains": allowed_domains
        }

    @staticmethod
    def Encrypt(x: List[float], domain: str, mpk: Dict[str, Any], pk: List[int]) -> Dict[str, Any]:
        """Encrypts vector x and attaches the document's domain/category."""
        ct = AdaIPFEEngine.Encrypt(x, mpk, pk)
        
        return {
            "ct": ct,
            "domain": domain
        }

    @staticmethod
    def Decrypt(sk_y: Dict[str, Any], ct_x: Dict[str, Any], mpk: Dict[str, Any]) -> float:
        """
        Decrypts the inner product. 
        Enforces domain boundaries: if document domain is outside query scope,
        the vector similarity is forced to 0 (simulating vector subspace orthogonality).
        """
        allowed_domains = sk_y["allowed_domains"]
        domain = ct_x["domain"]
        ct = ct_x["ct"]
        sk_hq = sk_y["sk_hq"]
        
        if domain in allowed_domains:
            # Document is in scope: return normal decrypted inner product
            return AdaIPFEEngine.Decrypt(sk_hq, ct, mpk)
        else:
            # Out of scope: force dot product to 0 (orthogonal projection)
            return 0.0
