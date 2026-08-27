# =========================================================================
# Step 3.1 Verification - Exact Function Signatures in qdcs_engine.py
# =========================================================================
# Output of grep -n "def " qdcs_engine.py:
# 8:  def Setup(lambda_bits: int, n: int) -> Tuple[Dict[str, Any], List[int]]:
# 13: def compute_projection_matrix(U: np.ndarray) -> np.ndarray:
# 21: def project_vector(x: Union[List[float], np.ndarray], U: np.ndarray) -> np.ndarray:
# 30: def KeyGen(y: List[float], allowed_domains: List[str], msk: List[int], mpk: Dict[str, Any], alpha: int, beta: int) -> Dict[str, Any]:
# 41: def Encrypt(x: List[float], domain: str, mpk: Dict[str, Any], pk: List[int]) -> Dict[str, Any]:
# 51: def Decrypt(sk_y: Dict[str, Any], ct_x: Dict[str, Any], mpk: Dict[str, Any]) -> float:
# =========================================================================

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from qdcs_engine import QDCSEngine


def build_tenant_subspaces(
    embedding_dim: int,
    num_tenants: int,
    subspace_dim_per_tenant: int
) -> Dict[str, np.ndarray]:
    """
    Splits the full embedding_dim-dimensional space into num_tenants
    mutually orthogonal subspaces, each of subspace_dim_per_tenant dimensions.

    Requires: num_tenants * subspace_dim_per_tenant <= embedding_dim.
    If this is violated, raises a ValueError immediately.

    Method: generates one random embedding_dim x embedding_dim orthogonal
    matrix Q via QR decomposition of a Gaussian random matrix.
    Assigns each tenant a contiguous, non-overlapping block of columns of Q
    as their orthonormal basis U_tenant.
    """
    total_required_dim = num_tenants * subspace_dim_per_tenant
    if total_required_dim > embedding_dim:
        raise ValueError(
            f"Cannot fit {num_tenants} tenants x {subspace_dim_per_tenant} dims "
            f"into {embedding_dim}-dim space without overlap."
        )

    # Generate random Gaussian matrix and perform QR factorization
    random_matrix = np.random.randn(embedding_dim, embedding_dim)
    Q, _ = np.linalg.qr(random_matrix)  # Q's columns are orthonormal

    tenant_bases: Dict[str, np.ndarray] = {}
    for t in range(num_tenants):
        start = t * subspace_dim_per_tenant
        end = start + subspace_dim_per_tenant
        tenant_bases[f"tenant_{t}"] = Q[:, start:end]

    return tenant_bases


def project_document_to_tenant_subspace(
    doc_embedding: np.ndarray,
    tenant_basis_U: np.ndarray,
    existing_projection_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = QDCSEngine.project_vector
) -> np.ndarray:
    """
    Projects a document embedding into a tenant's specific orthonormal subspace
    using QDCSEngine's projection primitive P_S(x) = U * (U^T * x).
    """
    return existing_projection_fn(doc_embedding, tenant_basis_U)


class MultiTenantQDCSEngine:
    """
    Multi-Tenant Query-Derived Cryptographic Scope (MT-QDCS) Engine.
    Guarantees mathematical non-interference across tenant boundaries by
    mapping distinct tenants to mutually orthogonal subspace bases:
    For any i != j, U_i^T @ U_j = 0, ensuring <P_i(q), P_j(d)> == 0.0 identically.
    """
    def __init__(self, embedding_dim: int = 384, num_tenants: int = 2, subspace_dim_per_tenant: int = 190):
        self.dim = embedding_dim
        self.num_tenants = num_tenants
        self.subspace_dim = subspace_dim_per_tenant
        self.bases = build_tenant_subspaces(embedding_dim, num_tenants, subspace_dim_per_tenant)

    def get_tenant_basis(self, tenant_id: str) -> np.ndarray:
        if tenant_id not in self.bases:
            raise KeyError(f"Unknown tenant identifier: {tenant_id}")
        return self.bases[tenant_id]

    def project_query(self, query_vec: np.ndarray, tenant_id: str) -> np.ndarray:
        U = self.get_tenant_basis(tenant_id)
        return project_document_to_tenant_subspace(query_vec, U)

    def project_doc(self, doc_vec: np.ndarray, tenant_id: str) -> np.ndarray:
        U = self.get_tenant_basis(tenant_id)
        return project_document_to_tenant_subspace(doc_vec, U)
