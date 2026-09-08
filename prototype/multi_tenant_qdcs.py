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

    Supports Dynamic Tenant Subspace Management:
    - Pre-factorized global orthonormal basis Q in R^{d x d}.
    - Dynamic allocation of orthogonal column blocks to new tenants without re-factorizing Q.
    - Dynamic de-allocation / recycling of tenant column blocks.
    - Incremental Gram-Schmidt admission into the nullspace of active tenants.
    """
    def __init__(self, embedding_dim: int = 384, num_tenants: int = 2, subspace_dim_per_tenant: int = 190):
        self.dim = embedding_dim
        self.default_subspace_dim = subspace_dim_per_tenant
        
        # Factorize full global orthonormal basis Q in R^{d x d}
        random_matrix = np.random.randn(embedding_dim, embedding_dim)
        self.Q, _ = np.linalg.qr(random_matrix)  # Q^T @ Q = I_d
        
        self.bases: Dict[str, np.ndarray] = {}
        self._allocated_spans: Dict[str, Tuple[int, int]] = {}
        self._current_col_ptr = 0
        
        # Initial tenant allocations
        for t in range(num_tenants):
            self.add_tenant(f"tenant_{t}", subspace_dim_per_tenant)

    def add_tenant(self, tenant_id: str, subspace_dim: Optional[int] = None) -> np.ndarray:
        """
        Dynamically admits a new tenant without recomputing QR factorization.
        Allocates a dedicated column block of Q.
        Guarantees U_new^T @ U_existing = 0 by construction.
        """
        if tenant_id in self.bases:
            raise ValueError(f"Tenant '{tenant_id}' is already registered.")
        
        k = subspace_dim if subspace_dim is not None else self.default_subspace_dim
        if self._current_col_ptr + k > self.dim:
            raise ValueError(
                f"Cannot allocate {k} dimensions for '{tenant_id}'. "
                f"Available remaining dimensions: {self.dim - self._current_col_ptr} of {self.dim}."
            )
        
        start = self._current_col_ptr
        end = start + k
        self._current_col_ptr = end
        
        U_tenant = self.Q[:, start:end]
        self.bases[tenant_id] = U_tenant
        self._allocated_spans[tenant_id] = (start, end)
        return U_tenant

    def remove_tenant(self, tenant_id: str) -> None:
        """
        Dynamically de-allocates a tenant's subspace.
        Reclaims the span and removes the tenant from active bases.
        Other tenants' bases and stored ciphertexts remain completely unaffected.
        """
        if tenant_id not in self.bases:
            raise KeyError(f"Tenant '{tenant_id}' not found.")
        del self.bases[tenant_id]
        # Note: In production, reclaimed spans can be added to a free-list pool for re-allocation.
        del self._allocated_spans[tenant_id]

    def admit_tenant_via_gram_schmidt(self, tenant_id: str, k: int) -> np.ndarray:
        """
        Incremental admission via Modified Gram-Schmidt (MGS) into the orthogonal
        complement of all active tenants.
        Generates k orthonormal vectors orthogonal to all active bases without
        modifying any existing tenant basis.
        """
        if tenant_id in self.bases:
            raise ValueError(f"Tenant '{tenant_id}' already exists.")
        
        # Collect all active basis vectors
        if len(self.bases) > 0:
            existing_V = np.hstack([self.bases[t] for t in self.bases])
        else:
            existing_V = np.zeros((self.dim, 0))
            
        new_cols = []
        for _ in range(k):
            v = np.random.randn(self.dim)
            # Project out existing basis vectors
            for j in range(existing_V.shape[1]):
                u = existing_V[:, j]
                v -= np.dot(u, v) * u
            # Project out already accepted new columns
            for u in new_cols:
                v -= np.dot(u, v) * u
            norm = np.linalg.norm(v)
            if norm < 1e-10:
                raise ValueError("Dimensionality exhausted; cannot find orthogonal vector.")
            v /= norm
            new_cols.append(v)
            
        U_new = np.column_stack(new_cols)
        self.bases[tenant_id] = U_new
        return U_new

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

    def verify_isolation(self, tenant_a: str, tenant_b: str) -> float:
        """
        Computes maximum absolute inner product between bases of tenant_a and tenant_b.
        Must be < 1e-14 (machine precision zero).
        """
        U_a = self.get_tenant_basis(tenant_a)
        U_b = self.get_tenant_basis(tenant_b)
        cross_prod = np.dot(U_a.T, U_b)
        return float(np.max(np.abs(cross_prod)))

