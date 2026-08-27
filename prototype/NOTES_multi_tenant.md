# Multi-Tenant Subspace Non-Interference Guarantee for QDCS

## Theorem (Multi-Tenant Orthogonal Isolation by Construction)
For any two distinct tenants $i \neq j$ allocated disjoint orthonormal basis matrices $U_i \in \mathbb{R}^{d \times k_i}$ and $U_j \in \mathbb{R}^{d \times k_j}$ constructed via QR decomposition of a shared ambient space $\mathbb{R}^d$ (where $\sum_t k_t \le d$), any query vector $q \in \mathbb{R}^d$ projected into tenant $i$'s subspace $S_{q,i} = \text{span}(U_i)$ has an inner product of **identically zero** with any document vector $x \in \mathbb{R}^d$ projected into tenant $j$'s subspace $S_{d,j} = \text{span}(U_j)$, **by construction**, independent of corpus size, document text, or query content:

$$\langle P_i(q), P_j(x) \rangle = \left( U_i U_i^T q \right)^T \left( U_j U_j^T x \right) = q^T U_i \left( U_i^T U_j \right) U_j^T x = q^T U_i \mathbf{0}_{k_i \times k_j} U_j^T x = 0$$

Because $U_i^T U_j = \mathbf{0}$ holds unconditionally across the entire index, cross-tenant leakage is mathematically impossible, elevating Query-Derived Cryptographic Scope (QDCS) from an empirical access filter into a provable multi-tenant isolation primitive.
