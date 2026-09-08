# PRIOR-ART DISTINCTION MEMO: SHIELD-RAG
**Date:** September 2026  
**Subject:** Technical Differentiation of SHIELD-RAG Novelty Claims Against Prior Art  
**Target Audience:** Patent Attorney / Registered Patent Agent / Institutional IPR Cell  

---

## Executive Summary
This memorandum establishes the explicit, structural, and technical distinctions between the inventions claimed in the SHIELD-RAG system and the closest identified prior art across both academic literature and granted/pending patents.

The primary novelties are centered on two core claims:
1. **Lead Claim (Claim A):** Multi-Tenant Vector Retrieval Isolation via Constructed Orthogonal Subspaces (Mechanism 2).
2. **Secondary Claim (Claim B):** Dynamic Clearance Revocation via Reuse of an Existing Sensitivity-Embedded Noise Gate at Maximal Value Without Ciphertext Modification (Mechanism 1).

---

## 1. Differentiation for Lead Claim (Mechanism 2 — QR Subspace Multi-Tenant Isolation)

### 1.1 Closest Prior Art Identified
* **US Patent Application US20230409731A1** (*"Native multi-tenant encryption for database system"*, Snowflake Inc. / Allison et al., published Dec 21, 2023): Multi-tenancy achieved by assigning distinct symmetric encryption keys per tenant.
* **US Patent US11755767B2 / US20220067193A1** (*"Systems and methods of multi-key encryption for multi-tenant database"*, Salesforce, Inc. / Kuhlmann et al., granted Sep 12, 2023): API vault issuing per-tenant client key IDs.
* **US Patent Application US20160259807A1** (*"Secure isolation of tenant resources using a security gateway"*, General Electric Co. / Brandle et al., published Sep 8, 2016): Namespace-based metadata filtering and tenant-ID access control lists (ACLs).
* **Trans-RAG (DASFAA 2026)** (*"Trans-RAG: Query-Centric Vector Transformation for Secure Cross-Organizational Retrieval"*, Y. Liu et al., Proceedings of the 31st International Conference on Database Systems for Advanced Applications, Springer, 2026): Key-derived, per-organization vector-space transformations (random permutation, cryptographic blinding, bounded non-linearity, and orthogonal rotation).

### 1.2 Structural & Methodological Distinctions
1. **Mechanism Class (Geometric Orthogonality vs. Key/Metadata Separation):**
   * *Prior Art:* All identified database patents rely on *operational controls*—either encrypting records under separate keys or using software metadata filters. If a key is compromised or a metadata filter misconfigured, cross-tenant records become visible.
   * *Present Invention:* Isolation is enforced through **constructed linear-algebraic orthogonality**. The embedding space $\mathbb{R}^d$ is partitioned into disjoint orthonormal column blocks $\mathbf{U}_i, \mathbf{U}_j$ via QR factorization. By construction of orthonormal matrices, $\mathbf{U}_i^T \mathbf{U}_j = \mathbf{0}$. Therefore:
     $$\langle \mathbf{P}_i \mathbf{q}, \mathbf{P}_j \mathbf{x} \rangle \equiv 0.0$$
     This guarantees **exact zero similarity** ($< 10^{-15}$ machine precision) between different tenants, completely independent of key material, access policies, or software query filters.

2. **Information-Theoretic vs. Computational Isolation (vs. Trans-RAG):**
   * *Trans-RAG:* Uses secret transformation keys to rotate vectors. This is *computational* isolation—the transformation is mathematically invertible if the secret key is learned or approximated.
   * *Present Invention:* Isolation is *information-theoretic* by geometric projection. Once a query from Tenant $i$ is projected onto $\mathbf{U}_i$, its components in the subspace of Tenant $j$ are identically zero. No adversary possessing arbitrary computing power or tenant keys can recover cross-tenant dot products because the mathematical projection annihilates the cross-subspace components.

3. **Dynamic Tenant Management Without Full Recomputation:**
   * Unlike static subspace designs, the present invention includes:
     * *Reserved Dimension Pools:* Slicing new tenant bases from pre-factorized reserve columns in $0.0116$\,ms ($O(1)$ pointer update).
     * *Incremental Null-Space Projection (Modified Gram-Schmidt):* Generating dynamic orthogonal complements orthogonal to all active tenants in $155.17$\,ms without altering any existing tenant's projection matrix.
     * *Dynamic De-allocation:* De-allocating a tenant in $0.0096$\,ms without requiring re-indexing or re-encryption of remaining tenants.

4. **Dimensionality-Recall Scaling Rule & Asymmetric Allocation (Preempting Enablement Objections):**
   * A potential attack from patent examiners is that for large $T$, per-tenant dimensionality $k_t = d/T$ collapses, degrading semantic retrieval.
   * *Present Invention Defense:* The claims explicitly incorporate:
     * **Dimension Scaling Condition ($d / T \ge k_{\text{threshold}}$):** Bounds the linear-algebraic compression ratio using an empirically calibrated threshold (e.g., $k_{\text{threshold}} \ge 48$) to prevent the severe semantic collapse observed at $k \le 24$.
     * **Asymmetric Subspace Allocation (Empirically Validated in Table 2.5B):** Allocates non-uniform column widths $\{k_t\}$ based on tenant corpus volume and recall requirements. In an 8-tenant deployment under $d=384$, allocating $k=120$ to a primary tenant elevates Hit@10 from $0.62$ to **$0.94$** (+32%), while intermediate tenants achieve $0.71$ ($k=60$) and satellite tenants achieve $0.53$ ($k=28$), with cross-tenant isolation remaining mathematically exact ($\le 2.57 \times 10^{-16}$).

---

## 2. Differentiation for Secondary Claim (Mechanism 1 — Revocation via Noise-Gate Reuse)

### 2.1 Closest Prior Art Identified
* **C. Ge et al., IEEE TDSC 2024** (*"Attribute-Based Proxy Re-Encryption With Direct Revocation Mechanism for Data Sharing in Clouds"*, IEEE Transactions on Dependable and Secure Computing, vol. 21, no. 2, pp. 949–960, 2024): Attribute-based proxy re-encryption with direct revocation via re-randomization of proxy re-encryption keys.
* **Ciphertext-Updatable Functional Encryption (CUFE)** (*"Ciphertext-Updatable Inner-Product Functional Encryption"*, J. Kim et al., Cryptology and Information Security Series / Academic FE Literature): Academic IPFE constructions allowing ciphertexts to be updated to revoke access without full re-encryption.
* **General CP-ABE Revocation Schemes:** Revocation achieved by distributing new private keys or maintaining ciphertext revocation trees.

### 2.2 Structural & Methodological Distinctions
1. **Reuse of Existing Access-Gate Formula vs. Distinct Revocation Primitive:**
   * *Prior Art (ABE-PRE / CUFE):* Introduces a separate, heavy cryptographic mechanism—either generating and distributing re-encryption keys ($rk_{A \to B}$), maintaining ciphertext versioning, or mutating stored ciphertexts on disk.
   * *Present Invention:* Revocation introduces **zero new cryptographic primitives**. It repurposes the exact same algebraic noise formula already used for normal clearance gating:
     $$\Delta = r \cdot \max(0, L_d - L_c) \bmod \lambda(N)$$
     When a user is revoked ($u \in \mathcal{R}_{\text{revoked}}$), the proxy sets effective clearance $L_c = 0$, deterministically injecting maximal algebraic noise:
     $$\Delta_{\text{max}} = r \cdot (5 - 0) \bmod \lambda(N)$$
     This renders all ciphertexts ($L_d \ge 1$) completely unresolvable, achieving revocation by *pushing an existing continuous gate to its extreme boundary*.

2. **Zero Ciphertext Modification & Zero Corpus Re-Encryption:**
   * *Prior Art (CUFE):* Must mutate every stored ciphertext when user permissions change.
   * *Present Invention:* Intercepts subkeys at the proxy layer during query time. The stored ciphertext database is modified **zero times** (verified across 50,000 documents).

3. **Instantaneous Reinstatement:**
   * Reinstating a revoked user is equally $O(1)$ ($0.0075$\,ms), simply removing the identifier from the in-memory revoked set $\mathcal{R}_{\text{revoked}}$ without re-issuing keys to other users or re-encrypting data.

4. **Enterprise-Scale Efficiency:**
   * Tested up to $N=50,000$ revoked users: batch revocation executes in $33.69$\,ms ($0.67\,\mu$s/user) and per-query status lookup requires only $1.24\,\mu$s.

---

## 3. Note on Benchmark Timing Reconciliation across Documents
If examiners cross-reference the companion conference paper (ICAISDA 2026) and this patent disclosure:
* The conference paper reports **$55$--$59$\,ms** QR latency: This measures the isolated mathematical `np.linalg.qr` core decomposition on warm state.
* The patent document reports **$94$--$133$\,ms** end-to-end setup latency: This measures cold-start engine instantiation, which includes random matrix generation, QR decomposition, column block slicing, memory layout alignment, and dictionary metadata registration for all $T$ tenants. Both measurements are consistent and reflect distinct system boundaries.

---

## 4. Summary of Claim Vulnerabilities Preempted

| Anticipated Examiner Objection | Why It Fails Based on Our Evidence |
|:---|:---|
| *"Combining ABE proxy revocation with IPFE is an obvious combination of known elements."* | Revocation does not use ABE re-encryption keys; it achieves access termination by reusing the existing clearance noise gate at $\Delta_{\text{max}}$. This structural economy eliminates the re-encryption apparatus entirely. |
| *"Multi-tenant key isolation is already well-known in database systems."* | All prior art isolates tenants via distinct keys or software filters. Our system isolates tenants geometrically in vector space such that cross-tenant dot products are identically zero, even if keys are shared. |
| *"QR subspace isolation cannot handle dynamic tenant additions without recomputing the entire matrix."* | Disproven by empirical benchmark: dynamic pool admission executes in $0.0116$\,ms and incremental Gram-Schmidt admission in $155.17$\,ms, preserving cross-tenant orthogonality ($< 10^{-15}$). |
| *"For large tenant populations T, subspace dimension k_t collapses and destroys retrieval utility, rendering the broad claim non-enabled."* | Preempted by dependent Claims 2 and 3: the invention explicitly claims the dimension scaling rule ($d/T \ge k_{\text{threshold}}$) to prevent sub-24 collapse, and proves via Table 2.5B that asymmetric allocation recovers 0.94 Hit@10 for core enterprise tenants under fixed dimensions. |