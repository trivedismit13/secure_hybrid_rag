# Phase 2 Notes — Ada-IPFE Encryption Layer

**Timestamp:** 2026-07-20T15:25:00+05:30
**Phase:** 2 of 5  
**Status:** Complete  

---

## Design Decisions

### 1. Adaptive Inner-Product Functional Encryption (Ada-IPFE)
- **Construction:** Implemented the Decisional Composite Residuosity (DCR) based IPFE scheme over the Paillier group ($Z_{N^2}^*$).
- **Quantization:** Since IPFE operates on integer vectors, float embeddings are quantized using a $SCALE=1000$ factor before encryption. The final decrypted inner product is de-quantized by dividing by $SCALE^2$.
- **Implementation:** Built from scratch using `gmpy2` for efficient large integer arithmetic. A 1024-bit modulus $N$ is used for benchmarking.
- **Wire Format Serialization:** Ciphertexts are strictly serialized into binary (`[4B dim][len][val]...`) rather than JSON to save space and represent the actual wire layout.

### 2. PRF Token Generation
- **Component:** `PRFGenerator`
- **Method:** `HMAC-SHA256(key, session_salt || node_id)`
- **Unlinkability:** The `session_salt` is introduced to ensure cross-session unlinkability, fulfilling the privacy requirement that the server cannot correlate queries across different sessions.

### 3. Type Tag Cipher & Blind Index
- **Component:** `TypeTagCipher`
- **Method:** Deterministic AES-GCM (nonce derived from `hash(key || plaintext)`).
- **Blind Index:** The server uses `SHA256(type_tag_ct)` to group identical encrypted type tags into "type clusters". This provides Component C with a mechanism to fetch decoy pools of the exact same semantic type without revealing the type to the server.

### 4. Encrypted Graph Store & Migration
- **Component:** `EncryptedStore` and `GraphMigrator`
- **Structure:** The store holds `EncryptedBucket` objects indexed by PRF tokens. It maintains the blind type cluster index `_type_clusters[cluster_id] -> set[tokens]`.
- **Adjacency Lists:** Stored simply as a list of neighbor PRF tokens. The migration script securely translates all plaintext nodes into this format using the initialized cryptographic primitives.

---

## Baseline Benchmarks

**Hardware/Params:** Python 3.13 / gmpy2, 1024-bit RSA modulus, Dimension = 384 (all-MiniLM-L6-v2), 50 Iterations.

| Metric | Value | Notes |
|--------|-------|-------|
| Setup Time | 3401.20 ms | One-time cost (generates primes and $h_i$) |
| Mean KeyGen Time | 0.06 ms | Extremely fast client-side operation |
| Mean Encrypt Time | 3332.53 ms | Heavy (384 exponentiations), but done offline during corpus build |
| Mean Decrypt Time | 48.53 ms | Server-side IP evaluation, well within interactive limits |
| Ciphertext Size | 97.75 KB | Compact enough for wire transport |

These numbers prove the feasibility of the IPFE layer: although encryption is slow (done once offline), the critical path (keygen + server-side decryption) takes < 50 ms per node.

---

## Frozen Interfaces (DO NOT CHANGE after Phase 3)

1. `AdaIPFE`: `setup`, `keygen`, `encrypt`, `decrypt`, `serialize`, `deserialize`
2. `PRFGenerator`: `get_token(session_salt, node_id) -> bytes`
3. `TypeTagCipher`: `encrypt_type`, `decrypt_type`, `get_cluster_id`
4. `EncryptedStore`: `add_bucket`, `fetch`, `fetch_batch`, `get_type_cluster`

---

## Deviations from Spec

- Used AES-GCM with a synthetic deterministic nonce for the `TypeTagCipher` instead of AES-SIV or a second IPFE instance. This is simpler to implement and provides the required exact-match property for blind indexing without revealing the type.
