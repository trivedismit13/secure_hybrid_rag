# SHIELD-RAG Architecture Components

This report details the three core cryptographic components of the SHIELD-RAG unified security framework, detailing how they enable confidential semantic search and multi-hop reasoning.

---

## 1. Core Components Infographic

Below is the conceptual illustration representing the three key mechanisms:

![SHIELD-RAG Key Components Infographic](output/shield_rag_components.jpg)

---

## 2. Component Explanations

1.  **Sensitivity-Embedded IPFE (SE-IPFE):**
    *   *Mechanism:* Embeds access clearance tokens directly inside encrypted node vector representations.
    *   *Result:* Only users with authorized credentials ($L_c \ge L_d$) can cancel out blenders to decrypt the inner product correctly. All other attempts return mathematical noise.
2.  **Query-Derived Cryptographic Scope (QDCS):**
    *   *Mechanism:* Uses query credentials to dynamically define a retrieval boundary in the vector space.
    *   *Result:* Out-of-scope database categories project onto the orthogonal complement of the query scope, forcing similarity scores to exactly `0.0`.
3.  **Progressive Onion Decryption (POD):**
    *   *Mechanism:* Nests vector embeddings under sequential layers of encryption, matching graph traversal hops.
    *   *Result:* Knowledge is revealed incrementally as the path traverses deeper, preventing shallow queries from unmasking deep database contents.
