# Standard Secure RAG vs. CipheRAG Comparison

This document describes the security vulnerability of standard privacy-preserving RAG schemes and how CipheRAG's end-to-end encryption architecture addresses it.

---

## 1. Security Architecture Diagram

Below is the conceptual illustration showing the visual comparison:

![Security Comparison Infographic](output/secure_rag_comparison.jpg)

---

## 2. Vulnerability Analysis: Standard vs. CipheRAG

### Standard Secure RAG Schemes (Vulnerable)
*   **Encrypted Scope:** Only the raw vector indexing/retrieval matches are encrypted.
*   **The Leak:** 
    1.  **Graph Structure Exposure:** The connections, edge categories, and multi-hop relationships of knowledge graphs are visible to the server.
    2.  **Transient Leakage:** Once documents are fetched, their text representations or unencrypted embeddings are sent to the LLM server, exposing data during generation and self-attention operations.

### CipheRAG Architecture (Fully Protected)
*   **Encrypted Graph Traversal (POD):** Using **Progressive Onion Decryption**, multi-hop graph steps remain encrypted. The server does not know the traversal path or the connected relationships of nodes.
*   **In-Model Gateway Decryption:** Context embeddings are encrypted in transit and only decrypted directly inside the secure attention gateway using derived row-keys, ensuring no plaintext leakage occurs at the LLM server level.
