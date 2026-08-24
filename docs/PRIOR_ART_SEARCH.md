# Prior-Art Search Log & Novelty Analysis

**Date:** July 20, 2026

## 1. Search Queries Executed
- "CipheRAG Ada-IPFE Adaptive Inner-Product Functional Encryption DCR assumption Paillier construction Setup KeyGen Encrypt Decrypt algorithms"
- "CipheRAG encrypted similarity search LLM attention layer decryption performance numbers"
- "IC-HRAG Intent-Constrained Hierarchical RAG semantic label design technical indicator true false evaluation F1 scores knowledge integration format"
- "EduRAG-Compose sub-query generation CoAG evaluation metrics"
- "Roumeliotis trust calibration ECE OCR Consistency Gap metric definitions consensus failure mode LLM"

## 2. Findings Summary
1. **Ada-IPFE (CipheRAG):** Current state-of-the-art uses Inner-Product Functional Encryption (IPFE) based on the Decisional Composite Residuosity (DCR) assumption to securely evaluate inner products (similarities) between a query vector and encrypted knowledge base vectors. CipheRAG reports up to 35x faster generation and 15x faster QKV computation compared to Fully Homomorphic Encryption (FHE) baselines.
2. **IC-HRAG:** Introduces intent-constrained logic, pushing user queries into structured ontology spaces. However, the retrieval and mapping are performed in plaintext.
3. **EduRAG-Compose:** Explores Chain-of-Augmented-Generation (CoAG) for recursive sub-query creation to handle multi-hop logic, but again, entirely in plaintext.
4. **Trust Calibration (Roumeliotis et al.):** Proposes Expected Calibration Error (ECE), Consistency Gap, and Overconfidence Ratio (OCR) to measure the reliability of LLM generation. Identifying the "consensus failure mode" (where models hallucinate with high confidence) is a recognized problem.

## 3. Novelty Delta (Patentable Inventions)
While the individual components (IPFE, Intent Constraints, Trust Calibration) exist in prior art, the **SHIELD-RAG architecture introduces two distinct, patentable novelties**:

1. **Ontology-Constrained Bounded-Decoy Oblivious Traversal:** 
   Unlike prior art which relies on computationally expensive ORAM for hiding access patterns, SHIELD-RAG uses the graph's ontology to execute $K$-anonymous graph traversals. By blindly requesting semantic clusters (using `TypeTagCipher`) and uniformly sampling decoys, the client achieves $O(K)$ bandwidth and $O(1)$ client complexity per hop without exposing the traversal path to the server.
   
2. **Structured In-Model Decryption via Attention Masking coupled with Boundary Re-verification:**
   While CipheRAG uses IPFE to avoid FHE in QKV layers, SHIELD-RAG uniquely implements an attention masking mechanism (`AttentionDecryptionHook`) that enforces structural access control using deterministic cryptographic capabilities (`RelationKeyManager`). Furthermore, SHIELD-RAG closes the loop by using Roumeliotis trust metrics to dynamically trigger oblivious re-traversals (Boundary Re-verification) when the generation layer enters a consensus failure mode. This closed-loop system over an encrypted graph is entirely novel.
