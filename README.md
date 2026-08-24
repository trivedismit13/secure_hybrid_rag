# SHIELD-RAG: Secure Hybrid Retrieval-Augmented Generation with Multi-Key Cryptography

This repository contains the complete end-to-end experimental prototype for **SHIELD-RAG** (built upon the **CipheRAG** framework, IEEE TDSC 2026). The project implements a privacy-preserving Retrieval-Augmented Generation pipeline operating entirely on encrypted vector embeddings, augmented with three novel access control engines: **SE-IPFE**, **QDCS**, and **POD**.

---

## 🚀 Key Architectural Features

SHIELD-RAG provides a unified framework addressing security vulnerabilities in standard secure RAG systems (which typically expose knowledge graph paths or post-retrieval self-attention states to the server):

1.  **Encrypted Semantic Search (Ada-IPFE):** High-dimensional S-BERT embeddings are hashed via Asymmetric Locality-Sensitive Hashing (ALSH) and encrypted using Adaptive Inner Product Functional Encryption (Ada-IPFE) with double key-blending.
2.  **Solidity Smart Contract Retrieval:** Query matching is performed as an inner-product check on-chain by an off-chain oracle, maintaining complete Zero-Knowledge leakage.
3.  **In-Model Gateway Decryption:** Context embeddings remain encrypted during transit and are decrypted directly inside the LLM's self-attention layers using row subkeys, avoiding transient plaintext exposures.
4.  **Sensitivity-Embedded IPFE (SE-IPFE):** Restricts functional decryptions based on user clearance clearance levels ($L_c \ge L_d$). Unauthorized decryption attempts inject algebraic noise to corrupt results.
5.  **Query-Derived Cryptographic Scope (QDCS):** restructures vector similarity by projecting out-of-scope document matches onto orthogonal complements, forcing similarity scores to exactly `0.0`.
6.  **Progressive Onion Decryption (POD):** Enforces graph traversal constraints by nesting vector embeddings under multi-layer encryption shells, peeling layers sequentially based on the graph traversal depth.

---

## 📂 Directory Structure

```text
secure_hybrid_rag/
├── config.py                 # Precision parameters and composite RSA bit sizes
├── crypto_engine.py          # Baseline Ada-IPFE mathematics and prime generator
├── se_ipfe_engine.py          # Sensitivity-Embedded IPFE access control gates
├── qdcs_engine.py            # Query-Derived Cryptographic Scope projections
├── pod_engine.py             # Progressive Onion Decryption layer peeling
├── alsh_engine.py            # Asymmetric Locality-Sensitive Hashing projections
├── ipfs_mock.py              # Mock storage for encrypted IPFS embeddings
├── rag_pipeline.py           # Core search (Algo 1) and gateway attention (Algo 2)
├── run_experiments.py        # Baseline robustness stress tests (contamination/drop)
├── run_comparisons.py        # Comparative latency and accuracy benchmarks
├── demo_presentation.py      # Console presentation interactive demo interface
├── requirements.txt          # Python library dependencies
├── blockchain/
│   ├── RetrievalContract.sol # Solidity smart contract managing audited queries
│   └── contract_helper.py    # Local EVM simulator interface
├── tests/
│   └── test_crypto.py        # Correctness unit tests for FE encryption
└── output/                   # Performance charts, diagrams, and raw metrics
    ├── benchmark_results.json
    ├── benchmark_results.png
    ├── comparison_results.json
    ├── comparison_results.png
    ├── secure_rag_comparison.jpg
    ├── shield_rag_components.jpg
    └── trl_readiness_basic.jpg
```

---

## 🛠️ Setup & Execution Guide

### 1. Installation
Install the required libraries (NumPy, PyTorch, Transformers, Matplotlib, SymPy, python-docx):
```bash
pip install -r requirements.txt
```

### 2. Verify Cryptographic Correctness (Unit Tests)
Validate the mathematical precision of the double key-blending Ada-IPFE implementation:
```bash
python -m unittest tests/test_crypto.py
```

### 3. Run Robustness & Query Drop Stress Tests
Evaluate the baseline RAG system against query drop rate losses and database contamination:
```bash
python run_experiments.py
```
*   Outputs: `output/benchmark_results.json` and `output/benchmark_results.png`.

### 4. Run Cryptographic Comparison Benchmarks
Compare all four implemented cryptographic backends (Ada-IPFE, SE-IPFE, QDCS, and POD) on the full corpus of 385 Wikipedia documents:
```bash
python run_comparisons.py
```
*   Outputs: `output/comparison_results.json` and `output/comparison_results.png`.

### 5. Start the Interactive Demo
Run a step-by-step interactive CLI demonstrating query encoding, blockchain matching, and attention gateway decryption:
```bash
python demo_presentation.py
```

---

## 📊 Performance Benchmarks & Comparisons

### Comparison 1: Ada-IPFE vs. Traditional Cryptography Baselines
*(Evaluated over 12-layer attention QKV projection and database search)*

| Evaluation Metric | Plaintext RAG | FHE Baseline | OT Baseline | Ada-IPFE (Ours) |
| :--- | :---: | :---: | :---: | :---: |
| **QKV Projection Latency** | ~0.001s | 38.50s | 8.20s | **0.18s** |
| **Retrieval Search Speed** | **~0.002s** | N/A | N/A | **6.80s** |
| **Data Privacy** | 0.0% | 100% | 100% | **100%** |
| **Retrieval Hit@10** | **92.0%** | N/A | N/A | **90.0%** |

### Comparison 2: SHIELD-RAG Cryptographic Backends (Full Corpus: 385 Docs)
*(Evaluated using SentenceTransformer embeddings on CPU)*

| Metric | Ada-IPFE (Baseline) | Sensitivity-Embedded (SE-IPFE) | Query-Derived Scope (QDCS) | Progressive Onion (POD - Depth 3) |
| :--- | :---: | :---: | :---: | :---: |
| **Key Gen Latency** | 6.73s | 2.15s | 2.90s | 6.15s |
| **Database Encryption Time** | 506.04s | 397.40s | 403.63s | 1187.36s *(3x layers)* |
| **Average Search Match Time** | 40.54s | 24.61s | 24.57s | 88.27s *(Peeling 3 layers)* |
| **Gateway Decryption Time** | 0.0108s | 0.0108s | 0.0108s | 0.235s |
| **Hit@10 Accuracy** | **90.00%** | **90.00%** *(Auth)*<br>**0.0%** *(Blocked)* | **90.00%** *(In-scope)*<br>**0.0%** *(Out-scope)* | **90.00%** *(Full depth)*<br>**0.0%** *(Shallow)* |
| **Response Relevancy** | **94.2%** | **94.2%** | **95.8%** *(Filters noise)* | **94.2%** |
| **Faithfulness** | **97.6%** | **97.6%** | **98.0%** *(Filters noise)* | **97.6%** |
| **Answer Correctness** | **90.5%** | **90.5%** | **90.5%** | **90.5%** |
| **Security Enforcement Success** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

---

## 📈 Technology Readiness Level (TRL)
The project currently achieves **TRL 3 (Analytical Proof-of-Concept)**, verifying modular exponentiation cancellation of the blenders, index matching contract constraints, and gateway attention decryptions. Roadmap tasks to reach TRL 4 (Component validation in lab environment) and TRL 5 (Validation in simulated EVM/IPFS/LLM environment) are detailed in [trl_readiness.md](trl_readiness.md).
