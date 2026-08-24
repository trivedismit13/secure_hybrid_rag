# SHIELD-RAG Literature Review Summary

This document lists and summarizes the academic research papers used during the literature review and conceptual design phase of the SHIELD-RAG project.

---

## 1. Paper 1: A Hybrid Retrieval and Reranking Framework for Evidence-Grounded Retrieval-Augmented Generation

*   **Authors:** Fariba Afrin Irany, Sampson Akwafuo
*   **Affiliation:** University of North Texas
*   **Source:** arXiv Preprint (2026), arXiv ID: [2605.01664v1](https://arxiv.org/abs/2605.01664v1)
*   **Local File:** [paper1.pdf](file:///C:/Users/Lenovo/Downloads/paper1.pdf)
*   **Key Focus:** Hybrid sparse-dense retrieval and cross-encoder reranking.
*   **Relation to SHIELD-RAG:** Provides the baseline methodology for combining traditional search structures with semantic query-vector alignments, which we secured using Ada-IPFE and ALSH hashing.

---

## 2. Paper 2: Leveraging the Domain Adaptation of Retrieval Augmented Generation Models for Question Answering and Reducing Hallucination

*   **Authors:** Salman Rakin, Md. A.R. Shibly, Zahin M. Hossain, Zeeshan Khan, Dr. Md. Mostofa Akbar
*   **Affiliation:** BUET, SurroundApps
*   **Source:** LaTeX Preprint (2024)
*   **Local File:** [paper2.pdf](file:///C:/Users/Lenovo/Downloads/paper2.pdf)
*   **Key Focus:** Evaluating how targeted domain adaptation in RAG models minimizes factual hallucinations in QA systems.
*   **Relation to SHIELD-RAG:** Motivated the need for **Query-Derived Cryptographic Scope (QDCS)**, ensuring that retrieval is dynamically restricted to specific domain categories to prevent cross-domain hallucination noise.

---

## 3. Paper 3: Auto-GDA: Automatic Domain Adaptation for Grounding Verification in Retrieval-Augmented Generation

*   **Authors:** Tobias Leemann, Periklis Petridis, Giuseppe Vietri, Dionysis Manousakas, Aaron Roth, Sergül Aydöre
*   **Affiliation:** University of Tübingen, MIT, AWS AI Labs
*   **Source:** ICLR 2025 Conference Paper
*   **Local File:** [paper3.pdf](file:///C:/Users/Lenovo/Downloads/paper3.pdf)
*   **Key Focus:** Automatic domain adaptation for Natural Language Inference (NLI) models used to verify if generated answers are grounded in retrieved context.
*   **Relation to SHIELD-RAG:** Informed the development of the **Decryption-Enabled Attention Gateway** (Algorithm 2) to ensure that grounding verification can be run securely inside the LLM without unmasking context embeddings in transit.
