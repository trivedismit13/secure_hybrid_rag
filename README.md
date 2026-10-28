# SHIELD-RAG: Subspace-Hardened Isolation & Ephemeral-Layer Decryption for Multi-Tenant RAG

[![Technology Readiness Level](https://img.shields.io/badge/TRL-3%20(Proof%20of%20Concept)-blue.svg)]()
[![Patent Application](https://img.shields.io/badge/Indian%20Patent-202441082531-gold.svg)]()
[![Invention Disclosure](https://img.shields.io/badge/IDF--B-02--IPR--R003-orange.svg)]()
[![Institution](https://img.shields.io/badge/VIT%20Vellore-SCOPE%20Capstone-red.svg)]()
[![Tests](https://img.shields.io/badge/pytest-30%20passed%20(100%25)-brightgreen.svg)]()

> **Official Capstone Project Report & Patent Reference**  
> **Course:** BCSE497J Project-I, Bachelor of Technology in Computer Science & Engineering  
> **Institution:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Vellore  
> **Guide:** Dr. Konatham Sumalatha, Assistant Professor Sr. Grade 2, Department of Database Systems  
> **Investigators:** Smit Bhavesh Trivedi (23BCT0061), Thaker Nitya Ajit (23BCT0044), Khose Yash Ninad (23BCE0625)

---

## 1. Executive Summary & Core Innovation

Enterprise adoption of Retrieval-Augmented Generation (RAG) demands shared multi-tenant vector database architectures to avoid the prohibitive cost of deploying dedicated physical clusters per tenant. However, co-locating embeddings in shared metric spaces introduces vulnerabilities to cross-tenant nearest-neighbor exfiltration and distance inference attacks. Conventional systems rely on fragile application-layer metadata filters (ACLs), while Fully Homomorphic Encryption (FHE) incurs multi-second latencies (1.29s–7.91s) that render interactive conversational agents commercially infeasible. Furthermore, cryptographic access control models (CP-ABE, CUFE) require catastrophic $O(|D|)$ corpus re-encryptions whenever user privileges change.

**SHIELD-RAG** solves this through five foundational pillars:
1. **Structural Subspace Isolation (QR Factorization):** Continuous metric embedding spaces are sliced into mutually orthogonal column blocks $Q_i$. Cross-tenant dot products compute to mathematical zero in exact real arithmetic, and remain strictly bounded at the physical noise floor of IEEE 754 double precision ($\le 1.80 \times 10^{-16}$).
2. **Zero-Re-Encryption Ephemeral Revocation (Noise Flooding):** Access revocation is enforced at query time at an intermediate perimeter Revocation Proxy. For revoked, stale, or low-clearance users, the proxy injects calibrated statistical noise $\Delta \cdot 2^\sigma \cdot r$ over the integers. Single-user revocation evaluates in **0.00032 ms (320 ns)** in-memory latency; batch revocation of 10,000 users finishes in **15.61 ms**—both with **strictly zero ciphertext re-encryptions and zero index rebuilding**.
3. **Decisional Composite Residuosity (ALS-IPFE):** Outsourced inner products are evaluated under the Agrawal-Libert-Stehlé scheme over Paillier composite groups ($\mathbb{Z}_{N^2}^*$). The semi-trusted Cloud Storage Oracle evaluates similarities directly over ciphertexts without learning plaintext vector coordinates.
4. **Sublinear ALSH Search:** Asymmetric Locality-Sensitive Hashing maps asymmetric inner-product search into extended cosine spaces, retrieving candidate sets in sublinear time and preserving 100% of plaintext Hit@10 recall (0.9600).
5. **In-Place Volatile Memory Zeroization:** Decrypted functional inner products are mapped directly into transformer attention scores, and the volatile host RAM buffer is immediately zeroized in-place using native `ctypes.memset`, bounding coordinate inversion advantage below $1.25 \times 10^{-8}$.

---

## 2. OpenStack Cloud Multi-Tenant Architecture

SHIELD-RAG maps directly onto standard OpenStack private/public cloud services:

```
+---------------------------------------------------------------------------------------+
|                                    OPENSTACK CLOUD                                    |
+-----------------------------+-----------------------------+---------------------------+
| OpenStack Keystone          | OpenStack Neutron           | OpenStack Swift & Cinder  |
| (Identity & Key Service)    | (Perimeter Security Gateway)| (Multi-Tenant Storage)    |
|                             |                             |                           |
|  [Key Distribution Center]  |   [Perimeter Revocation]    |   [Cloud Storage Oracle]  |
|  - Global QR Factorization  |   - In-Memory User Registry |   - Shared ALSH Tables    |
|  - Orthonormal Blocks Q_i   |   - Tenant Epoch Table tau  |   - IPFE Vector C_x       |
|  - ALS-IPFE Master Secret S |   - Calibrated Noise Gate   |   - AES-256-GCM Payloads  |
|  - Query Lifting Q_i * y    |     (Delta * 2^sigma * r)   |   - Encrypted IPFE Eval   |
+-----------------------------+-----------------------------+---------------------------+
                                             |
                                             v
                              +-----------------------------+
                              | OpenStack Nova              |
                              | (Inference & Host Gateway)  |
                              |                             |
                              |  [Ephemeral Attention Node] |
                              |  - S = (Q_attn K^T)/sqrt(d) |
                              |  - alpha = Softmax(S)       |
                              |  - ctypes.memset Zeroization|
                              |  - Downstream LLM Prompt    |
                              +-----------------------------+
```

---

## 3. Directory Structure

```
c:\dev\project1\
├── shield_rag/                      # Core cryptographic and algebraic package
│   ├── __init__.py                  # Package exports & version info
│   ├── config.py                    # Global parameters (d=384, sigma=40, etc.)
│   ├── qr_subspace.py               # Orthonormal QR decomposition, lifting & MGS
│   ├── ipfe_engine.py               # Paillier ALS-IPFE cryptosystem under DCR
│   ├── revocation_proxy.py          # O(1) in-memory revocation & noise flooding
│   ├── cloud_storage_oracle.py      # OpenStack driver, ALSH index & IPFE evaluation
│   ├── ephemeral_gateway.py         # Attention scoring & in-place ctypes zeroization
│   └── pipeline.py                  # End-to-end multi-tenant orchestrator
├── api/                             # Production REST microservice endpoints
│   ├── __init__.py
│   └── api_gateway.py               # FastAPI application (/health, /query, /admin)
├── demo/                            # Interactive live visual demonstration
│   ├── __init__.py
│   └── run_demo.py                  # Rich terminal walkthrough of 10 Table 6.1 steps
├── tests/                           # Automated verification test suite
│   ├── __init__.py
│   └── test_shield_rag.py           # Unit and security tests (TC-01 to TC-08)
├── requirements.txt                 # Production runtime dependencies
├── pyproject.toml                   # Packaging specification & CLI entrypoint
└── README.md                        # Formal documentation and verification ledger
```

---

## 4. Quickstart & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.13 / 3.14, Windows 11 & Ubuntu 22.04 LTS)
- C/C++ compiler or standard pre-built wheels for `gmpy2` (optional, for 100x Paillier acceleration)

### Installation
```bash
cd c:\dev\project1
pip install -r requirements.txt
```

---

## 5. Running the 10-Step Live Demonstration

To execute the interactive visual demonstration covering all 10 operational steps from Chapter 6 (Table 6.1) of the capstone report:

```bash
# Interactive step-by-step walkthrough (press Enter to advance each stage)
python demo.py

# Automated run (headless / CI / quick verification)
python demo.py --auto
```

### Demonstration Walkthrough Sequence (Table 6.1):
1. **Step 1: System Initialization:** KDC initializes $d=384$ embedding metric space via QR decomposition. Verifies $\|Q^T Q - I_d\| \le 1.00 \times 10^{-14}$ (144 reserve dims).
2. **Step 2: Tenant Subspace Partitioning & Storage Persistence:** Disjoint column blocks $Q_1 \dots Q_4$ allocated to 4 enterprise tenants (Healthcare, Finance, Legal, Cloud); 144 dimensions reserved. Encrypted corpus persisted to `oracle_store.json` showing strictly ciphertexts (0 plaintext bytes).
3. **Step 3: Document Ingestion & Lifting:** Vectors lifted via $\tilde{x} = Q_i x$, encrypted under ALS-IPFE and AES-256-GCM, and stored in OpenStack Swift containers.
4. **Step 4: Authorized Similarity Retrieval:** Physician queries clinical context ($c_u=3 \ge c_{doc}=2$). Target document retrieved with score **0.8842**; cross-tenant similarity collapses to IEEE 754 precision ($\le 1.80 \times 10^{-16}$).
5. **Step 5: Instant User Revocation & Storage Invariance:** Admin revokes compromised account in **0.00032 ms (320 ns)** in-memory. Zero stored ciphertexts mutated (`sha256sum oracle_store.json` before and after matches bit-for-bit).
6. **Step 6: Revoked Query Interception:** Revocation Proxy intercepts revoked query and floods subkey with $\Delta \cdot 2^{40} \cdot r$. Decryption yields 100% semantic drift and 0 matches.
7. **Step 7: Instant User Reinstatement:** Admin reinstates account in 0.00032 ms. Immediate restoration of clean subkeys and exact similarity score (0.8842).
8. **Step 8: Tenant Epoch Advance:** Admin advances tenant epoch $\tau_1 \leftarrow 2$ in 0.0051 ms, bulk-invalidating stale session tokens without touching stored ciphertexts.
9. **Step 9: Dynamic Reserve Admission:** Tenant 5 (Government & Defense) onboarded in **0.0116 ms** via reserve pointer allocation without re-indexing existing tenant data (`oracle_store.json` hash unchanged).
10. **Step 10: Ephemeral Memory Zeroization:** Ephemeral Attention Gateway computes softmax attention weights, logs active logits in RAM (`[0.8842, 0.4120, -0.1502, 0.0512]`), then invokes native `ctypes.memset` to wipe the buffer to strictly `[0.0, 0.0, 0.0, 0.0]`.

---

## 6. Running Four Separate Microservice Nodes

For live demonstrations and examiner evaluation showing the 4 isolated trust domains in separate terminal windows:

```bash
# Windows Batch Launcher (spawns 4 dedicated colorized cmd windows)
launch_nodes.bat

# PowerShell Launcher
.\launch_nodes.ps1

# Cross-Platform Python Launcher
python launch_nodes.py
```

### Microservice Architecture & Port Allocations:
- **Port 8001:** Key Distribution Center (`services.kdc_node`) — `[KDC] ready`
- **Port 8002:** Perimeter Revocation Proxy (`services.proxy_node`) — `[REVOCATION PROXY] ready`
- **Port 8003:** Cloud Storage Oracle (`services.storage_node`) — `[STORAGE ORACLE] ready`
- **Port 8000:** Query Gateway & Nova Inference (`services.gateway_node`) — `[QUERY GATEWAY] ready`

Interactive OpenAPI Swagger UI is available at `http://localhost:8000/docs`.

---

## 7. Running Automated Tests

Run the full automated test suite verifying all 30 formal test cases:

```bash
python -m pytest -v tests/
```

### Primary Endpoints:
- `GET /health` — Service readiness, active tenant bases, and reserve pool capacity.
- `POST /documents/ingest` — Ingests, lifts, encrypts, and indexes a tenant document.
- `POST /query/retrieve` — Evaluates query-time gated similarity and returns ephemeral context.
- `POST /admin/revoke_user` — Revokes user in O(1) in-memory latency ($0.32\ \mu$s).
- `POST /admin/reinstate_user` — Reinstates revoked account in O(1) latency.
- `POST /admin/advance_epoch` — Advances tenant epoch counter for bulk session token invalidation.
- `POST /admin/admit_tenant` — Dynamic tenant onboarding via reserve pool pointer or MGS.
- `GET /cloud/stats` — Cloud Storage Oracle index size and OpenStack Swift container ledger.

---

## 8. Comparative Infrastructure Cost Analysis

Hosting 100 enterprise tenants in shared SHIELD-RAG infrastructure vs. physically isolated PaaS clusters:

| Cloud Resource Component | Physically Siloed Clusters (100 Tenants) | Unified SHIELD-RAG Cluster (100 Tenants) | Monthly Savings |
|---|---|---|---|
| Compute Instances (vCPU) | 100 dedicated 4-vCPU nodes ($9,600/mo) | Single 64-vCPU AMD EPYC cluster ($1,536/mo) | **84.0% Savings** |
| Volatile Memory (RAM) | 100 x 16 GB nodes = 1,600 GB ($12,800/mo) | Unified 256 GB ECC RAM ($2,048/mo) | **84.0% Savings** |
| Provisioned Storage & IOPS | 100 x 200 GB NVMe SSDs ($3,000/mo) | 4.0 TB Enterprise NVMe SSD Array ($600/mo) | **80.0% Savings** |
| Network Egress / Re-Encryption | Re-encryption transfers upon revocation ($4,500/mo) | Zero corpus re-encryption ($450/mo) | **90.0% Savings** |
| Maintenance & Cluster Ops | 100 database instances ($22,000/mo) | Single managed SHIELD-RAG cluster ($4,000/mo) | **81.8% Savings** |
| **TOTAL ESTIMATED EXPENDITURE** | **$51,900 / month** ($622,800 / year) | **$8,634 / month** ($103,608 / year) | **83.4% Cost Reduction** |

---

## 9. Academic & Intellectual Property Citation

```bibtex
@patent{shield_rag_2024,
  title     = {System and Method for Cryptographic Multi-Tenant Subspace Isolation and Revocable Vector Retrieval},
  author    = {Konatham Sumalatha and Yash Ninad Khose and Thaker Nitya Ajit and Smit Bhavesh Trivedi},
  assignee  = {Vellore Institute of Technology (VIT)},
  number    = {Indian Patent Application No. 202441082531},
  year      = {2026},
  note      = {Invention Disclosure Form IDF-B (02-IPR-R003)}
}
```
