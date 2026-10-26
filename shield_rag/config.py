"""
Global configuration constants for SHIELD-RAG.
All system parameters referenced in the reports are centralized here.
"""

# ─── Embedding & Subspace ────────────────────────────────────────────
GLOBAL_DIM = 384                # d: global embedding metric-space dimension
NUM_TENANTS = 4                 # M: initial number of enterprise tenants
DIMS_PER_TENANT = 60            # d_i: subspace dimensionality per tenant (uniform)
RESERVE_DIMS = GLOBAL_DIM - (NUM_TENANTS * DIMS_PER_TENANT)  # 144 reserve pool
SEED = 42                       # reproducible RNG seed

# ─── IPFE Cryptographic Parameters ───────────────────────────────────
IPFE_KEY_BITS = 512             # |N| bit-length for demo speed (prod: >=3072)
IPFE_SCALE_FACTOR = 10**6       # fixed-point scaling for float→int lifting
SIGMA = 40                      # σ: statistical noise-flooding security parameter
RHO = 60                        # ρ: polynomial-size bound on integer dot products

# ─── Revocation Proxy ────────────────────────────────────────────────
DEFAULT_CLEARANCE = 3           # default user clearance level
DEFAULT_DOC_CLEARANCE = 2       # default document classification level
INITIAL_EPOCH = 1               # τ₀: starting epoch for every tenant

# ─── ALSH Parameters ─────────────────────────────────────────────────
ALSH_NUM_TABLES = 10            # L: number of hash tables
ALSH_NUM_HASHES = 8             # K: hash functions per table
ALSH_BUCKET_WIDTH = 4.0         # w: bucket width
ALSH_TOP_K = 10                 # top-K candidates to retrieve
ALSH_M_PARAM = 3                # m: ALSH augmentation parameter for P/Q transforms

# ─── Tenant Metadata ─────────────────────────────────────────────────
TENANT_NAMES = {
    0: "Healthcare",
    1: "Financial",
    2: "Legal",
    3: "Cloud Technology",
}

# ─── Demo Sample Documents ───────────────────────────────────────────
SAMPLE_DOCUMENTS = {
    0: [  # Healthcare
        "Patient records and clinical trial data management systems",
        "HIPAA-compliant electronic health record encryption protocols",
        "Medical imaging analysis using deep learning neural networks",
        "Drug interaction prediction models for pharmacy databases",
        "Telemedicine infrastructure and secure video consultation",
    ],
    1: [  # Financial
        "High-frequency trading algorithmic risk assessment models",
        "Anti-money laundering transaction monitoring pipelines",
        "Blockchain-based digital asset custody solutions",
        "Credit scoring machine learning feature engineering",
        "Real-time fraud detection using anomaly detection systems",
    ],
    2: [  # Legal
        "Natural language processing for contract clause extraction",
        "Intellectual property patent landscape analysis tools",
        "Regulatory compliance automated audit trail systems",
        "Legal case similarity search and precedent matching",
        "Data privacy impact assessment automation frameworks",
    ],
    3: [  # Cloud Technology
        "Kubernetes container orchestration security hardening",
        "Multi-cloud infrastructure cost optimization dashboards",
        "Serverless function cold-start latency reduction methods",
        "Cloud-native microservice observability and tracing",
        "Zero-trust network architecture implementation guides",
    ],
}

SAMPLE_QUERIES = {
    0: "Secure patient health records with encryption",
    1: "Fraud detection in financial transaction monitoring",
    2: "Contract analysis and legal document retrieval",
    3: "Cloud infrastructure security and cost optimization",
}
