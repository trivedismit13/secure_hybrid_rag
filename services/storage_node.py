"""
Cloud Storage Oracle Microservice Node for SHIELD-RAG.
Runs on Port 8003.

Role:
- Honest-but-Curious Cloud Storage Oracle (OpenStack Swift/Cinder).
- ALS-IPFE vector ciphertext storage in (Z_{N^2}^*)^{d+1}.
- AES-256-GCM authenticated payload encryption.
- Encrypted document persistence to oracle_store.json for zero-re-encryption verification.
- ALSH sublinear indexing & encrypted inner-product MIPS search.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from shield_rag.cloud_storage_oracle import CloudStorageOracle
from shield_rag.config import (
    ALSH_TOP_K,
    DEFAULT_DOC_CLEARANCE,
    DIMS_PER_TENANT,
    GLOBAL_DIM,
    IPFE_KEY_BITS,
    NUM_TENANTS,
    SAMPLE_DOCUMENTS,
    SEED,
)
from shield_rag.ipfe_engine import IPFECryptosystem
from services.kdc_node import embed_text_kdc

app = FastAPI(
    title="SHIELD-RAG Cloud Storage Oracle Node",
    description="Encrypted Vector Storage & Functional Decryption Oracle",
    version="1.0.0",
)

STORE_FILEPATH = "oracle_store.json"

_oracle: Optional[CloudStorageOracle] = None


def get_oracle() -> CloudStorageOracle:
    global _oracle
    if _oracle is None:
        ipfe = IPFECryptosystem(key_bits=IPFE_KEY_BITS, dim=GLOBAL_DIM, seed=SEED)
        _oracle = CloudStorageOracle(
            dim=GLOBAL_DIM,
            ipfe=ipfe,
            seed=SEED,
            store_filepath=STORE_FILEPATH,
        )
        # Attempt to load persisted encrypted records from JSON store
        loaded = _oracle.load_store_from_json(STORE_FILEPATH)
        if loaded == 0:
            # Pre-populate sample enterprise documents for Tenants 0-3
            from shield_rag.qr_subspace import QRSubspaceEngine
            engine = QRSubspaceEngine(
                global_dim=GLOBAL_DIM,
                num_tenants=NUM_TENANTS,
                dims_per_tenant=DIMS_PER_TENANT,
                seed=SEED,
            )
            for t_idx, docs in SAMPLE_DOCUMENTS.items():
                t_id = f"tenant_{t_idx}"
                _oracle.openstack.keystone_create_project(t_id, f"Enterprise Tenant {t_idx}")
                tenant_docs = list(docs[:2])
                if t_idx == 0:
                    tenant_docs.insert(
                        0,
                        "Pediatric immunotherapy dosage bounds and clinical trial protocols: recommended dosage ranges between 2.5mg/kg and 5.0mg/kg administered bi-weekly.",
                    )
                for doc_idx, doc_text in enumerate(tenant_docs):
                    tenant_dim = engine.tenant_bases[t_id].shape[1]
                    base_x = embed_text_kdc(doc_text, tenant_dim)
                    x_lifted = engine.lift_document(t_id, base_x)
                    doc_id = f"doc_{t_id}_{doc_idx+1:04d}"
                    _oracle.ingest_document(
                        doc_id=doc_id,
                        tenant_id=t_id,
                        x_lifted=x_lifted,
                        plaintext_text=doc_text,
                        clearance=DEFAULT_DOC_CLEARANCE,
                    )
            _oracle.export_store_to_json(STORE_FILEPATH)
    return _oracle


# ─── Pydantic Models ──────────────────────────────────────────────────

class StorageIngestRequest(BaseModel):
    doc_id: str = Field(..., description="Unique document ID")
    tenant_id: str = Field(..., description="Enterprise tenant ID")
    x_lifted: List[float] = Field(..., description="Lifted document vector in R^d")
    plaintext_text: str = Field(..., description="Raw text payload to encrypt via AES-256-GCM")
    clearance: int = Field(DEFAULT_DOC_CLEARANCE, description="Document classification level")


class StorageIngestResponse(BaseModel):
    doc_id: str
    tenant_id: str
    clearance: int
    status: str
    store_hash: str
    latency_ms: float


class StorageSearchRequest(BaseModel):
    gated_subkey: int = Field(..., description="Functional subkey (authorized or noise-flooded)")
    y_lifted: List[float] = Field(..., description="Lifted query vector in R^d")
    top_k: int = Field(ALSH_TOP_K, description="Number of top candidates to retrieve")


class RankedCandidate(BaseModel):
    rank: int
    doc_id: str
    tenant_id: str
    clearance: int
    similarity_score: float
    aes_iv_hex: str
    aes_ciphertext_hex: str
    aes_tag_hex: str


class StorageSearchResponse(BaseModel):
    candidates: List[RankedCandidate]
    total_candidates_searched: int
    latency_ms: float


# ─── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    oracle = get_oracle()
    return {
        "node": "STORAGE_ORACLE",
        "status": "ready",
        "port": 8003,
        "documents_indexed": len(oracle.documents),
        "tenants_active": list(oracle.tenant_documents.keys()),
        "store_filepath": STORE_FILEPATH,
        "store_hash": oracle.get_store_hash(STORE_FILEPATH),
    }


@app.post("/storage/ingest", response_model=StorageIngestResponse)
def ingest_record(req: StorageIngestRequest) -> StorageIngestResponse:
    t0 = time.perf_counter()
    oracle = get_oracle()
    lifted_np = np.array(req.x_lifted, dtype=np.float64)

    record = oracle.ingest_document(
        doc_id=req.doc_id,
        tenant_id=req.tenant_id,
        x_lifted=lifted_np,
        plaintext_text=req.plaintext_text,
        clearance=req.clearance,
    )
    store_hash = oracle.export_store_to_json(STORE_FILEPATH)
    t1 = time.perf_counter()

    return StorageIngestResponse(
        doc_id=record.doc_id,
        tenant_id=record.tenant_id,
        clearance=record.clearance,
        status="INGESTED_AND_ENCRYPTED",
        store_hash=store_hash,
        latency_ms=(t1 - t0) * 1000.0,
    )


@app.post("/storage/search", response_model=StorageSearchResponse)
def search_candidates(req: StorageSearchRequest) -> StorageSearchResponse:
    t0 = time.perf_counter()
    oracle = get_oracle()
    lifted_np = np.array(req.y_lifted, dtype=np.float64)

    results = oracle.search(
        gated_subkey=req.gated_subkey,
        y_lifted=lifted_np,
        top_k=req.top_k,
    )
    t1 = time.perf_counter()

    candidates: List[RankedCandidate] = []
    for rank_idx, (doc_id, sim, rec) in enumerate(results, 1):
        candidates.append(
            RankedCandidate(
                rank=rank_idx,
                doc_id=doc_id,
                tenant_id=rec.tenant_id,
                clearance=rec.clearance,
                similarity_score=sim,
                aes_iv_hex=rec.aes_iv.hex(),
                aes_ciphertext_hex=rec.aes_ciphertext.hex(),
                aes_tag_hex=rec.aes_tag.hex(),
            )
        )

    return StorageSearchResponse(
        candidates=candidates,
        total_candidates_searched=len(oracle.documents),
        latency_ms=(t1 - t0) * 1000.0,
    )


@app.get("/storage/store_hash")
def get_store_hash() -> Dict[str, Any]:
    oracle = get_oracle()
    h = oracle.get_store_hash(STORE_FILEPATH)
    fsize = os.path.getsize(STORE_FILEPATH) if os.path.exists(STORE_FILEPATH) else 0
    return {
        "filepath": STORE_FILEPATH,
        "store_hash": h,
        "size_bytes": fsize,
        "document_count": len(oracle.documents),
    }


@app.get("/storage/export")
def export_store() -> Dict[str, Any]:
    oracle = get_oracle()
    h = oracle.export_store_to_json(STORE_FILEPATH)
    return {
        "status": "EXPORTED",
        "filepath": STORE_FILEPATH,
        "store_hash": h,
        "document_count": len(oracle.documents),
    }


@app.get("/storage/tenant_aes_key/{tenant_id}")
def get_tenant_aes_key(tenant_id: str) -> Dict[str, str]:
    oracle = get_oracle()
    key = oracle.get_or_create_tenant_aes_key(tenant_id)
    return {
        "tenant_id": tenant_id,
        "key_hex": key.hex(),
    }


@app.get("/storage/stats")
def storage_stats() -> Dict[str, Any]:
    oracle = get_oracle()
    return oracle.openstack.get_cloud_summary()


def print_banner() -> None:
    print("=" * 70)
    print(" [STORAGE ORACLE] ready on http://localhost:8003")
    print(" Role: Cloud Storage Oracle (OpenStack Swift/Cinder Encrypted Store)")
    print(f" Storage Persistence: {STORE_FILEPATH}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    print_banner()
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="warning")
