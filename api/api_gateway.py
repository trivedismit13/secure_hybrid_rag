"""
Production REST API Gateway for SHIELD-RAG.

FastAPI microservice implementation providing HTTP endpoints for:
- Vector ingestion and subspace lifting.
- Encrypted similarity retrieval and ephemeral attention scoring.
- Query-time subkey gating and administrative user revocation.
- Tenant epoch advance and dynamic reserve pool tenant admission.
- Cloud Storage Oracle and OpenStack Swift metadata inspection.
"""

from typing import Any, Dict, List, Optional
import time
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from shield_rag.config import (
    DEFAULT_CLEARANCE,
    DEFAULT_DOC_CLEARANCE,
    GLOBAL_DIM,
)
from shield_rag.pipeline import ShieldRAGPipeline

app = FastAPI(
    title="SHIELD-RAG Cryptographic Vector Service",
    description="Subspace-Hardened Isolation and Ephemeral-Layer Decryption for Multi-Tenant RAG",
    version="1.0.0",
)

# Global pipeline instance
_pipeline: Optional[ShieldRAGPipeline] = None


def get_pipeline() -> ShieldRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ShieldRAGPipeline(global_dim=128, num_tenants=4, dims_per_tenant=20, key_bits=512)
        _pipeline.populate_sample_data()
        _pipeline.revocation_proxy.register_user("user_1", "tenant_0", clearance=3)
        _pipeline.revocation_proxy.register_user("user_2", "tenant_1", clearance=3)
    return _pipeline


# ─── Request & Response Models ───────────────────────────────────────

class IngestRequest(BaseModel):
    tenant_id: str = Field(..., description="Enterprise tenant identifier (e.g. 'tenant_0')")
    text: str = Field(..., description="Plaintext document text to ingest")
    clearance: int = Field(DEFAULT_DOC_CLEARANCE, description="Document sensitivity classification level")
    doc_id: Optional[str] = Field(None, description="Optional custom document ID")


class IngestResponse(BaseModel):
    doc_id: str
    tenant_id: str
    clearance: int
    alsh_hash_count: int
    status: str


class RetrieveRequest(BaseModel):
    user_id: str = Field(..., description="Querier user ID")
    tenant_id: str = Field(..., description="Querier enterprise tenant ID")
    query_text: str = Field(..., description="Natural language search query")
    token_epoch: Optional[int] = Field(None, description="User's session credential epoch")
    user_clearance: Optional[int] = Field(None, description="User's security clearance level")
    top_k: int = Field(5, description="Number of top candidates to retrieve")


class RetrieveResponse(BaseModel):
    user_id: str
    tenant_id: str
    query_text: str
    is_noise_flooded: bool
    status_message: str
    retrieval_latency_ms: float
    top_similarity: float
    cross_tenant_leakage: float
    zeroization_verified: bool
    top_documents: List[Dict[str, Any]]
    context_text: str


class RevokeUserRequest(BaseModel):
    user_id: str = Field(..., description="Target user identifier to revoke")


class ReinstateUserRequest(BaseModel):
    user_id: str = Field(..., description="Target user identifier to reinstate")


class AdvanceEpochRequest(BaseModel):
    tenant_id: str = Field(..., description="Target tenant ID to advance epoch for")


class AdmitTenantRequest(BaseModel):
    tenant_id: str = Field(..., description="New tenant identifier")
    dim: int = Field(20, description="Subspace dimensionality to allocate")
    mechanism: str = Field("reserve", description="'reserve' (pointer) or 'mgs' (Gram-Schmidt)")


# ─── Endpoints ───────────────────────────────────────────────────────

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Returns service health status and cryptographic readiness."""
    p = get_pipeline()
    max_id, max_leakage = p.subspace_engine.verify_orthonormality()
    return {
        "status": "healthy",
        "service": "SHIELD-RAG",
        "version": "1.0.0",
        "global_dimension": p.global_dim,
        "active_tenants": list(p.subspace_engine.tenant_bases.keys()),
        "reserve_dimensions": p.subspace_engine.reserve_dimensions,
        "max_orthonormality_error": max_id,
        "max_cross_tenant_leakage": max_leakage,
    }


@app.post("/documents/ingest", response_model=IngestResponse)
def ingest_document(req: IngestRequest) -> IngestResponse:
    """Ingests, lifts, and encrypts a document for an enterprise tenant."""
    p = get_pipeline()
    if req.tenant_id not in p.subspace_engine.tenant_bases:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{req.tenant_id}' has no allocated subspace.",
        )
    record = p.ingest_document(
        tenant_id=req.tenant_id,
        text=req.text,
        clearance=req.clearance,
        doc_id=req.doc_id,
    )
    return IngestResponse(
        doc_id=record.doc_id,
        tenant_id=record.tenant_id,
        clearance=record.clearance,
        alsh_hash_count=len(record.alsh_hash),
        status="INGESTED_AND_ENCRYPTED",
    )


@app.post("/query/retrieve", response_model=RetrieveResponse)
def retrieve_query(req: RetrieveRequest) -> RetrieveResponse:
    """Performs end-to-end encrypted similarity retrieval with subkey gating."""
    p = get_pipeline()
    if req.tenant_id not in p.subspace_engine.tenant_bases:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{req.tenant_id}' is not an active tenant.",
        )
    res = p.retrieve(
        user_id=req.user_id,
        tenant_id=req.tenant_id,
        query_text=req.query_text,
        token_epoch=req.token_epoch,
        user_clearance=req.user_clearance,
        top_k=req.top_k,
    )
    return RetrieveResponse(
        user_id=res.user_id,
        tenant_id=res.tenant_id,
        query_text=res.query_text,
        is_noise_flooded=res.is_noise_flooded,
        status_message=res.status_message,
        retrieval_latency_ms=res.retrieval_latency_ms,
        top_similarity=res.top_similarity,
        cross_tenant_leakage=res.cross_tenant_leakage,
        zeroization_verified=res.zeroization_verified,
        top_documents=res.top_documents,
        context_text=res.context_text,
    )


@app.post("/admin/revoke_user")
def revoke_user(req: RevokeUserRequest) -> Dict[str, Any]:
    """Revokes a user in O(1) in-memory latency (~0.00032 ms)."""
    p = get_pipeline()
    elapsed_ms = p.revoke_user(req.user_id)
    return {
        "user_id": req.user_id,
        "is_revoked": True,
        "revocation_latency_ms": elapsed_ms,
        "ciphertexts_re_encrypted": 0,
        "status": "REVOKED",
    }


@app.post("/admin/reinstate_user")
def reinstate_user(req: ReinstateUserRequest) -> Dict[str, Any]:
    """Reinstates a revoked user in O(1) in-memory latency."""
    p = get_pipeline()
    elapsed_ms = p.reinstate_user(req.user_id)
    return {
        "user_id": req.user_id,
        "is_revoked": False,
        "reinstatement_latency_ms": elapsed_ms,
        "ciphertexts_re_encrypted": 0,
        "status": "REINSTATED",
    }


@app.post("/admin/advance_epoch")
def advance_epoch(req: AdvanceEpochRequest) -> Dict[str, Any]:
    """Advances a tenant's active epoch to invalidate all stale session tokens."""
    p = get_pipeline()
    t0 = time.perf_counter_ns()
    new_epoch = p.advance_tenant_epoch(req.tenant_id)
    t1 = time.perf_counter_ns()
    return {
        "tenant_id": req.tenant_id,
        "new_epoch": new_epoch,
        "epoch_advance_latency_ms": (t1 - t0) / 1_000_000.0,
        "ciphertexts_re_encrypted": 0,
        "status": "EPOCH_ADVANCED",
    }


@app.post("/admin/admit_tenant")
def admit_tenant(req: AdmitTenantRequest) -> Dict[str, Any]:
    """Dynamically onboards a new tenant via reserve pool or MGS expansion."""
    p = get_pipeline()
    t0 = time.perf_counter()
    if req.mechanism == "reserve":
        Q_new = p.admit_tenant_reserve(req.tenant_id, req.dim)
        mech_used = "reserve_pool_pointer"
    else:
        Q_new = p.admit_tenant_mgs(req.tenant_id, req.dim)
        mech_used = "incremental_modified_gram_schmidt"
    t1 = time.perf_counter()
    return {
        "tenant_id": req.tenant_id,
        "allocated_dim": req.dim,
        "mechanism": mech_used,
        "admission_latency_ms": (t1 - t0) * 1000.0,
        "reserve_remaining": p.subspace_engine.reserve_dimensions,
        "status": "ADMITTED",
    }


class DeallocateTenantRequest(BaseModel):
    tenant_id: str = Field(..., description="Enterprise tenant ID to deallocate")


@app.post("/admin/deallocate_tenant")
def deallocate_tenant(req: DeallocateTenantRequest) -> Dict[str, Any]:
    """De-allocates an enterprise tenant in O(1) latency without altering ciphertexts."""
    p = get_pipeline()
    return p.deallocate_tenant(req.tenant_id)


@app.get("/cloud/stats")
def cloud_stats() -> Dict[str, Any]:
    """Inspects OpenStack Swift containers and Cloud Storage Oracle index."""
    p = get_pipeline()
    return {
        "total_documents_indexed": len(p.storage_oracle.documents),
        "tenants_with_documents": list(p.storage_oracle.tenant_documents.keys()),
        "openstack_swift_containers": list(p.storage_oracle.openstack.swift_containers.keys()),
        "alsh_num_tables": p.storage_oracle.alsh_index.num_tables,
        "alsh_num_hashes": p.storage_oracle.alsh_index.num_hashes,
    }


@app.get("/openstack/status")
def openstack_status() -> Dict[str, Any]:
    """Returns comprehensive OpenStack Cloud deployment metrics (Keystone, Neutron, Swift, Cinder, Nova)."""
    p = get_pipeline()
    return p.storage_oracle.openstack.get_cloud_summary()
