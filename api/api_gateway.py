"""
Production REST API Gateway for SHIELD-RAG.

FastAPI microservice implementation providing HTTP endpoints for:
- Vector ingestion and subspace lifting.
- Encrypted similarity retrieval and ephemeral attention scoring.
- Query-time subkey gating and administrative user revocation.
- Tenant epoch advance and dynamic reserve pool tenant admission.
- Cloud Storage Oracle and OpenStack Swift metadata inspection.
- Ephemeral memory zeroization diagnostics via ctypes.memset.
"""

from services.gateway_node import (
    AdmitTenantRequest,
    AdvanceEpochRequest,
    DeallocateTenantRequest,
    IngestRequest,
    IngestResponse,
    ReinstateUserRequest,
    RetrieveRequest,
    RetrieveResponse,
    RevokeUserRequest,
    app,
    get_pipeline,
)

__all__ = [
    "app",
    "get_pipeline",
    "IngestRequest",
    "IngestResponse",
    "RetrieveRequest",
    "RetrieveResponse",
    "RevokeUserRequest",
    "ReinstateUserRequest",
    "AdvanceEpochRequest",
    "AdmitTenantRequest",
    "DeallocateTenantRequest",
]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.api_gateway:app", host="0.0.0.0", port=8000, reload=True)
