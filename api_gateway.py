"""
Root-level API Gateway shim for SHIELD-RAG.
Supports running `uvicorn api_gateway:app --port 8000` directly from the repository root.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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
    uvicorn.run("api_gateway:app", host="0.0.0.0", port=8000, reload=True)
