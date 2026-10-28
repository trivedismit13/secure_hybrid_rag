"""
Perimeter Revocation Proxy Microservice Node for SHIELD-RAG.
Runs on Port 8002.

Role:
- OpenStack Neutron perimeter security group emulation.
- O(1) in-memory revocation registry (~320 ns latency).
- Monotonically increasing tenant epoch table (tau_i) for bulk stale token invalidation.
- Statistical noise flooding mechanism: sk' = sk_y + Delta * 2^sigma * r.
"""

import sys
import time
from typing import Any, Dict, List, Optional
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

STORAGE_URL = "http://127.0.0.1:8003"

from shield_rag.config import (
    DEFAULT_CLEARANCE,
    DEFAULT_DOC_CLEARANCE,
    INITIAL_EPOCH,
    RHO,
    SIGMA,
)
from shield_rag.revocation_proxy import RevocationProxyEngine

app = FastAPI(
    title="SHIELD-RAG Revocation Proxy Node",
    description="Perimeter Revocation Proxy & Noise Gating Engine",
    version="1.0.0",
)

_proxy: Optional[RevocationProxyEngine] = None


def get_proxy() -> RevocationProxyEngine:
    global _proxy
    if _proxy is None:
        _proxy = RevocationProxyEngine(sigma=SIGMA, rho=RHO)
        # Register default users
        _proxy.ensure_tenant("tenant_0")
        _proxy.ensure_tenant("tenant_1")
        _proxy.ensure_tenant("tenant_2")
        _proxy.ensure_tenant("tenant_3")
        _proxy.register_user("user_1", "tenant_0", clearance=3)
        _proxy.register_user("user_2", "tenant_1", clearance=3)
        _proxy.register_user("user_3", "tenant_2", clearance=3)
        _proxy.register_user("user_4", "tenant_3", clearance=3)
    return _proxy


# ─── Pydantic Models ──────────────────────────────────────────────────

class GateSubkeyRequest(BaseModel):
    user_id: str = Field(..., description="Requesting user ID")
    tenant_id: str = Field(..., description="Requesting tenant ID")
    token_epoch: Optional[int] = Field(None, description="User's session epoch token")
    user_clearance: Optional[int] = Field(None, description="User's clearance level")
    doc_clearance: int = Field(DEFAULT_DOC_CLEARANCE, description="Document clearance requirement")
    raw_subkey: int = Field(..., description="Raw functional subkey from KDC")


class GateSubkeyResponse(BaseModel):
    user_id: str
    tenant_id: str
    gated_subkey: int
    raw_subkey: int
    is_flooded: bool
    status_message: str
    gate_latency_ns: int
    gate_latency_ms: float


class SearchForwardRequest(BaseModel):
    user_id: str = Field(..., description="Requesting user ID")
    tenant_id: str = Field(..., description="Requesting tenant ID")
    raw_subkey: int = Field(..., description="Raw functional subkey from KDC")
    y_lifted: List[float] = Field(..., description="Lifted query vector in R^d")
    top_k: int = Field(5, description="Number of top candidates to retrieve")
    token_epoch: Optional[int] = Field(None, description="User's session epoch token")
    user_clearance: Optional[int] = Field(None, description="User's clearance level")
    doc_clearance: int = Field(DEFAULT_DOC_CLEARANCE, description="Document clearance requirement")


class SearchForwardResponse(BaseModel):
    user_id: str
    tenant_id: str
    gated_subkey: int
    is_flooded: bool
    status_message: str
    gate_latency_ns: int
    candidates: List[Dict[str, Any]]


class RevokeUserRequest(BaseModel):
    user_id: str = Field(..., description="Target user ID to revoke")


class ReinstateUserRequest(BaseModel):
    user_id: str = Field(..., description="Target user ID to reinstate")


class AdvanceEpochRequest(BaseModel):
    tenant_id: str = Field(..., description="Target tenant ID")


class RegisterUserRequest(BaseModel):
    user_id: str = Field(..., description="New user ID")
    tenant_id: str = Field(..., description="Tenant ID")
    clearance: int = Field(DEFAULT_CLEARANCE, description="Clearance level")


# ─── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    proxy = get_proxy()
    return {
        "node": "REVOCATION_PROXY",
        "status": "ready",
        "port": 8002,
        "registered_users": len(proxy.user_registry),
        "active_tenants": list(proxy.tenant_epoch_table.keys()),
        "sigma": proxy.sigma,
        "rho": proxy.rho,
    }


@app.post("/proxy/gate_subkey", response_model=GateSubkeyResponse)
def gate_subkey(req: GateSubkeyRequest) -> GateSubkeyResponse:
    proxy = get_proxy()
    t0 = time.perf_counter_ns()
    u_clearance = (
        req.user_clearance
        if req.user_clearance is not None
        else proxy.user_registry.get(req.user_id, {}).get("clearance", DEFAULT_CLEARANCE)
    )
    t_epoch = (
        req.token_epoch
        if req.token_epoch is not None
        else proxy.get_tenant_epoch(req.tenant_id)
    )

    gated_sk, is_flooded, status_msg = proxy.gate_subkey(
        user_id=req.user_id,
        tenant_id=req.tenant_id,
        token_epoch=t_epoch,
        user_clearance=u_clearance,
        doc_clearance=req.doc_clearance,
        raw_subkey=req.raw_subkey,
    )
    t1 = time.perf_counter_ns()
    elapsed_ns = t1 - t0

    return GateSubkeyResponse(
        user_id=req.user_id,
        tenant_id=req.tenant_id,
        gated_subkey=gated_sk,
        raw_subkey=req.raw_subkey,
        is_flooded=is_flooded,
        status_message=status_msg,
        gate_latency_ns=elapsed_ns,
        gate_latency_ms=elapsed_ns / 1_000_000.0,
    )


@app.post("/proxy/search_forward", response_model=SearchForwardResponse)
def proxy_search_forward(req: SearchForwardRequest) -> SearchForwardResponse:
    """
    Perimeter Revocation Proxy subkey gating & forwarder to Cloud Storage Oracle:
    1. Gates subkey in O(1) in-memory latency (~320 ns), injecting noise if revoked or epoch stale.
    2. Forwards gated subkey and lifted query vector to Cloud Storage Oracle (:8003).
    3. Returns candidates and gating status to Query Gateway (:8000).
    """
    proxy = get_proxy()
    t0 = time.perf_counter_ns()
    u_clearance = (
        req.user_clearance
        if req.user_clearance is not None
        else proxy.user_registry.get(req.user_id, {}).get("clearance", DEFAULT_CLEARANCE)
    )
    t_epoch = (
        req.token_epoch
        if req.token_epoch is not None
        else proxy.get_tenant_epoch(req.tenant_id)
    )

    gated_sk, is_flooded, status_msg = proxy.gate_subkey(
        user_id=req.user_id,
        tenant_id=req.tenant_id,
        token_epoch=t_epoch,
        user_clearance=u_clearance,
        doc_clearance=req.doc_clearance,
        raw_subkey=req.raw_subkey,
    )
    t1 = time.perf_counter_ns()
    gate_latency_ns = t1 - t0

    if is_flooded:
        print(f"\n[REVOCATION PROXY :8002] [BLOCKED] -> User '{req.user_id}' status: {status_msg}! Injected noise flooding (sigma={proxy.sigma}) in {gate_latency_ns} ns", flush=True)
    else:
        print(f"\n[REVOCATION PROXY :8002] [AUTHORIZED] -> User '{req.user_id}' ({req.tenant_id}) | Subkey gated in {gate_latency_ns} ns | Forwarding to Storage Oracle :8003", flush=True)

    # Forward to Cloud Storage Oracle
    candidates: List[Dict[str, Any]] = []
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{STORAGE_URL}/storage/search",
                json={
                    "gated_subkey": gated_sk,
                    "y_lifted": req.y_lifted,
                    "top_k": req.top_k,
                },
            )
            if resp.status_code == 200:
                candidates = resp.json().get("candidates", [])
    except Exception as ex:
        print(f"[REVOCATION PROXY] Forwarding to Storage Oracle error: {ex}", flush=True)

    return SearchForwardResponse(
        user_id=req.user_id,
        tenant_id=req.tenant_id,
        gated_subkey=gated_sk,
        is_flooded=is_flooded,
        status_message=status_msg,
        gate_latency_ns=gate_latency_ns,
        candidates=candidates,
    )


@app.post("/admin/revoke_user")
def revoke_user(req: RevokeUserRequest) -> Dict[str, Any]:
    proxy = get_proxy()
    t0 = time.perf_counter_ns()
    proxy.revoke_user(req.user_id)
    t1 = time.perf_counter_ns()
    elapsed_ns = t1 - t0
    print(f"\n[REVOCATION PROXY :8002] [ADMIN REVOCATION] -> User '{req.user_id}' revoked in {elapsed_ns} ns (~{elapsed_ns/1000.0:.2f} us)! Re-encryptions: 0", flush=True)

    return {
        "user_id": req.user_id,
        "is_revoked": True,
        "revocation_latency_ns": elapsed_ns,
        "revocation_latency_ms": elapsed_ns / 1_000_000.0,
        "ciphertexts_re_encrypted": 0,
        "status": "REVOKED",
    }


@app.post("/admin/reinstate_user")
def reinstate_user(req: ReinstateUserRequest) -> Dict[str, Any]:
    proxy = get_proxy()
    t0 = time.perf_counter_ns()
    proxy.reinstate_user(req.user_id)
    t1 = time.perf_counter_ns()
    elapsed_ns = t1 - t0
    print(f"\n[REVOCATION PROXY :8002] [ADMIN REINSTATE] -> User '{req.user_id}' reinstated in {elapsed_ns} ns! Status restored.", flush=True)

    return {
        "user_id": req.user_id,
        "is_revoked": False,
        "reinstatement_latency_ns": elapsed_ns,
        "reinstatement_latency_ms": elapsed_ns / 1_000_000.0,
        "ciphertexts_re_encrypted": 0,
        "status": "REINSTATED",
    }


@app.post("/admin/advance_epoch")
def advance_epoch(req: AdvanceEpochRequest) -> Dict[str, Any]:
    proxy = get_proxy()
    t0 = time.perf_counter_ns()
    new_epoch = proxy.advance_tenant_epoch(req.tenant_id)
    t1 = time.perf_counter_ns()
    elapsed_ns = t1 - t0

    return {
        "tenant_id": req.tenant_id,
        "new_epoch": new_epoch,
        "epoch_advance_latency_ns": elapsed_ns,
        "epoch_advance_latency_ms": elapsed_ns / 1_000_000.0,
        "ciphertexts_re_encrypted": 0,
        "status": "EPOCH_ADVANCED",
    }


@app.post("/admin/register_user")
def register_user(req: RegisterUserRequest) -> Dict[str, Any]:
    proxy = get_proxy()
    proxy.register_user(req.user_id, req.tenant_id, req.clearance)
    return {
        "user_id": req.user_id,
        "tenant_id": req.tenant_id,
        "clearance": req.clearance,
        "status": "REGISTERED",
    }


@app.get("/admin/users")
def list_users() -> Dict[str, Any]:
    proxy = get_proxy()
    return proxy.user_registry


def print_banner() -> None:
    print("=" * 70)
    print(" [REVOCATION PROXY] ready on http://localhost:8002")
    print(" Role: Perimeter Revocation Proxy (Neutron Security Group Gating)")
    print(" Revocation Latency Benchmark: ~320 ns | Noise Security Parameter: sigma=40")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    print_banner()
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="warning")
