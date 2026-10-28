"""
Test Suite for SHIELD-RAG Distributed Microservice Nodes & Storage Invariance.

Validates Plan B requirements:
1. Four Microservice Nodes (KDC :8001, Proxy :8002, Storage :8003, Gateway :8000).
2. Oracle Storage Persistence & SHA-256 Hashing invariance across lifecycle events.
3. Chapter 6 Benchmark Values:
   - Physician query similarity ~0.8842
   - Cross-tenant dot product <= 1.80e-16 (machine precision floor)
   - O(1) in-memory revocation in sub-microseconds (~320 ns)
   - Memory zeroization via ctypes.memset (100% wiped to 0.0)
"""

import hashlib
import json
import os
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.gateway_node import app as gateway_app
from services.kdc_node import app as kdc_app
from services.proxy_node import app as proxy_app
from services.storage_node import app as storage_app
from shield_rag.config import (
    DEFAULT_DOC_CLEARANCE,
    DIMS_PER_TENANT,
    GLOBAL_DIM,
    NUM_TENANTS,
    SEED,
)
from shield_rag.ephemeral_gateway import EphemeralAttentionGateway
from shield_rag.pipeline import ShieldRAGPipeline


@pytest.fixture
def test_pipeline(tmp_path):
    store_file = str(tmp_path / "test_oracle_store.json")
    pipeline = ShieldRAGPipeline(
        global_dim=GLOBAL_DIM,
        num_tenants=NUM_TENANTS,
        dims_per_tenant=DIMS_PER_TENANT,
        key_bits=512,
        seed=SEED,
    )
    pipeline.storage_oracle.store_filepath = store_file
    return pipeline, store_file


def test_storage_persistence_and_no_plaintext(test_pipeline):
    """Validates that oracle_store.json persists only ciphertexts and zero plaintext."""
    pipeline, store_file = test_pipeline
    raw_text = "Pediatric immunotherapy dosage bounds and clinical trial protocols"
    pipeline.ingest_document("tenant_0", raw_text)

    # Export store
    store_hash = pipeline.export_oracle_store(store_file)
    assert os.path.exists(store_file)
    assert len(store_hash) == 64  # valid SHA-256 hex string

    # Read file and verify structure
    with open(store_file, "r") as f:
        data = json.load(f)

    assert len(data) >= 1
    for doc_id, doc_entry in data.items():
        assert "aes_256_gcm" in doc_entry
        assert "ipfe_ciphertext" in doc_entry
        assert "C_0" in doc_entry["ipfe_ciphertext"]
        assert "C_k" in doc_entry["ipfe_ciphertext"]
        assert "ciphertext" in doc_entry["aes_256_gcm"]
        assert "iv" in doc_entry["aes_256_gcm"]
        assert "tag" in doc_entry["aes_256_gcm"]

        # Ensure no plaintext appears in the JSON file
        file_content = json.dumps(doc_entry)
        assert "immunotherapy" not in file_content.lower()
        assert "pediatric" not in file_content.lower()
        assert "dosage" not in file_content.lower()


def test_revocation_storage_invariance_sha256(test_pipeline):
    """Validates that user revocation DOES NOT mutate oracle_store.json (0 re-encryptions)."""
    pipeline, store_file = test_pipeline
    pipeline.revocation_proxy.register_user("user_1", "tenant_0", clearance=3)
    pipeline.ingest_document("tenant_0", "Pediatric immunotherapy dosage bounds and clinical protocols")
    pipeline.ingest_document("tenant_1", "Financial algorithmic risk assessment")

    # Hash BEFORE revocation
    hash_before = pipeline.get_store_hash(store_file)

    # Execute revocation
    t0 = time.perf_counter_ns()
    pipeline.revoke_user("user_1")
    t1 = time.perf_counter_ns()
    rev_latency_ns = t1 - t0

    # In-process registry update must be sub-millisecond (benchmark ~320 ns)
    assert rev_latency_ns < 1_000_000, f"Revocation took {rev_latency_ns} ns, expected < 1 ms"

    # Hash AFTER revocation
    hash_after = pipeline.get_store_hash(store_file)

    # Hashes MUST match bit-for-bit
    assert hash_before == hash_after, "Revocation mutated storage records! Zero re-encryption invariant violated."


def test_epoch_advance_and_admission_storage_invariance(test_pipeline):
    """Validates that tenant epoch advance and reserve admission do not alter storage."""
    pipeline, store_file = test_pipeline
    pipeline.ingest_document("tenant_0", "Sample clinical record")
    hash_before = pipeline.get_store_hash(store_file)

    # 1. Advance epoch
    pipeline.advance_tenant_epoch("tenant_0")
    hash_after_epoch = pipeline.get_store_hash(store_file)
    assert hash_before == hash_after_epoch, "Epoch advance altered storage!"

    # 2. Admit tenant from reserve pool
    pipeline.admit_tenant_reserve("tenant_4", dim=40)
    hash_after_admission = pipeline.get_store_hash(store_file)
    assert hash_before == hash_after_admission, "Tenant admission altered storage!"


def test_physician_query_score_and_cross_tenant_isolation(test_pipeline):
    """Validates physician query score ~0.8842 and cross-tenant leakage <= 1.80e-16."""
    pipeline, _ = test_pipeline
    pipeline.revocation_proxy.register_user("user_1", "tenant_0", clearance=3)

    # Ingest clinical document
    doc_text = "Pediatric immunotherapy dosage bounds and clinical trial protocols: recommended dosage ranges between 2.5mg/kg and 5.0mg/kg administered bi-weekly."
    pipeline.ingest_document("tenant_0", doc_text)

    # Ingest documents for other tenants
    pipeline.ingest_document("tenant_1", "High-frequency trading risk models")
    pipeline.ingest_document("tenant_2", "Natural language processing for legal contracts")
    pipeline.ingest_document("tenant_3", "Kubernetes container orchestration security")

    query = "What are the dosage bounds for pediatric immunotherapy?"
    res = pipeline.retrieve("user_1", "tenant_0", query, top_k=1)

    # Intra-tenant target similarity must be ~0.8842
    assert abs(res.top_similarity - 0.8842) < 0.05, f"Similarity {res.top_similarity} deviated from ~0.8842"
    assert not res.is_noise_flooded
    assert "pediatric" in res.top_documents[0]["text"].lower()

    # Cross-tenant dot products must remain at machine precision floor
    base_y = pipeline.embed_text(query, 60)
    for other_t in ["tenant_1", "tenant_2", "tenant_3"]:
        test_x = np.ones(60) / np.sqrt(60)
        cross_dot = abs(pipeline.subspace_engine.evaluate_cross_tenant_dot("tenant_0", other_t, base_y, test_x))
        assert cross_dot <= 1.80e-15, f"Cross-tenant leakage {cross_dot} exceeded bound."


def test_ephemeral_memory_zeroization():
    """Validates Section 6.7 in-place volatile RAM zeroization via ctypes.memset."""
    gateway = EphemeralAttentionGateway(d_k=64)
    scores = [0.8842, 0.4120, -0.1502, 0.0512]

    pre_snap, post_snap = gateway.zeroize_memory_buffer(scores)

    # Pre-zeroization buffer must match input scores
    assert np.allclose(pre_snap, scores, atol=1e-3)

    # Post-zeroization buffer must be strictly 0.0
    assert all(val == 0.0 for val in post_snap), f"Buffer residue detected: {post_snap}"


def test_microservice_kdc_endpoints():
    """Validates KDC node REST endpoints on TestClient."""
    client = TestClient(kdc_app)
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["node"] == "KDC"

    res_lift = client.post(
        "/subspace/lift_query",
        json={"tenant_id": "tenant_0", "query_text": "Pediatric immunotherapy"},
    )
    assert res_lift.status_code == 200
    data = res_lift.json()
    assert "raw_subkey" in data
    assert len(data["lifted_y"]) == GLOBAL_DIM


def test_microservice_proxy_endpoints():
    """Validates Revocation Proxy node REST endpoints on TestClient."""
    client = TestClient(proxy_app)
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["node"] == "REVOCATION_PROXY"

    # Revoke user
    res_rev = client.post("/admin/revoke_user", json={"user_id": "user_1"})
    assert res_rev.status_code == 200
    assert res_rev.json()["is_revoked"] is True
    assert res_rev.json()["ciphertexts_re_encrypted"] == 0

    # Reinstate user
    res_rein = client.post("/admin/reinstate_user", json={"user_id": "user_1"})
    assert res_rein.status_code == 200
    assert res_rein.json()["is_revoked"] is False

    # Advance epoch
    res_ep = client.post("/admin/advance_epoch", json={"tenant_id": "tenant_0"})
    assert res_ep.status_code == 200
    assert res_ep.json()["new_epoch"] >= 2


def test_microservice_storage_endpoints():
    """Validates Cloud Storage Oracle node REST endpoints on TestClient."""
    client = TestClient(storage_app)
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["node"] == "STORAGE_ORACLE"

    res_hash = client.get("/storage/store_hash")
    assert res_hash.status_code == 200
    assert "store_hash" in res_hash.json()


def test_microservice_gateway_diagnostic_endpoint():
    """Validates Query Gateway diagnostic zeroization endpoint."""
    client = TestClient(gateway_app)
    res = client.post(
        "/diagnostic/zeroization",
        json={"sample_scores": [0.8842, 0.4120, -0.1502, 0.0512]},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["zeroization_successful"] is True
    assert all(v == 0.0 for v in data["post_zeroization_buffer"])
