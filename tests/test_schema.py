"""
Tests for the SHIELD-RAG frozen schema package.

Tests serialization round-trips for all wire types, enum completeness,
edge cases (empty embeddings, empty adjacency lists), and ontology constraints.
"""

import json
import os
import uuid

import pytest

from shield_rag.schema.ontology import (
    VALID_RELATION_SCHEMA,
    GraphEdge,
    GraphNode,
    IntentLabel,
    NodeType,
    RelationType,
    RetrievedTriple,
)
from shield_rag.schema.wire import (
    EncryptedBucket,
    HopResult,
    TraversalRequest,
    TraversalSession,
)


# =============================================================================
# Ontology enum tests
# =============================================================================


class TestNodeType:
    """Tests for the NodeType closed-set enum."""

    def test_all_values_present(self):
        expected = {"Requirement", "Action", "Block", "Parameter"}
        actual = {nt.value for nt in NodeType}
        assert actual == expected, f"NodeType mismatch: {actual} != {expected}"

    def test_string_enum(self):
        """NodeType members are also strings for JSON serialization."""
        for nt in NodeType:
            assert isinstance(nt, str)
            assert nt == nt.value

    def test_from_string(self):
        assert NodeType("Requirement") == NodeType.REQUIREMENT
        assert NodeType("Action") == NodeType.ACTION
        assert NodeType("Block") == NodeType.BLOCK
        assert NodeType("Parameter") == NodeType.PARAMETER

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            NodeType("InvalidType")


class TestRelationType:
    """Tests for the RelationType closed-set enum."""

    def test_all_values_present(self):
        expected = {"Satisfy", "Trace", "Allocate", "HasParameter", "PartOf"}
        actual = {rt.value for rt in RelationType}
        assert actual == expected

    def test_string_enum(self):
        for rt in RelationType:
            assert isinstance(rt, str)

    def test_from_string(self):
        assert RelationType("Satisfy") == RelationType.SATISFY
        assert RelationType("PartOf") == RelationType.PART_OF

    def test_invalid_relation_raises(self):
        with pytest.raises(ValueError):
            RelationType("UnknownRel")


class TestValidRelationSchema:
    """Tests for the ontology constraint map."""

    def test_all_relation_types_covered(self):
        """Every RelationType has a valid (src, dst) pair defined."""
        for rt in RelationType:
            assert rt in VALID_RELATION_SCHEMA, f"{rt} missing from VALID_RELATION_SCHEMA"

    def test_schema_types_are_node_types(self):
        for rt, (src, dst) in VALID_RELATION_SCHEMA.items():
            assert isinstance(src, NodeType), f"{rt}: src is not NodeType"
            assert isinstance(dst, NodeType), f"{rt}: dst is not NodeType"

    def test_specific_constraints(self):
        assert VALID_RELATION_SCHEMA[RelationType.SATISFY] == (
            NodeType.BLOCK,
            NodeType.REQUIREMENT,
        )
        assert VALID_RELATION_SCHEMA[RelationType.TRACE] == (
            NodeType.REQUIREMENT,
            NodeType.ACTION,
        )
        assert VALID_RELATION_SCHEMA[RelationType.PART_OF] == (
            NodeType.BLOCK,
            NodeType.BLOCK,
        )


# =============================================================================
# GraphNode / GraphEdge tests
# =============================================================================


class TestGraphNode:
    def test_default_construction(self):
        node = GraphNode()
        assert isinstance(node.node_id, str)
        assert len(node.node_id) == 36  # UUID format
        assert node.node_type == NodeType.BLOCK
        assert node.text == ""
        assert node.embedding == []

    def test_custom_construction(self):
        emb = [0.1, 0.2, 0.3]
        node = GraphNode(
            node_id="test-id",
            node_type=NodeType.REQUIREMENT,
            text="Pump must withstand 500 PSI",
            embedding=emb,
        )
        assert node.node_id == "test-id"
        assert node.node_type == NodeType.REQUIREMENT
        assert node.text == "Pump must withstand 500 PSI"
        assert node.embedding == emb

    def test_string_type_coercion(self):
        """NodeType can be provided as string and gets coerced."""
        node = GraphNode(node_type="Action")
        assert node.node_type == NodeType.ACTION

    def test_unique_ids(self):
        nodes = [GraphNode() for _ in range(100)]
        ids = [n.node_id for n in nodes]
        assert len(set(ids)) == 100, "Node IDs must be unique"

    def test_empty_embedding(self):
        node = GraphNode(embedding=[])
        assert node.embedding == []

    def test_large_embedding(self):
        emb = [float(i) / 384 for i in range(384)]
        node = GraphNode(embedding=emb)
        assert len(node.embedding) == 384


class TestGraphEdge:
    def test_default_construction(self):
        edge = GraphEdge()
        assert edge.src_id == ""
        assert edge.dst_id == ""
        assert edge.relation == RelationType.PART_OF

    def test_custom_construction(self):
        edge = GraphEdge(src_id="a", dst_id="b", relation=RelationType.SATISFY)
        assert edge.src_id == "a"
        assert edge.dst_id == "b"
        assert edge.relation == RelationType.SATISFY

    def test_string_relation_coercion(self):
        edge = GraphEdge(relation="Trace")
        assert edge.relation == RelationType.TRACE


# =============================================================================
# IntentLabel / RetrievedTriple tests
# =============================================================================


class TestIntentLabel:
    def test_default(self):
        label = IntentLabel()
        assert label.target_type == NodeType.BLOCK
        assert label.allowed_relations == set()
        assert label.confidence == 0.0

    def test_with_relations(self):
        label = IntentLabel(
            target_type=NodeType.REQUIREMENT,
            allowed_relations={RelationType.SATISFY, RelationType.TRACE},
            confidence=0.95,
        )
        assert NodeType.REQUIREMENT == label.target_type
        assert len(label.allowed_relations) == 2
        assert RelationType.SATISFY in label.allowed_relations


class TestRetrievedTriple:
    def test_to_text_format(self):
        head = GraphNode(text="Pump Assembly", node_type=NodeType.BLOCK)
        tail = GraphNode(text="500 PSI rating", node_type=NodeType.REQUIREMENT)
        triple = RetrievedTriple(
            head=head,
            relation=RelationType.SATISFY,
            tail=tail,
            score=0.92,
        )
        text = triple.to_text()
        assert "Pump Assembly" in text
        assert "Satisfy" in text
        assert "500 PSI rating" in text
        assert "–Satisfy→" in text


# =============================================================================
# Wire format tests — EncryptedBucket
# =============================================================================


class TestEncryptedBucket:
    @pytest.fixture
    def sample_bucket(self):
        return EncryptedBucket(
            token=os.urandom(32),
            ciphertext=os.urandom(256),
            type_tag_ct=os.urandom(16),
            adjacency_ct=[os.urandom(32) for _ in range(5)],
        )

    def test_json_round_trip(self, sample_bucket):
        d = sample_bucket.to_dict()
        restored = EncryptedBucket.from_dict(d)
        assert restored.token == sample_bucket.token
        assert restored.ciphertext == sample_bucket.ciphertext
        assert restored.type_tag_ct == sample_bucket.type_tag_ct
        assert restored.adjacency_ct == sample_bucket.adjacency_ct

    def test_json_serializable(self, sample_bucket):
        """Dict form must be JSON-serializable (base64 strings, not raw bytes)."""
        d = sample_bucket.to_dict()
        json_str = json.dumps(d)
        restored_dict = json.loads(json_str)
        restored = EncryptedBucket.from_dict(restored_dict)
        assert restored.token == sample_bucket.token

    def test_binary_round_trip(self, sample_bucket):
        raw = sample_bucket.to_bytes()
        restored = EncryptedBucket.from_bytes(raw)
        assert restored.token == sample_bucket.token
        assert restored.ciphertext == sample_bucket.ciphertext
        assert restored.type_tag_ct == sample_bucket.type_tag_ct
        assert restored.adjacency_ct == sample_bucket.adjacency_ct

    def test_empty_bucket(self):
        bucket = EncryptedBucket()
        raw = bucket.to_bytes()
        restored = EncryptedBucket.from_bytes(raw)
        assert restored.token == b""
        assert restored.ciphertext == b""
        assert restored.adjacency_ct == []

    def test_empty_adjacency(self):
        bucket = EncryptedBucket(
            token=b"tok", ciphertext=b"ct", type_tag_ct=b"tag", adjacency_ct=[]
        )
        raw = bucket.to_bytes()
        restored = EncryptedBucket.from_bytes(raw)
        assert restored.adjacency_ct == []

    def test_large_adjacency(self):
        """Stress test: 100 neighbor tokens."""
        adj = [os.urandom(32) for _ in range(100)]
        bucket = EncryptedBucket(
            token=os.urandom(32),
            ciphertext=os.urandom(512),
            type_tag_ct=os.urandom(16),
            adjacency_ct=adj,
        )
        raw = bucket.to_bytes()
        restored = EncryptedBucket.from_bytes(raw)
        assert len(restored.adjacency_ct) == 100
        for orig, rest in zip(adj, restored.adjacency_ct):
            assert orig == rest

    def test_binary_and_json_consistency(self, sample_bucket):
        """Both serialization paths must produce equivalent results."""
        from_json = EncryptedBucket.from_dict(sample_bucket.to_dict())
        from_bin = EncryptedBucket.from_bytes(sample_bucket.to_bytes())
        assert from_json.token == from_bin.token
        assert from_json.ciphertext == from_bin.ciphertext
        assert from_json.type_tag_ct == from_bin.type_tag_ct
        assert from_json.adjacency_ct == from_bin.adjacency_ct


# =============================================================================
# Wire format tests — TraversalRequest
# =============================================================================


class TestTraversalRequest:
    @pytest.fixture
    def sample_request(self):
        return TraversalRequest(
            hop_index=3,
            requested_tokens=[os.urandom(32) for _ in range(8)],  # k=8
        )

    def test_json_round_trip(self, sample_request):
        d = sample_request.to_dict()
        restored = TraversalRequest.from_dict(d)
        assert restored.hop_index == sample_request.hop_index
        assert restored.requested_tokens == sample_request.requested_tokens

    def test_json_serializable(self, sample_request):
        d = sample_request.to_dict()
        json_str = json.dumps(d)
        restored = TraversalRequest.from_dict(json.loads(json_str))
        assert restored.hop_index == 3
        assert len(restored.requested_tokens) == 8

    def test_binary_round_trip(self, sample_request):
        raw = sample_request.to_bytes()
        restored = TraversalRequest.from_bytes(raw)
        assert restored.hop_index == sample_request.hop_index
        assert restored.requested_tokens == sample_request.requested_tokens

    def test_empty_tokens(self):
        req = TraversalRequest(hop_index=0, requested_tokens=[])
        raw = req.to_bytes()
        restored = TraversalRequest.from_bytes(raw)
        assert restored.hop_index == 0
        assert restored.requested_tokens == []

    def test_single_token(self):
        """Edge case: k=1 (no decoys)."""
        tok = os.urandom(32)
        req = TraversalRequest(hop_index=0, requested_tokens=[tok])
        restored = TraversalRequest.from_bytes(req.to_bytes())
        assert restored.requested_tokens == [tok]

    def test_binary_and_json_consistency(self, sample_request):
        from_json = TraversalRequest.from_dict(sample_request.to_dict())
        from_bin = TraversalRequest.from_bytes(sample_request.to_bytes())
        assert from_json.hop_index == from_bin.hop_index
        assert from_json.requested_tokens == from_bin.requested_tokens


# =============================================================================
# HopResult / TraversalSession tests
# =============================================================================


class TestHopResult:
    def test_default(self):
        hr = HopResult()
        assert hr.hop_index == 0
        assert hr.real_triples == []
        assert hr.next_candidate_tokens == []
        assert hr.decoy_count == 0

    def test_with_data(self):
        hr = HopResult(
            hop_index=2,
            real_triples=[{"head": "A", "relation": "Satisfy", "tail": "B"}],
            next_candidate_tokens=[b"tok1", b"tok2"],
            decoy_count=7,
        )
        assert hr.hop_index == 2
        assert len(hr.real_triples) == 1
        assert hr.decoy_count == 7


class TestTraversalSession:
    def test_properties(self):
        hops = [
            HopResult(hop_index=0, real_triples=[{"a": 1}, {"b": 2}], decoy_count=3),
            HopResult(hop_index=1, real_triples=[{"c": 3}], decoy_count=7),
        ]
        session = TraversalSession(
            session_id="sess-001", query_hash="abc123", hops=hops
        )
        assert session.hop_count == 2
        assert session.total_real_triples == 3
        assert session.total_decoys_used == 10
