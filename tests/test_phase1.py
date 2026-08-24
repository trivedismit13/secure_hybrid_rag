"""
Tests for Phase 1 — Plaintext functional baseline.

Tests the graph store, intent classifier, anchor matcher, constrained
expansion, and answer generation independently and in integration.
"""

import pytest
import numpy as np

from shield_rag.schema.ontology import (
    GraphNode,
    GraphEdge,
    NodeType,
    RelationType,
    IntentLabel,
    RetrievedTriple,
    VALID_RELATION_SCHEMA,
)
from shield_rag.graph_store.plaintext_store import PlaintextGraphStore
from shield_rag.intent.classifier import IntentClassifier
from shield_rag.retrieval.anchor_match import AnchorMatcher, SentenceBERTEmbedder
from shield_rag.retrieval.expand import ConstrainedExpander, ExpansionConfig
from shield_rag.generation.answer import AnswerGenerator, GenerationConfig, format_triples_for_prompt


# =============================================================================
# Fixtures
# =============================================================================


def _make_node(nid: str, ntype: NodeType, text: str, emb: list[float] | None = None) -> GraphNode:
    """Helper to create a node with a deterministic embedding."""
    if emb is None:
        rng = np.random.RandomState(hash(nid) % 2**31)
        emb = rng.randn(384).tolist()
    return GraphNode(node_id=nid, node_type=ntype, text=text, embedding=emb)


@pytest.fixture
def sample_graph():
    """Create a small graph for testing multi-hop retrieval.

    Graph structure:
        Block(pump_assembly) --PartOf--> Block(main_system)
        Block(pump_assembly) --Satisfy--> Requirement(pressure_req)
        Requirement(pressure_req) --Trace--> Action(pressure_test)
        Block(pump_assembly) --HasParameter--> Parameter(max_pressure)
        Block(impeller) --PartOf--> Block(pump_assembly)
        Block(impeller) --HasParameter--> Parameter(impeller_diameter)
        Action(pressure_test) --Allocate--> Block(test_equipment)
    """
    store = PlaintextGraphStore()

    nodes = [
        _make_node("pump_assembly", NodeType.BLOCK, "Centrifugal pump assembly unit rated for industrial applications"),
        _make_node("main_system", NodeType.BLOCK, "Main pumping system for fluid transfer operations"),
        _make_node("pressure_req", NodeType.REQUIREMENT, "Pump casing must withstand 500 PSI operating pressure"),
        _make_node("pressure_test", NodeType.ACTION, "Perform hydrostatic pressure test at 750 PSI for 30 minutes"),
        _make_node("max_pressure", NodeType.PARAMETER, "Maximum operating pressure: 500 PSI"),
        _make_node("impeller", NodeType.BLOCK, "Impeller assembly with 12-inch diameter vanes"),
        _make_node("impeller_diameter", NodeType.PARAMETER, "Impeller diameter: 12 inches"),
        _make_node("test_equipment", NodeType.BLOCK, "Hydrostatic pressure testing equipment and gauges"),
    ]

    for node in nodes:
        store.add_node(node)

    edges = [
        GraphEdge(src_id="pump_assembly", dst_id="main_system", relation=RelationType.PART_OF),
        GraphEdge(src_id="pump_assembly", dst_id="pressure_req", relation=RelationType.SATISFY),
        GraphEdge(src_id="pressure_req", dst_id="pressure_test", relation=RelationType.TRACE),
        GraphEdge(src_id="pump_assembly", dst_id="max_pressure", relation=RelationType.HAS_PARAMETER),
        GraphEdge(src_id="impeller", dst_id="pump_assembly", relation=RelationType.PART_OF),
        GraphEdge(src_id="impeller", dst_id="impeller_diameter", relation=RelationType.HAS_PARAMETER),
        GraphEdge(src_id="pressure_test", dst_id="test_equipment", relation=RelationType.ALLOCATE),
    ]

    for edge in edges:
        store.add_edge(edge)

    return store, nodes, edges


# =============================================================================
# PlaintextGraphStore tests
# =============================================================================


class TestPlaintextGraphStore:
    def test_add_and_get_node(self):
        store = PlaintextGraphStore()
        node = _make_node("n1", NodeType.BLOCK, "Test node")
        store.add_node(node)
        retrieved = store.get_node("n1")
        assert retrieved is not None
        assert retrieved.node_id == "n1"
        assert retrieved.node_type == NodeType.BLOCK

    def test_get_nonexistent(self):
        store = PlaintextGraphStore()
        assert store.get_node("nonexistent") is None

    def test_add_edge(self, sample_graph):
        store, nodes, edges = sample_graph
        assert store.node_count() == 8
        assert store.edge_count() == 7

    def test_edge_missing_node_raises(self):
        store = PlaintextGraphStore()
        store.add_node(_make_node("a", NodeType.BLOCK, "A"))
        with pytest.raises(KeyError):
            store.add_edge(GraphEdge(src_id="a", dst_id="missing", relation=RelationType.PART_OF))

    def test_get_neighbors_outgoing(self, sample_graph):
        store, _, _ = sample_graph
        neighbors = store.get_neighbors("pump_assembly", direction="outgoing")
        neighbor_ids = {n.node_id for n, _ in neighbors}
        assert "main_system" in neighbor_ids
        assert "pressure_req" in neighbor_ids
        assert "max_pressure" in neighbor_ids

    def test_get_neighbors_incoming(self, sample_graph):
        store, _, _ = sample_graph
        neighbors = store.get_neighbors("pump_assembly", direction="incoming")
        neighbor_ids = {n.node_id for n, _ in neighbors}
        assert "impeller" in neighbor_ids

    def test_get_neighbors_filtered(self, sample_graph):
        store, _, _ = sample_graph
        neighbors = store.get_neighbors(
            "pump_assembly",
            rel_filter={RelationType.SATISFY},
            direction="outgoing",
        )
        assert len(neighbors) == 1
        assert neighbors[0][0].node_id == "pressure_req"
        assert neighbors[0][1] == RelationType.SATISFY

    def test_type_index(self, sample_graph):
        store, _, _ = sample_graph
        blocks = store.get_nodes_by_type(NodeType.BLOCK)
        block_ids = {n.node_id for n in blocks}
        assert "pump_assembly" in block_ids
        assert "impeller" in block_ids
        assert "main_system" in block_ids
        assert "test_equipment" in block_ids
        assert len(blocks) == 4

        params = store.get_nodes_by_type(NodeType.PARAMETER)
        assert len(params) == 2

        reqs = store.get_nodes_by_type(NodeType.REQUIREMENT)
        assert len(reqs) == 1

    def test_similarity_search(self, sample_graph):
        store, nodes, _ = sample_graph
        # Search with the embedding of the first node
        query_emb = nodes[0].embedding
        results = store.similarity_search(query_emb, top_k=3)
        assert len(results) == 3
        # The first result should be the node itself (highest self-similarity)
        assert results[0][0].node_id == nodes[0].node_id
        assert results[0][1] > 0.99  # self-similarity ≈ 1.0

    def test_similarity_search_empty_store(self):
        store = PlaintextGraphStore()
        results = store.similarity_search([0.1, 0.2], top_k=5)
        assert results == []

    def test_clear(self, sample_graph):
        store, _, _ = sample_graph
        store.clear()
        assert store.node_count() == 0
        assert store.edge_count() == 0

    def test_node_degree(self, sample_graph):
        store, _, _ = sample_graph
        degree_out = store.get_node_degree("pump_assembly", direction="outgoing")
        assert degree_out == 3  # main_system, pressure_req, max_pressure
        degree_in = store.get_node_degree("pump_assembly", direction="incoming")
        assert degree_in == 1  # impeller
        degree_both = store.get_node_degree("pump_assembly", direction="both")
        assert degree_both == 4


# =============================================================================
# IntentClassifier tests
# =============================================================================


class TestIntentClassifier:
    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    def test_requirement_query(self, classifier):
        label = classifier.classify("What are the safety requirements for the pump?")
        assert label.target_type == NodeType.REQUIREMENT
        assert RelationType.SATISFY in label.allowed_relations
        assert label.confidence > 0.3

    def test_action_query(self, classifier):
        label = classifier.classify("How to replace the mechanical seal?")
        assert label.target_type == NodeType.ACTION
        assert label.confidence > 0.3

    def test_parameter_query(self, classifier):
        label = classifier.classify("What is the maximum operating pressure in PSI?")
        assert label.target_type == NodeType.PARAMETER
        assert RelationType.HAS_PARAMETER in label.allowed_relations

    def test_block_query(self, classifier):
        label = classifier.classify("Describe the impeller assembly component")
        assert label.target_type == NodeType.BLOCK

    def test_comparison_query(self, classifier):
        label = classifier.classify("Is the operating pressure above 400 PSI?")
        # Should detect comparison pattern
        assert label.confidence > 0.3
        # Should include parameter-related relations
        assert RelationType.HAS_PARAMETER in label.allowed_relations

    def test_unknown_query(self, classifier):
        label = classifier.classify("xyzzy foobar baz")
        # Should default to Block with all relations
        assert label.target_type == NodeType.BLOCK
        assert len(label.allowed_relations) == len(RelationType)
        assert label.confidence == 0.3

    def test_expansion_depth(self, classifier):
        label, hops = classifier.classify_with_expansion_depth(
            "Is the pump pressure rating above the requirement?"
        )
        assert hops >= 1
        assert hops <= 3


# =============================================================================
# AnchorMatcher tests
# =============================================================================


class TestAnchorMatcher:
    def test_find_anchors(self, sample_graph):
        store, _, _ = sample_graph
        matcher = AnchorMatcher(store, top_k=3)
        anchors = matcher.find_anchors("pump pressure rating")
        assert len(anchors) <= 3
        assert all(isinstance(n, GraphNode) for n, _ in anchors)
        assert all(isinstance(s, float) for _, s in anchors)

    def test_find_anchors_with_type_filter(self, sample_graph):
        store, _, _ = sample_graph
        matcher = AnchorMatcher(store, top_k=5)
        anchors = matcher.find_anchors("pressure", type_filter=NodeType.PARAMETER)
        for node, _ in anchors:
            assert node.node_type == NodeType.PARAMETER

    def test_embed_corpus(self):
        nodes = [
            GraphNode(node_id="a", node_type=NodeType.BLOCK, text="Pump assembly"),
            GraphNode(node_id="b", node_type=NodeType.BLOCK, text="Motor housing"),
        ]
        embedder = SentenceBERTEmbedder()
        updated = embedder.encode(["Pump assembly", "Motor housing"])
        assert len(updated) == 2
        assert len(updated[0]) == 384  # default dimension


# =============================================================================
# ConstrainedExpander tests
# =============================================================================


class TestConstrainedExpander:
    def test_basic_expansion(self, sample_graph):
        store, nodes, _ = sample_graph
        config = ExpansionConfig(max_hops=2, similarity_threshold=0.0)
        expander = ConstrainedExpander(store, config)

        intent = IntentLabel(
            target_type=NodeType.REQUIREMENT,
            allowed_relations={RelationType.SATISFY, RelationType.TRACE},
            confidence=0.9,
        )

        # Start from pump_assembly
        anchors = [(store.get_node("pump_assembly"), 0.9)]
        query_emb = nodes[0].embedding  # pump_assembly embedding
        triples = expander.expand(anchors, intent, query_emb)

        assert len(triples) > 0
        # Should find pump_assembly --Satisfy--> pressure_req
        relations_found = {t.relation for t in triples}
        assert RelationType.SATISFY in relations_found

    def test_relation_filtering(self, sample_graph):
        store, nodes, _ = sample_graph
        config = ExpansionConfig(max_hops=1, similarity_threshold=0.0)
        expander = ConstrainedExpander(store, config)

        # Only allow PartOf
        intent = IntentLabel(
            target_type=NodeType.BLOCK,
            allowed_relations={RelationType.PART_OF},
            confidence=0.9,
        )

        anchors = [(store.get_node("pump_assembly"), 0.9)]
        triples = expander.expand(anchors, intent, nodes[0].embedding)

        for triple in triples:
            assert triple.relation == RelationType.PART_OF

    def test_empty_anchors(self, sample_graph):
        store, _, _ = sample_graph
        expander = ConstrainedExpander(store)
        intent = IntentLabel(
            target_type=NodeType.BLOCK,
            allowed_relations=set(RelationType),
        )
        triples = expander.expand([], intent, [0.0] * 384)
        assert triples == []

    def test_multi_hop(self, sample_graph):
        store, nodes, _ = sample_graph
        config = ExpansionConfig(max_hops=3, similarity_threshold=0.0, max_triples=20)
        expander = ConstrainedExpander(store, config)

        intent = IntentLabel(
            target_type=NodeType.BLOCK,
            allowed_relations=set(RelationType),  # all relations
            confidence=0.9,
        )

        anchors = [(store.get_node("impeller"), 0.9)]
        triples = expander.expand(anchors, intent, nodes[5].embedding)

        # Should find multi-hop chains:
        # impeller --PartOf--> pump_assembly --Satisfy--> pressure_req etc.
        assert len(triples) >= 2

    def test_max_triples_limit(self, sample_graph):
        store, nodes, _ = sample_graph
        config = ExpansionConfig(max_hops=3, similarity_threshold=0.0, max_triples=2)
        expander = ConstrainedExpander(store, config)

        intent = IntentLabel(
            target_type=NodeType.BLOCK,
            allowed_relations=set(RelationType),
        )

        anchors = [(store.get_node("pump_assembly"), 0.9)]
        triples = expander.expand(anchors, intent, nodes[0].embedding)

        assert len(triples) <= 2


# =============================================================================
# AnswerGenerator tests
# =============================================================================


class TestAnswerGenerator:
    def test_format_triples(self):
        head = GraphNode(text="Pump Assembly", node_type=NodeType.BLOCK)
        tail = GraphNode(text="500 PSI rating", node_type=NodeType.REQUIREMENT)
        triples = [
            RetrievedTriple(head=head, relation=RelationType.SATISFY, tail=tail, score=0.92),
        ]
        text = format_triples_for_prompt(triples)
        assert "[0]" in text
        assert "Pump Assembly" in text
        assert "Satisfy" in text
        assert "0.920" in text

    def test_extractive_generator(self):
        """Test the extractive baseline (no GPU needed)."""
        config = GenerationConfig(use_local_model=False)
        gen = AnswerGenerator(config)

        head = GraphNode(text="Pump casing rated at 500 PSI", node_type=NodeType.BLOCK)
        tail = GraphNode(text="Maximum pressure 500 PSI", node_type=NodeType.PARAMETER)
        triples = [
            RetrievedTriple(head=head, relation=RelationType.HAS_PARAMETER, tail=tail, score=0.9),
        ]

        result = gen.generate(
            "Is the maximum operating pressure above 400 PSI?",
            triples,
        )

        assert result.verdict in ("TRUE", "FALSE", "INSUFFICIENT_EVIDENCE", None)
        assert result.latency_ms >= 0
        assert result.query != ""

    def test_extractive_true_false(self):
        """Test extractive baseline correctly evaluates numeric comparisons."""
        config = GenerationConfig(use_local_model=False)
        gen = AnswerGenerator(config)

        head = GraphNode(text="Pump Assembly", node_type=NodeType.BLOCK)
        tail = GraphNode(text="Operating pressure: 500 PSI", node_type=NodeType.PARAMETER)
        triples = [
            RetrievedTriple(head=head, relation=RelationType.HAS_PARAMETER, tail=tail, score=0.9),
        ]

        # 500 > 400 → TRUE
        result = gen.generate("Is the operating pressure above 400 PSI?", triples)
        assert result.verdict == "TRUE"

        # 500 > 600 → FALSE
        result = gen.generate("Is the operating pressure above 600 PSI?", triples)
        assert result.verdict == "FALSE"

    def test_answer_result_to_dict(self):
        from shield_rag.generation.answer import AnswerResult
        result = AnswerResult(
            query="test",
            verdict="TRUE",
            confidence=0.9,
        )
        d = result.to_dict()
        assert d["verdict"] == "TRUE"
        assert d["confidence"] == 0.9


# =============================================================================
# Integration test — full pipeline
# =============================================================================


class TestPhase1Integration:
    def test_end_to_end(self, sample_graph):
        """Run the full Phase 1 pipeline on the sample graph."""
        store, nodes, _ = sample_graph

        # 1. Classify intent
        classifier = IntentClassifier()
        intent = classifier.classify("Is the pump pressure rating above 400 PSI?")
        assert intent.target_type in (NodeType.PARAMETER, NodeType.REQUIREMENT)

        # 2. Find anchors
        matcher = AnchorMatcher(store, top_k=3)
        anchors = matcher.find_anchors("pump pressure rating 400 PSI")
        assert len(anchors) > 0

        # 3. Expand
        config = ExpansionConfig(max_hops=2, similarity_threshold=0.0)
        expander = ConstrainedExpander(store, config)
        query_emb = matcher._embedder.encode_single("pump pressure rating 400 PSI")
        triples = expander.expand(anchors, intent, query_emb)
        assert len(triples) > 0

        # 4. Generate answer
        gen_config = GenerationConfig(use_local_model=False)
        generator = AnswerGenerator(gen_config)
        result = generator.generate(
            "Is the pump pressure rating above 400 PSI?",
            triples,
        )
        assert result.query != ""
        assert result.latency_ms >= 0
