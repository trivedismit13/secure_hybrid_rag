"""
Constrained Expansion — ontology-guided multi-hop BFS.

Expands from anchor nodes following only the allowed relation types
specified by the intent classifier, with a confidence threshold cutoff
based on cosine similarity to the original query.

This is the plaintext version of what Component C (Phase 3) will later
make oblivious. The traversal logic here is the reference implementation
against which Phase 3's oblivious version must produce identical results
(modulo decoy noise).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from shield_rag.schema.ontology import (
    VALID_RELATION_SCHEMA,
    GraphEdge,
    GraphNode,
    IntentLabel,
    NodeType,
    RelationType,
    RetrievedTriple,
)


@dataclass
class ExpansionConfig:
    """Configuration for constrained BFS expansion.

    Attributes:
        max_hops:            Maximum number of hops from anchor nodes.
        similarity_threshold: Minimum cosine similarity to query for a node
                             to be included in the expanded subgraph.
        max_triples:         Maximum number of triples to return.
        max_nodes_per_hop:   Maximum new nodes to expand per hop level.
    """

    max_hops: int = 3
    similarity_threshold: float = 0.15
    max_triples: int = 50
    max_nodes_per_hop: int = 20


class ConstrainedExpander:
    """BFS-based graph expansion constrained by ontology intent labels.

    Given anchor nodes and an IntentLabel, expands outward through the graph
    following only allowed relation types. Nodes are included if their
    embedding similarity to the query exceeds a threshold.

    This produces a subgraph of (head, relation, tail) triples that form
    the context for the generation stage.
    """

    def __init__(
        self,
        graph_store,  # PlaintextGraphStore (or EncryptedStore in Phase 2+)
        config: Optional[ExpansionConfig] = None,
    ) -> None:
        self._store = graph_store
        self._config = config or ExpansionConfig()

    def expand(
        self,
        anchors: list[tuple[GraphNode, float]],
        intent: IntentLabel,
        query_embedding: list[float],
    ) -> list[RetrievedTriple]:
        """Expand from anchor nodes following allowed relations.

        Args:
            anchors:         List of (anchor_node, anchor_score) from AnchorMatcher.
            intent:          IntentLabel from the classifier (constrains relations).
            query_embedding: Dense vector of the original query for similarity scoring.

        Returns:
            List of RetrievedTriple, sorted by score descending.
        """
        if not anchors:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 1e-10:
            query_vec = query_vec / query_norm

        allowed_relations = intent.allowed_relations
        if not allowed_relations:
            allowed_relations = set(RelationType)

        # BFS state
        visited_nodes: set[str] = set()
        visited_edges: set[tuple[str, str, str]] = set()  # (src, rel, dst)
        triples: list[RetrievedTriple] = []

        # Initialize BFS queue with anchor nodes
        queue: deque[tuple[str, int]] = deque()  # (node_id, current_hop)
        for node, score in anchors:
            visited_nodes.add(node.node_id)
            queue.append((node.node_id, 0))

        # BFS expansion
        while queue and len(triples) < self._config.max_triples:
            node_id, hop = queue.popleft()

            if hop >= self._config.max_hops:
                continue

            # Get neighbors via allowed relations
            neighbors = self._store.get_neighbors(
                node_id,
                rel_filter=allowed_relations,
                direction="both",
            )

            new_at_this_hop = 0
            for neighbor_node, relation in neighbors:
                # Deduplicate edges
                edge_key = (node_id, relation.value, neighbor_node.node_id)
                reverse_key = (neighbor_node.node_id, relation.value, node_id)
                if edge_key in visited_edges or reverse_key in visited_edges:
                    continue
                visited_edges.add(edge_key)

                # Check ontology type constraint
                src_node = self._store.get_node(node_id)
                if src_node is None:
                    continue

                if not self._is_valid_edge(src_node, relation, neighbor_node):
                    # Try reverse direction
                    if not self._is_valid_edge(neighbor_node, relation, src_node):
                        continue
                    # Swap to correct direction
                    src_node, neighbor_node = neighbor_node, src_node

                # Compute similarity score for the neighbor
                score = self._compute_similarity(neighbor_node, query_vec)

                if score >= self._config.similarity_threshold or hop == 0:
                    triple = RetrievedTriple(
                        head=src_node,
                        relation=relation,
                        tail=neighbor_node,
                        score=score,
                    )
                    triples.append(triple)

                    # Queue neighbor for further expansion if not visited
                    if (
                        neighbor_node.node_id not in visited_nodes
                        and new_at_this_hop < self._config.max_nodes_per_hop
                    ):
                        visited_nodes.add(neighbor_node.node_id)
                        queue.append((neighbor_node.node_id, hop + 1))
                        new_at_this_hop += 1

        # Sort by score descending and limit
        triples.sort(key=lambda t: t.score, reverse=True)
        return triples[: self._config.max_triples]

    def expand_from_node_ids(
        self,
        anchor_ids: list[str],
        intent: IntentLabel,
        query_embedding: list[float],
    ) -> list[RetrievedTriple]:
        """Convenience: expand from node IDs instead of (node, score) tuples."""
        anchors = []
        for nid in anchor_ids:
            node = self._store.get_node(nid)
            if node:
                score = self._compute_similarity(
                    node, np.array(query_embedding, dtype=np.float32)
                )
                anchors.append((node, score))
        return self.expand(anchors, intent, query_embedding)

    @staticmethod
    def _is_valid_edge(
        src: GraphNode, relation: RelationType, dst: GraphNode
    ) -> bool:
        """Check if an edge satisfies the ontology type constraint."""
        expected = VALID_RELATION_SCHEMA.get(relation)
        if expected is None:
            return False
        expected_src_type, expected_dst_type = expected
        return src.node_type == expected_src_type and dst.node_type == expected_dst_type

    @staticmethod
    def _compute_similarity(node: GraphNode, query_vec: np.ndarray) -> float:
        """Cosine similarity between a node's embedding and the query vector."""
        if not node.embedding:
            return 0.0
        node_vec = np.array(node.embedding, dtype=np.float32)
        node_norm = np.linalg.norm(node_vec)
        if node_norm < 1e-10:
            return 0.0
        return float(np.dot(node_vec / node_norm, query_vec))
