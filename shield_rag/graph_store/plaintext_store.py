"""
Plaintext Graph Store — Phase 1 baseline.

In-memory adjacency-list graph store with the SAME interface that the
encrypted store (Component B, Phase 2) will expose. Method signatures
are frozen from this point forward to ensure the encrypted version is
a drop-in replacement.

This module has NO novelty claim — it is scaffolding / baseline.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from shield_rag.schema.ontology import (
    VALID_RELATION_SCHEMA,
    GraphEdge,
    GraphNode,
    NodeType,
    RelationType,
)


class GraphStoreInterface(Protocol):
    """Protocol defining the interface shared by plaintext and encrypted stores.

    This interface is FROZEN — the encrypted store in Phase 2 must implement
    exactly these methods. Do not add methods without logging the change.
    """

    def add_node(self, node: GraphNode) -> None: ...
    def add_edge(self, edge: GraphEdge) -> None: ...
    def get_node(self, node_id: str) -> Optional[GraphNode]: ...
    def get_neighbors(
        self,
        node_id: str,
        rel_filter: Optional[set[RelationType]] = None,
        direction: str = "outgoing",
    ) -> list[tuple[GraphNode, RelationType]]: ...
    def similarity_search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[GraphNode, float]]: ...
    def get_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]: ...
    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...


class PlaintextGraphStore:
    """In-memory graph store using dict-of-dicts adjacency representation.

    Provides:
    - O(1) node lookup by ID
    - O(degree) neighbor traversal with optional relation-type filtering
    - O(n) brute-force cosine similarity search over all node embeddings
    - O(1) type-based node grouping (via type index)

    The type index is maintained eagerly on add_node() to match the
    type-cluster-id index that the encrypted store will maintain via
    the blind-index mechanism in Phase 2.
    """

    def __init__(self) -> None:
        # Core storage
        self._nodes: dict[str, GraphNode] = {}
        # Adjacency: node_id -> list of (neighbor_id, relation)
        self._adj_out: dict[str, list[tuple[str, RelationType]]] = defaultdict(list)
        self._adj_in: dict[str, list[tuple[str, RelationType]]] = defaultdict(list)
        # Type index: NodeType -> set of node_ids
        self._type_index: dict[NodeType, set[str]] = {nt: set() for nt in NodeType}
        # Cached embedding matrix for vectorized similarity search
        self._embedding_matrix: Optional[np.ndarray] = None
        self._embedding_ids: list[str] = []
        self._embeddings_dirty: bool = True

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the store. Overwrites if node_id already exists."""
        self._nodes[node.node_id] = node
        self._type_index[node.node_type].add(node.node_id)
        self._embeddings_dirty = True

    def add_edge(self, edge: GraphEdge) -> None:
        """Add a directed edge. Both src and dst nodes must already exist."""
        if edge.src_id not in self._nodes:
            raise KeyError(f"Source node {edge.src_id} not found in store")
        if edge.dst_id not in self._nodes:
            raise KeyError(f"Destination node {edge.dst_id} not found in store")
        self._adj_out[edge.src_id].append((edge.dst_id, edge.relation))
        self._adj_in[edge.dst_id].append((edge.src_id, edge.relation))

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by ID, or None if not found."""
        return self._nodes.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        rel_filter: Optional[set[RelationType]] = None,
        direction: str = "outgoing",
    ) -> list[tuple[GraphNode, RelationType]]:
        """Get neighbors of a node, optionally filtered by relation type.

        Args:
            node_id:    The node to get neighbors for.
            rel_filter: If provided, only return neighbors connected by these relation types.
            direction:  'outgoing', 'incoming', or 'both'.

        Returns:
            List of (neighbor_node, relation_type) tuples.
        """
        results: list[tuple[GraphNode, RelationType]] = []

        if direction in ("outgoing", "both"):
            for neighbor_id, rel in self._adj_out.get(node_id, []):
                if rel_filter is None or rel in rel_filter:
                    neighbor = self._nodes.get(neighbor_id)
                    if neighbor:
                        results.append((neighbor, rel))

        if direction in ("incoming", "both"):
            for neighbor_id, rel in self._adj_in.get(node_id, []):
                if rel_filter is None or rel in rel_filter:
                    neighbor = self._nodes.get(neighbor_id)
                    if neighbor:
                        results.append((neighbor, rel))

        return results

    def get_edges(self, node_id: str, direction: str = "outgoing") -> list[GraphEdge]:
        """Get all edges for a node as GraphEdge objects."""
        edges: list[GraphEdge] = []
        if direction in ("outgoing", "both"):
            for dst_id, rel in self._adj_out.get(node_id, []):
                edges.append(GraphEdge(src_id=node_id, dst_id=dst_id, relation=rel))
        if direction in ("incoming", "both"):
            for src_id, rel in self._adj_in.get(node_id, []):
                edges.append(GraphEdge(src_id=src_id, dst_id=node_id, relation=rel))
        return edges

    def similarity_search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[GraphNode, float]]:
        """Brute-force cosine similarity search over all node embeddings.

        Args:
            query_embedding: Dense query vector (same dimension as node embeddings).
            top_k: Number of most similar nodes to return.

        Returns:
            List of (node, similarity_score) tuples, sorted descending by score.
        """
        self._rebuild_embedding_matrix()

        if self._embedding_matrix is None or len(self._embedding_ids) == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return []
        query = query / query_norm

        # Cosine similarity: dot product of normalized vectors
        norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = self._embedding_matrix / norms
        similarities = normalized @ query

        # Top-k
        k = min(top_k, len(similarities))
        if k <= 0:
            return []
        if k >= len(similarities):
            # All elements requested — just sort
            top_indices = np.argsort(-similarities)
        else:
            top_indices = np.argpartition(-similarities, k)[:k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

        results = []
        for idx in top_indices:
            node_id = self._embedding_ids[idx]
            node = self._nodes.get(node_id)
            if node:
                results.append((node, float(similarities[idx])))
        return results

    def get_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        """Get all nodes of a specific type. Uses the type index for O(1) lookup."""
        return [
            self._nodes[nid]
            for nid in self._type_index.get(node_type, set())
            if nid in self._nodes
        ]

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return sum(len(adj) for adj in self._adj_out.values())

    def get_all_nodes(self) -> list[GraphNode]:
        """Return all nodes in the store."""
        return list(self._nodes.values())

    def get_all_edges(self) -> list[GraphEdge]:
        """Return all edges in the store."""
        edges = []
        for src_id, adj_list in self._adj_out.items():
            for dst_id, rel in adj_list:
                edges.append(GraphEdge(src_id=src_id, dst_id=dst_id, relation=rel))
        return edges

    def get_node_degree(self, node_id: str, direction: str = "both") -> int:
        """Get the degree of a node."""
        degree = 0
        if direction in ("outgoing", "both"):
            degree += len(self._adj_out.get(node_id, []))
        if direction in ("incoming", "both"):
            degree += len(self._adj_in.get(node_id, []))
        return degree

    def _rebuild_embedding_matrix(self) -> None:
        """Rebuild the cached embedding matrix if nodes have changed."""
        if not self._embeddings_dirty:
            return

        ids = []
        embeddings = []
        for node_id, node in self._nodes.items():
            if node.embedding and len(node.embedding) > 0:
                ids.append(node_id)
                embeddings.append(node.embedding)

        if embeddings:
            self._embedding_matrix = np.array(embeddings, dtype=np.float32)
            self._embedding_ids = ids
        else:
            self._embedding_matrix = None
            self._embedding_ids = []

        self._embeddings_dirty = False

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._nodes.clear()
        self._adj_out.clear()
        self._adj_in.clear()
        for nt in NodeType:
            self._type_index[nt] = set()
        self._embedding_matrix = None
        self._embedding_ids = []
        self._embeddings_dirty = True
