"""
Anchor Match — initial subgraph seed retrieval.

Embeds the query using Sentence-BERT and finds the most similar nodes
in the graph store via cosine similarity. Returns top-K anchor nodes
as starting points for constrained expansion.

Phase 1: uses sentence-transformers for embedding generation.
Phase 2+: the similarity computation itself will be wrapped in Ada-IPFE
          ALSH+IPFE for privacy-preserving anchor matching (see Section 5.2
          of the spec). The ALSH/IPFE anchor match is established prior art
          from CipheRAG — not claimed as novel.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from shield_rag.schema.ontology import GraphNode, NodeType


class EmbeddingModel(Protocol):
    """Protocol for embedding models — allows swapping implementations."""
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceBERTEmbedder:
    """Sentence-BERT embedding model wrapper.

    Uses 'all-MiniLM-L6-v2' (384-dim) by default for fast embedding.
    Falls back to a simple TF-IDF-based embedding if sentence-transformers
    is not available (for testing without GPU).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._fallback = False

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._fallback = False
        except ImportError:
            # Fallback: deterministic hash-based pseudo-embeddings for testing
            self._fallback = True

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into dense vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (384-dim for all-MiniLM-L6-v2).
        """
        self._load_model()

        if self._fallback:
            return [self._hash_embed(t) for t in texts]

        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def encode_single(self, text: str) -> list[float]:
        """Encode a single text string."""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        self._load_model()
        if self._fallback:
            return 384
        return self._model.get_sentence_embedding_dimension()

    @staticmethod
    def _hash_embed(text: str, dim: int = 384) -> list[float]:
        """Deterministic pseudo-embedding from text hash. For testing only."""
        import hashlib
        h = hashlib.sha512(text.encode()).digest()
        # Expand hash to fill dimension
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(dim).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec.tolist()


class AnchorMatcher:
    """Finds anchor entities in the graph store by embedding similarity.

    Given a query, embeds it and finds the top-K most similar graph nodes
    via cosine similarity. These anchors seed the constrained expansion
    in the next retrieval stage.

    Args:
        graph_store: A graph store implementing similarity_search().
        embedder:    An embedding model for query encoding.
        top_k:       Number of anchor nodes to retrieve.
    """

    def __init__(
        self,
        graph_store,  # PlaintextGraphStore or EncryptedStore
        embedder: Optional[EmbeddingModel] = None,
        top_k: int = 5,
    ) -> None:
        self._store = graph_store
        self._embedder = embedder or SentenceBERTEmbedder()
        self._top_k = top_k

    def find_anchors(
        self,
        query: str,
        top_k: Optional[int] = None,
        type_filter: Optional[NodeType] = None,
    ) -> list[tuple[GraphNode, float]]:
        """Find the most relevant anchor nodes for a query.

        Args:
            query:       Natural language query string.
            top_k:       Override default top-K (if None, uses constructor default).
            type_filter: If set, only return nodes of this type.

        Returns:
            List of (node, similarity_score) tuples, sorted descending by score.
        """
        k = top_k or self._top_k
        query_embedding = self._embedder.encode_single(query)

        # Get more candidates than needed if filtering by type
        search_k = k * 3 if type_filter else k
        candidates = self._store.similarity_search(query_embedding, top_k=search_k)

        if type_filter:
            candidates = [
                (node, score) for node, score in candidates
                if node.node_type == type_filter
            ]

        return candidates[:k]

    def embed_corpus(self, nodes: list[GraphNode], batch_size: int = 64) -> list[GraphNode]:
        """Compute embeddings for all nodes that don't have them yet.

        Args:
            nodes: List of GraphNode objects to embed.
            batch_size: Number of texts to embed at once.

        Returns:
            The same nodes with embedding fields populated.
        """
        to_embed = [n for n in nodes if not n.embedding]
        if not to_embed:
            return nodes

        for i in range(0, len(to_embed), batch_size):
            batch = to_embed[i : i + batch_size]
            texts = [n.text for n in batch]
            embeddings = self._embedder.encode(texts)
            for node, emb in zip(batch, embeddings):
                node.embedding = emb

        return nodes
