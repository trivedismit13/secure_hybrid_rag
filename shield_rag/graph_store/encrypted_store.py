"""
Encrypted Graph Store (Component B).

Honest-but-curious server-side storage. Stores EncryptedBucket objects
indexed by PRF tokens. Maintains a blind index (type_cluster_id) for
decoy pool selection, without knowing the plaintext types.
"""

from typing import Optional
from collections import defaultdict

from shield_rag.schema.wire import EncryptedBucket
from shield_rag.crypto.type_tag_cipher import TypeTagCipher


class EncryptedStore:
    """Server-side encrypted knowledge graph store.

    Maintains buckets addressed by PRF tokens, and a blind type-cluster index
    for rapid decoy sampling.
    """

    def __init__(self) -> None:
        self._buckets: dict[bytes, EncryptedBucket] = {}
        # type_cluster_id (hash of type_tag_ct) -> set of tokens
        self._type_clusters: dict[bytes, set[bytes]] = defaultdict(set)

    def add_bucket(self, bucket: EncryptedBucket) -> None:
        """Add an encrypted bucket to the store."""
        self._buckets[bucket.token] = bucket
        
        # Add to blind type index
        cluster_id = TypeTagCipher.get_cluster_id(bucket.type_tag_ct)
        self._type_clusters[cluster_id].add(bucket.token)

    def fetch(self, token: bytes) -> Optional[EncryptedBucket]:
        """Fetch a single encrypted bucket by its token."""
        return self._buckets.get(token)

    def fetch_batch(self, tokens: list[bytes]) -> list[EncryptedBucket]:
        """Fetch multiple encrypted buckets.

        Returns only the buckets that exist, in the requested order if possible,
        or omits missing ones.
        """
        results = []
        for t in tokens:
            b = self._buckets.get(t)
            if b is not None:
                results.append(b)
        return results

    def get_type_cluster(self, cluster_id: bytes) -> list[bytes]:
        """Get all tokens belonging to the specified type cluster.

        Used by Component C to sample decoys of the same type as a target node.
        """
        return list(self._type_clusters.get(cluster_id, set()))

    def node_count(self) -> int:
        return len(self._buckets)

    def clear(self) -> None:
        self._buckets.clear()
        self._type_clusters.clear()
