"""
Decoy Selection Policies for Oblivious Traversal.

Provides strategies for selecting k-1 decoys from a type cluster.
- UniformPolicy: Purely random selection.
- DistanceBalancedPolicy: Selects decoys that balance the graph-theoretic
  distance to mask the traversal depth.
"""

import random
from typing import Protocol


class DecoyPolicy(Protocol):
    def select_decoys(
        self, 
        target_token: bytes, 
        cluster_tokens: list[bytes], 
        k_decoys: int
    ) -> list[bytes]:
        ...


class UniformPolicy:
    """Selects k-1 decoys uniformly at random from the cluster pool."""
    
    def select_decoys(
        self, 
        target_token: bytes, 
        cluster_tokens: list[bytes], 
        k_decoys: int
    ) -> list[bytes]:
        pool = [t for t in cluster_tokens if t != target_token]
        num_decoys = min(len(pool), k_decoys - 1)
        if num_decoys <= 0:
            return []
        return random.sample(pool, num_decoys)


class DistanceBalancedPolicy:
    """Selects decoys considering graph properties to minimize leakage.
    
    (Phase 3 simplified version: falls back to uniform since actual distance
    requires server-side pre-computation or client-side caching).
    """
    
    def select_decoys(
        self, 
        target_token: bytes, 
        cluster_tokens: list[bytes], 
        k_decoys: int
    ) -> list[bytes]:
        # Fallback to uniform for v1 implementation
        pool = [t for t in cluster_tokens if t != target_token]
        num_decoys = min(len(pool), k_decoys - 1)
        if num_decoys <= 0:
            return []
        return random.sample(pool, num_decoys)
