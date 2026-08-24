"""
Secure Stopping Logic (Component C).

Hides the termination of a graph traversal branch from the server by
continuing to fetch random decoy buckets up to a pre-defined maximum
depth (max_hops) even if the client has logically stopped exploring
a branch. This prevents path length leakage.
"""

from typing import Optional
from shield_rag.schema.wire import TraversalRequest, HopResult


class SecureStopController:
    """Manages padding hops to hide true traversal depth."""
    
    def __init__(self, max_hops: int):
        self.max_hops = max_hops

    def should_continue_padding(self, current_hop: int) -> bool:
        """Determines if dummy hops should be executed to reach max depth."""
        return current_hop < self.max_hops

    def generate_dummy_request(self, hop_index: int, k_decoys: int, server_store) -> TraversalRequest:
        """Generate a request consisting entirely of decoys (no real target).
        
        Args:
            hop_index: The current hop index.
            k_decoys: Total batch size K.
            server_store: Reference to store (to sample completely random decoys).
            
        Returns:
            A TraversalRequest full of random tokens.
        """
        import random
        # Sample K tokens from anywhere in the graph since there is no real type
        # For simulation, just sample random bytes. In practice, the client requests
        # K random tokens from a randomly selected cluster.
        dummy_tokens = [os.urandom(32) for _ in range(k_decoys)]
        return TraversalRequest(hop_index=hop_index, requested_tokens=dummy_tokens)

    def process_dummy_result(self, request: TraversalRequest, buckets: list) -> HopResult:
        """Process the server's response for a dummy hop (discard everything)."""
        return HopResult(
            hop_index=request.hop_index,
            next_candidate_tokens=[],
            decoy_count=len(request.requested_tokens)
        )
