"""
Boundary-Preserving Re-verification (Component E).

When the orchestrator detects high uncertainty or a consensus failure mode 
(high consistency gap, high ECE), this module is triggered to backtrack and 
re-verify the specific claims via targeted sub-queries to the encrypted store.
"""

from typing import List, Tuple, Any
from shield_rag.schema.ontology import IntentLabel, NodeType
from shield_rag.schema.wire import EncryptedBucket
from shield_rag.oblivious_traversal.engine import ObliviousTraversalEngine
from shield_rag.crypto.ada_ipfe import FunctionalKey, AdaIPFE


class Reverifier:
    """Handles falling back to the knowledge graph when generation is uncertain."""
    
    def __init__(self, traversal_engine: ObliviousTraversalEngine, ipfe: AdaIPFE):
        self.engine = traversal_engine
        self.ipfe = ipfe

    def requires_reverification(self, ece: float, ocr: float, cg: float) -> bool:
        """
        Policy to determine if re-verification is needed based on trust metrics.
        Thresholds are heuristic for Phase 5.
        """
        # High ECE indicates poor calibration
        if ece > 0.15:
            return True
            
        # High OCR indicates confident hallucinations
        if ocr > 0.10:
            return True
            
        # High Consistency Gap (fluctuating answers)
        if cg > 0.30:
            return True
            
        return False

    def execute_reverification(
        self, 
        anchor_buckets: List[EncryptedBucket], 
        fallback_intent: IntentLabel,
        query_func_key: FunctionalKey,
        max_hops: int = 1
    ) -> Tuple[bool, List[EncryptedBucket]]:
        """
        Executes a targeted re-traversal to gather missing context.
        
        Args:
            anchor_buckets: The nodes at the boundary where uncertainty occurred.
            fallback_intent: A broadened intent to fetch supporting/contradicting context.
            query_func_key: The original or paraphrased query key.
            max_hops: Usually 1 for local re-verification.
            
        Returns:
            Tuple of (success, additional_buckets_fetched)
        """
        # Execute the oblivious traversal with the broadened intent
        collected, hop_results = self.engine.orchestrate(
            anchors=anchor_buckets,
            intent=fallback_intent,
            query_func_key=query_func_key,
            max_hops=max_hops,
            similarity_threshold=0.0  # Fetch everything along this relation
        )
        
        # If we fetched new buckets (beyond the anchors), re-verification succeeded 
        # in finding new context.
        new_buckets = [b for b in collected if b not in anchor_buckets]
        return len(new_buckets) > 0, new_buckets
