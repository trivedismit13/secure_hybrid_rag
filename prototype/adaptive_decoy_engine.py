# =========================================================================
# Step 6.1 Verification - Exact References in rag_pipeline.py
# =========================================================================
# Output of grep for decoy/traversal in rag_pipeline.py:
# 13: from zk_traversal_proof import TraversalProof
# 61: self.audit_proofs: Dict[str, TraversalProof] = {}
# 243: def execute_bounded_decoy_traversal(
# 246:     decoy_cids: List[str],
# 249: ) -> Tuple[str, Optional[TraversalProof]]:
# 252: # 1. Shuffles target_cid among K-1 decoy_cids.
# 258: candidates = [target_cid] + list(decoy_cids)
# 259: random.shuffle(candidates)
# 
# (a) Where K is currently set: K = 1 + len(decoy_cids) in execute_bounded_decoy_traversal.
# (b) How decoys are selected: Passed as a list of same-relation-type decoy node CIDs.
# =========================================================================

import math
import random
import time
from typing import Dict, List, Tuple, Optional, Any, Union


class InsufficientAnonymitySetError(Exception):
    """
    Raised when the available number of same-relation-type candidates
    in the local graph is strictly less than min_k, preventing a false
    or degraded anonymity guarantee.
    """
    pass


class KnowledgeGraphIndex:
    """
    In-memory relation-aware graph and cluster index.
    Maps node_id -> {relation_type: [connected_neighbor_ids]},
    and relation_type -> [all_nodes_possessing_relation].
    """
    def __init__(self):
        # node_id -> dict of relation -> list of target nodes
        self.adj: Dict[str, Dict[str, List[str]]] = {}
        # relation_type -> set of all node IDs
        self.relation_clusters: Dict[str, List[str]] = {}

    def add_edge(self, src: str, relation: str, dst: str):
        if src not in self.adj:
            self.adj[src] = {}
        if relation not in self.adj[src]:
            self.adj[src][relation] = []
        self.adj[src][relation].append(dst)

        if relation not in self.relation_clusters:
            self.relation_clusters[relation] = []
        if src not in self.relation_clusters[relation]:
            self.relation_clusters[relation].append(src)
        if dst not in self.relation_clusters[relation]:
            self.relation_clusters[relation].append(dst)


def compute_adaptive_k(
    graph: Union[KnowledgeGraphIndex, Dict[str, Any]],
    node_id: str,
    relation_type: str,
    min_k: int = 3,
    max_k: Optional[int] = None
) -> int:
    """
    Computes graph-degree-aware adaptive anonymity parameter K.
    Counts available same-relation-type candidate nodes at this hop.
    
    If available < min_k: raises InsufficientAnonymitySetError to prevent
    false privacy claims.
    If max_k is provided: K = min(available, max_k) to bound decryption latency.
    """
    available = 0
    if isinstance(graph, KnowledgeGraphIndex):
        candidates = graph.relation_clusters.get(relation_type, [])
        # Exclude self from decoy candidates
        other_candidates = [n for n in candidates if n != node_id]
        available = 1 + len(other_candidates)  # 1 (target) + decoys
    elif isinstance(graph, dict):
        if "clusters" in graph:
            candidates = graph["clusters"].get(relation_type, [])
            other_candidates = [n for n in candidates if n != node_id]
            available = 1 + len(other_candidates)
        elif node_id in graph and relation_type in graph[node_id]:
            available = 1 + len(graph[node_id][relation_type])
        else:
            available = 0
    else:
        available = 0

    if available < min_k:
        raise InsufficientAnonymitySetError(
            f"Node '{node_id}' has only {available} same-relation '{relation_type}' candidates, "
            f"which is below the required min_k={min_k} anonymity threshold."
        )

    if max_k is not None:
        return min(available, max_k)
    return available


def select_adaptive_decoys(
    graph: Union[KnowledgeGraphIndex, Dict[str, Any]],
    target_node_id: str,
    relation_type: str,
    min_k: int = 3,
    max_k: Optional[int] = None
) -> Tuple[int, List[str]]:
    """
    Selects K-1 random decoys sharing relation_type with target_node_id.
    Returns (K, decoy_list).
    """
    k = compute_adaptive_k(graph, target_node_id, relation_type, min_k=min_k, max_k=max_k)
    
    if isinstance(graph, KnowledgeGraphIndex):
        candidates = [n for n in graph.relation_clusters.get(relation_type, []) if n != target_node_id]
    elif isinstance(graph, dict) and "clusters" in graph:
        candidates = [n for n in graph["clusters"].get(relation_type, []) if n != target_node_id]
    else:
        candidates = []

    # Sample exactly K-1 decoys
    num_decoys_needed = k - 1
    if len(candidates) >= num_decoys_needed:
        selected_decoys = random.sample(candidates, num_decoys_needed)
    else:
        selected_decoys = list(candidates)
        
    return k, selected_decoys


def verify_k_anonymity_bound(
    k_used: int,
    num_distinguishing_queries: int,
    observed_correct_guesses: int
) -> Dict[str, Any]:
    """
    Validates the theoretical k-anonymity guarantee:
    Theoretical guess rate: 1 / k_used.
    Checks if empirical_rate - theoretical_rate <= 95% binomial confidence interval.
    """
    empirical_rate = observed_correct_guesses / num_distinguishing_queries
    theoretical_rate = 1.0 / k_used
    excess = empirical_rate - theoretical_rate
    
    # 95% normal approximation confidence interval half-width
    ci_half_width = 1.96 * math.sqrt((theoretical_rate * (1.0 - theoretical_rate)) / num_distinguishing_queries)
    within_bound = bool(abs(excess) <= ci_half_width)

    return {
        "k_used": k_used,
        "num_queries": num_distinguishing_queries,
        "observed_correct_guesses": observed_correct_guesses,
        "empirical_rate": empirical_rate,
        "theoretical_rate": theoretical_rate,
        "excess": excess,
        "ci_half_width": ci_half_width,
        "within_bound": within_bound
    }


def simulate_oracle_guessing_attack(
    traversal_records: List[Dict[str, Any]],
    strategy: str = "response_time"
) -> List[str]:
    """
    Simulates honest-but-curious Oracle guessing attacks against K candidates:
    - 'response_time': Picks candidate with highest measured decryption response latency.
    - 'random': Picks candidate uniformly at random from the K candidates.
    """
    oracle_guesses = []
    
    for record in traversal_records:
        candidates = record["candidates"]  # List of candidate node IDs
        
        if strategy == "response_time" and "candidate_timings" in record:
            timings = record["candidate_timings"]  # dict of node_id -> float latency
            # Guess the candidate with the highest response latency
            best_cand = max(candidates, key=lambda c: timings.get(c, 0.0))
            oracle_guesses.append(best_cand)
        else:
            # Fallback uniform random guess
            oracle_guesses.append(random.choice(candidates))
            
    return oracle_guesses
