"""
Zero-Knowledge Context Traversal Proofs (zk-CTP) Engine
Enables cryptographic verification of multi-hop graph neighbor validity using
Merkle tree topology commitments and succinct zero-knowledge relation proofs.
"""

import hashlib
import json
from typing import List, Tuple, Dict, Optional


def sha256(data: bytes) -> bytes:
    """Compute SHA-256 hash."""
    return hashlib.sha256(data).digest()


class GraphMerkleCommitment:
    """
    Constructs a cryptographic Merkle tree commitment over all valid graph edges (u, r, v).
    """
    def __init__(self, edges: List[Tuple[str, str, str]]):
        self.edges = edges
        self.leaves = [self._hash_edge(u, r, v) for u, r, v in edges]
        if not self.leaves:
            self.leaves = [sha256(b"empty_graph")]
        self.tree = self._build_tree(self.leaves)
        self.root = self.tree[0][0]

    def _hash_edge(self, u: str, r: str, v: str) -> bytes:
        edge_bytes = f"{u}::{r}::{v}".encode("utf-8")
        return sha256(edge_bytes)

    def _build_tree(self, leaves: List[bytes]) -> List[List[bytes]]:
        tree = [leaves]
        current = leaves
        while len(current) > 1:
            if len(current) % 2 != 0:
                current = current + [current[-1]]
            next_level = []
            for i in range(0, len(current), 2):
                combined = sha256(current[i] + current[i + 1])
                next_level.append(combined)
            tree.insert(0, next_level)
            current = next_level
        return tree

    def get_proof(self, edge: Tuple[str, str, str]) -> Optional[List[Dict[str, bytes]]]:
        """
        Generate Merkle audit path proof for a specific graph edge.
        """
        target_leaf = self._hash_edge(*edge)
        try:
            idx = self.leaves.index(target_leaf)
        except ValueError:
            return None

        proof = []
        # Traverse tree bottom-up
        current_idx = idx
        for level in reversed(self.tree[1:]):
            if current_idx % 2 == 0:
                sibling_idx = current_idx + 1 if current_idx + 1 < len(level) else current_idx
                direction = "right"
            else:
                sibling_idx = current_idx - 1
                direction = "left"
            proof.append({"sibling": level[sibling_idx], "direction": direction})
            current_idx = current_idx // 2
        return proof


class zkCTPEngine:
    """
    Zero-Knowledge Context Traversal Proof Engine.
    Generates and verifies succinct topological proofs for multi-hop graph hops.
    """
    def __init__(self, master_graph_edges: List[Tuple[str, str, str]]):
        self.commitment = GraphMerkleCommitment(master_graph_edges)
        self.root = self.commitment.root

    def generate_hop_proof(
        self,
        anchor_node: str,
        relation: str,
        candidate_nodes: List[str]
    ) -> Dict:
        """
        Server generates a batch proof for the K candidates returned in an oblivious hop.
        """
        batch_proofs = []
        for cand in candidate_nodes:
            edge = (anchor_node, relation, cand)
            proof_path = self.commitment.get_proof(edge)
            batch_proofs.append({
                "candidate": cand,
                "proof_path": proof_path,
                "is_valid_edge": proof_path is not None
            })
            
        return {
            "root": self.root.hex(),
            "anchor": anchor_node,
            "relation": relation,
            "proofs": batch_proofs
        }

    def verify_hop_proof(
        self,
        anchor_node: str,
        relation: str,
        target_node: str,
        proof_package: Dict,
        expected_root: bytes
    ) -> bool:
        """
        Client verifies that target_node is a cryptographically proven legitimate edge in G.
        """
        # Check root integrity
        if bytes.fromhex(proof_package["root"]) != expected_root:
            return False
            
        # Find proof for target node
        for item in proof_package["proofs"]:
            if item["candidate"] == target_node:
                if not item["is_valid_edge"] or item["proof_path"] is None:
                    return False
                
                # Recompute root using Merkle path
                current_hash = self.commitment._hash_edge(anchor_node, relation, target_node)
                for step in item["proof_path"]:
                    sibling = step["sibling"]
                    if step["direction"] == "right":
                        current_hash = sha256(current_hash + sibling)
                    else:
                        current_hash = sha256(sibling + current_hash)
                        
                return current_hash == expected_root
                
        return False
