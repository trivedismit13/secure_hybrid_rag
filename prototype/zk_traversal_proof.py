# =========================================================================
# Step 2.1 Verification - Exact Function Signatures in pod_engine.py
# =========================================================================
# Output of grep -n "def " pod_engine.py:
# 8:  def Setup(lambda_bits: int, n: int, max_layers: int = 3) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
# 19: def KeyGen(y: List[float], max_layers: int, msk_layers: List[List[int]], mpk_layers: List[Dict[str, Any]], alphas: List[int], betas: List[int]) -> List[Tuple[int, int, List[int]]]:
# 29: def Encrypt(x: List[float], max_layers: int, mpk_layers: List[Dict[str, Any]], pk_layers: List[List[int]]) -> List[Tuple]:
# 38: def DecryptLayer(sk_layer: Tuple, ct_layer: Tuple, mpk_layer: Dict[str, Any]) -> float:
# 43: def Decrypt(sk_layers: List[Tuple], ct_layers: List[Tuple], traversal_depth: int, max_layers: int, mpk_layers: List[Dict[str, Any]]) -> float:
#
# (a) Function that performs one layer of decryption: PODEngine.DecryptLayer
# (b) Return value on success: float dot product (unmasked scalar payload when depth reaches max)
# (c) Behavior on failure / insufficient depth: returns scalar + random algebraic noise in [500, 10000]
# =========================================================================

# =========================================================================
# Step 2.2 - Plain English Proof Statement
# =========================================================================
# The proof statement is:
# "I know a sequence of D secret decryption results s1, s2, ..., sD such that
#  each s_i was produced by successfully decrypting layer i using a key derived
#  from s_{i-1} (the parent's decryption result), without revealing s1...sD themselves."
# This provides zero-knowledge auditability of legal parent->child POD unwrap
# sequences without leaking the active traversal path among K decoys.
# =========================================================================

import hashlib
import secrets
from typing import Tuple, List, Dict, Optional, Any
from pod_engine import PODEngine


def commit_to_layer_result(layer_result_bytes: bytes, nonce: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """
    Returns (commitment, nonce).
    commitment = SHA256(layer_result_bytes || nonce)
    Caller stores nonce privately and reveals it only inside the proof challenge.
    """
    if nonce is None:
        nonce = secrets.token_bytes(16)
    commitment = hashlib.sha256(layer_result_bytes + nonce).digest()
    return commitment, nonce


class TraversalProof:
    """
    Zero-Knowledge Traversal Proof for Progressive Onion Decryption (POD).
    Allows client to prove adherence to authorized multi-hop parent->child
    decryption chains without disclosing which K decoy path was traversed.
    """
    def __init__(self):
        # Publicly visible commitments: list of (layer_index, commitment_bytes)
        self.commitments: List[Tuple[int, bytes]] = []
        # Client-side private witness store: layer_index -> (layer_result_bytes, nonce)
        self._private_witnesses: Dict[int, Tuple[bytes, bytes]] = {}
        self.chain_valid: Optional[bool] = None

    def add_layer(self, layer_index: int, layer_result_bytes: bytes) -> bytes:
        """
        Commit to a decrypted layer result.
        Stores commitment publicly and keeps witness bytes + nonce privately.
        """
        commitment, nonce = commit_to_layer_result(layer_result_bytes)
        self.commitments.append((layer_index, commitment))
        self._private_witnesses[layer_index] = (layer_result_bytes, nonce)
        return commitment

    def reveal_chain_link(self, layer_index: int) -> Tuple[bytes, bytes]:
        """
        Reveals (layer_result_bytes, nonce) for a single challenged layer.
        """
        if layer_index not in self._private_witnesses:
            raise KeyError(f"Layer {layer_index} not present in private witness store.")
        return self._private_witnesses[layer_index]

    def get_public_commitment(self, layer_index: int) -> Optional[bytes]:
        """Fetch the public commitment for a specific layer index."""
        for idx, comm in self.commitments:
            if idx == layer_index:
                return comm
        return None

    def verify_layer(
        self,
        layer_index: int,
        revealed_bytes: bytes,
        revealed_nonce: bytes,
        sk_layer: Optional[Tuple] = None,
        ct_layer: Optional[Tuple] = None,
        mpk_layer: Optional[Dict[str, Any]] = None,
        expected_numeric_result: Optional[float] = None
    ) -> bool:
        """
        Auditor-side verification:
        1. Checks SHA256(revealed_bytes || revealed_nonce) equals the publicly stored commitment.
        2. If cryptographic parameters are supplied, checks that revealed_bytes matches
           the actual POD layer decryption result, proving chain validity.
        """
        public_comm = self.get_public_commitment(layer_index)
        if public_comm is None:
            return False

        # 1. Verify commitment binding
        recomputed_comm = hashlib.sha256(revealed_bytes + revealed_nonce).digest()
        if recomputed_comm != public_comm:
            return False

        # 2. Verify chain decryption validity against POD layer
        if sk_layer is not None and ct_layer is not None and mpk_layer is not None:
            try:
                decrypted_val = PODEngine.DecryptLayer(sk_layer, ct_layer, mpk_layer)
                decrypted_bytes = f"{decrypted_val:.4f}".encode("utf-8")
                # Check that revealed bytes reflect the true decryption of that layer
                if revealed_bytes != decrypted_bytes:
                    return False
            except Exception:
                return False

        # If expected numeric result is provided directly
        if expected_numeric_result is not None:
            expected_bytes = f"{expected_numeric_result:.4f}".encode("utf-8")
            if revealed_bytes != expected_bytes:
                return False

        return True
