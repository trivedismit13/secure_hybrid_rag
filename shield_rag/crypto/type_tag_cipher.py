"""
Type Tag Cipher — Client-side encryption and server-side blind index.

Encrypts the NodeType of a knowledge graph node. The encryption must be
deterministic so the server can group nodes of the same type into
"type clusters" without knowing the plaintext types.

This provides the decoy pool for Component C's bounded-decoy traversal:
the server can quickly return k-1 decoys from the same cluster as the target.
"""

import hashlib
import os
from typing import Optional

from Crypto.Cipher import AES
from shield_rag.schema.ontology import NodeType


class TypeTagCipher:
    """Deterministic authenticated encryption for node types."""

    def __init__(self, key: Optional[bytes] = None) -> None:
        """
        Args:
            key: 32-byte AES key. Generated randomly if None.
        """
        self.key = key or os.urandom(32)

    def encrypt_type(self, node_type: NodeType) -> bytes:
        """Encrypt a NodeType into a deterministic ciphertext.

        Uses AES-GCM with a synthetic IV derived from the plaintext
        to ensure deterministic output for the same type.
        """
        type_bytes = node_type.value.encode("utf-8")
        
        # Synthetic IV (nonce) for deterministic encryption
        # Using a hash of the key and plaintext
        nonce_mac = hashlib.sha256(self.key + type_bytes).digest()
        nonce = nonce_mac[:12]  # GCM uses 12-byte nonce
        
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(type_bytes)
        
        # Format: [12B nonce][16B tag][ciphertext]
        return nonce + tag + ciphertext

    def decrypt_type(self, type_tag_ct: bytes) -> NodeType:
        """Decrypt a type tag ciphertext back to a NodeType."""
        nonce = type_tag_ct[:12]
        tag = type_tag_ct[12:28]
        ciphertext = type_tag_ct[28:]
        
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        
        return NodeType(plaintext.decode("utf-8"))

    @staticmethod
    def get_cluster_id(type_tag_ct: bytes) -> bytes:
        """Compute the server-side blind index for a type tag.

        The server uses this hash to group identical type tags into
        clusters without knowing the plaintext type or the encryption key.
        """
        return hashlib.sha256(type_tag_ct).digest()
