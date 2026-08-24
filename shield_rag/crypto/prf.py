"""
Keyed PRF for generating server-visible tokens.

Generates opaque tokens for node IDs. Includes a per-session salt
to ensure cross-session unlinkability, matching the privacy requirements
of Component C.
"""

import hashlib
import hmac
import os
from typing import Optional


class PRFGenerator:
    """Generates deterministic but unpredictable tokens for node IDs."""

    def __init__(self, key: Optional[bytes] = None) -> None:
        """
        Args:
            key: 32-byte master key. Generated randomly if None.
        """
        self.key = key or os.urandom(32)

    def get_token(self, session_salt: bytes, node_id: str) -> bytes:
        """Generate a token for a node ID in a specific session.

        Args:
            session_salt: Random salt rotated per session for unlinkability.
            node_id:      The plaintext node UUID.

        Returns:
            32-byte PRF token.
        """
        msg = session_salt + node_id.encode("utf-8")
        mac = hmac.new(self.key, msg, hashlib.sha256)
        return mac.digest()
