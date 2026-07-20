"""
SHIELD-RAG Wire Formats — FROZEN after initial commit.

Defines the ONLY data structures that cross the client/server trust boundary
from Phase 2 onward. Changing these after Phase 2 invalidates cross-phase
benchmark comparisons.

Serialization uses msgpack for compact binary encoding on the wire, with
JSON as a human-readable fallback for debugging/logging.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

# We use a lightweight custom serialization rather than depending on msgpack
# at the schema level — this avoids a hard dependency in the frozen schema layer.
# Higher-level code can wrap these in msgpack if desired.


@dataclass
class EncryptedBucket:
    """An encrypted node bucket as stored/transmitted on the server side.

    The server sees tokens and ciphertexts but cannot recover plaintext
    node content, type, or adjacency structure without the client's keys.

    Attributes:
        token:         PRF(key, session_salt, node_id) — server-visible address.
        ciphertext:    Ada-IPFE ciphertext of the node's embedding vector.
        type_tag_ct:   Encrypted NodeType (reveals nothing without type-checking subkey).
        adjacency_ct:  List of PRF tokens for neighbors, encrypted at rest.
                       The server stores these but cannot interpret them without
                       the client's PRF key for the relevant session.
    """

    token: bytes = b""
    ciphertext: bytes = b""
    type_tag_ct: bytes = b""
    adjacency_ct: list[bytes] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (base64-encoded bytes)."""
        import base64
        return {
            "token": base64.b64encode(self.token).decode("ascii"),
            "ciphertext": base64.b64encode(self.ciphertext).decode("ascii"),
            "type_tag_ct": base64.b64encode(self.type_tag_ct).decode("ascii"),
            "adjacency_ct": [
                base64.b64encode(ct).decode("ascii") for ct in self.adjacency_ct
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EncryptedBucket:
        """Deserialize from a JSON-compatible dict."""
        import base64
        return cls(
            token=base64.b64decode(d["token"]),
            ciphertext=base64.b64decode(d["ciphertext"]),
            type_tag_ct=base64.b64decode(d["type_tag_ct"]),
            adjacency_ct=[base64.b64decode(ct) for ct in d["adjacency_ct"]],
        )

    def to_bytes(self) -> bytes:
        """Compact binary serialization for wire transport.

        Format:
            [4B token_len][token][4B ct_len][ciphertext]
            [4B tag_len][type_tag_ct]
            [4B adj_count][ [4B adj_len][adj_ct] ... ]
        """
        parts: list[bytes] = []
        # Token
        parts.append(struct.pack("!I", len(self.token)))
        parts.append(self.token)
        # Ciphertext
        parts.append(struct.pack("!I", len(self.ciphertext)))
        parts.append(self.ciphertext)
        # Type tag
        parts.append(struct.pack("!I", len(self.type_tag_ct)))
        parts.append(self.type_tag_ct)
        # Adjacency list
        parts.append(struct.pack("!I", len(self.adjacency_ct)))
        for adj in self.adjacency_ct:
            parts.append(struct.pack("!I", len(adj)))
            parts.append(adj)
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> EncryptedBucket:
        """Deserialize from compact binary format."""
        offset = 0

        def read_blob() -> bytes:
            nonlocal offset
            (length,) = struct.unpack_from("!I", data, offset)
            offset += 4
            blob = data[offset : offset + length]
            offset += length
            return blob

        token = read_blob()
        ciphertext = read_blob()
        type_tag_ct = read_blob()

        (adj_count,) = struct.unpack_from("!I", data, offset)
        offset += 4
        adjacency_ct = [read_blob() for _ in range(adj_count)]

        return cls(
            token=token,
            ciphertext=ciphertext,
            type_tag_ct=type_tag_ct,
            adjacency_ct=adjacency_ct,
        )


@dataclass
class TraversalRequest:
    """A single-hop traversal request sent from client to server.

    The server receives k tokens (1 real + k-1 decoys) and cannot
    distinguish which is real. This is the core privacy mechanism
    of Component C's bounded-decoy oblivious traversal.

    Attributes:
        hop_index:        Which hop in the multi-hop traversal (0-indexed).
        requested_tokens: List of k PRF tokens — 1 real target + (k-1) decoys,
                          shuffled so position reveals nothing.
    """

    hop_index: int = 0
    requested_tokens: list[bytes] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        import base64
        return {
            "hop_index": self.hop_index,
            "requested_tokens": [
                base64.b64encode(t).decode("ascii") for t in self.requested_tokens
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraversalRequest:
        """Deserialize from JSON-compatible dict."""
        import base64
        return cls(
            hop_index=d["hop_index"],
            requested_tokens=[base64.b64decode(t) for t in d["requested_tokens"]],
        )

    def to_bytes(self) -> bytes:
        """Compact binary serialization.

        Format: [4B hop_index][4B token_count][ [4B tok_len][token] ... ]
        """
        parts: list[bytes] = []
        parts.append(struct.pack("!I", self.hop_index))
        parts.append(struct.pack("!I", len(self.requested_tokens)))
        for tok in self.requested_tokens:
            parts.append(struct.pack("!I", len(tok)))
            parts.append(tok)
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> TraversalRequest:
        """Deserialize from compact binary format."""
        offset = 0
        (hop_index,) = struct.unpack_from("!I", data, offset)
        offset += 4
        (token_count,) = struct.unpack_from("!I", data, offset)
        offset += 4
        tokens: list[bytes] = []
        for _ in range(token_count):
            (tok_len,) = struct.unpack_from("!I", data, offset)
            offset += 4
            tokens.append(data[offset : offset + tok_len])
            offset += tok_len
        return cls(hop_index=hop_index, requested_tokens=tokens)


@dataclass
class HopResult:
    """Result of a single oblivious traversal hop (client-side view).

    Attributes:
        hop_index:            Which hop produced this result.
        real_triples:         The decrypted (head, relation, tail) triples recovered.
        next_candidate_tokens: PRF tokens for the next hop's starting points.
        decoy_count:          Number of decoys used (k-1), for audit logging.
    """

    hop_index: int = 0
    real_triples: list[dict[str, Any]] = field(default_factory=list)
    next_candidate_tokens: list[bytes] = field(default_factory=list)
    decoy_count: int = 0


@dataclass
class TraversalSession:
    """Complete record of a multi-hop oblivious traversal (client-side audit log).

    Attributes:
        session_id:   Unique session identifier.
        query_hash:   Hash of the original query (for audit, not the query itself).
        hops:         Ordered list of HopResults.
        total_real_triples: Count of real triples recovered across all hops.
        total_decoys_used:  Total decoy tokens fetched across all hops.
    """

    session_id: str = ""
    query_hash: str = ""
    hops: list[HopResult] = field(default_factory=list)

    @property
    def total_real_triples(self) -> int:
        return sum(len(h.real_triples) for h in self.hops)

    @property
    def total_decoys_used(self) -> int:
        return sum(h.decoy_count for h in self.hops)

    @property
    def hop_count(self) -> int:
        return len(self.hops)
