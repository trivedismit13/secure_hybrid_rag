"""
Relation Subkeys (Component D).

Provides deterministic derivation of symmetric subkeys for each RelationType
from a master context key. These subkeys are provided to the LLM's attention
layer to "unmask" corresponding structural relations.
"""

import hmac
import hashlib
from typing import Dict, Set

from shield_rag.schema.ontology import RelationType


class RelationKeyManager:
    """Derives and manages relation subkeys."""

    def __init__(self, master_context_key: bytes):
        """
        Args:
            master_context_key: A 32-byte master key for the current query context.
        """
        if len(master_context_key) < 16:
            raise ValueError("Master context key must be at least 16 bytes.")
        self.master_key = master_context_key
        
    def derive_subkey(self, relation: RelationType) -> bytes:
        """Derive a 32-byte subkey for a specific RelationType."""
        # Simple HKDF-like derivation using HMAC-SHA256
        # Info string is simply the relation name
        info = relation.value.encode('utf-8')
        return hmac.new(self.master_key, b"relation_subkey:" + info, hashlib.sha256).digest()

    def get_authorized_subkeys(self, allowed_relations: Set[RelationType]) -> Dict[RelationType, bytes]:
        """Get the dictionary of subkeys ONLY for the allowed relations."""
        return {
            rel: self.derive_subkey(rel)
            for rel in allowed_relations
        }

    def verify_token_tag(self, relation: RelationType, provided_subkey: bytes) -> bool:
        """
        Verify if a provided subkey correctly matches the derived subkey for a relation.
        In practice, the attention hook does this mathematically, but this provides
        a logical check for testing.
        """
        expected = self.derive_subkey(relation)
        return hmac.compare_digest(expected, provided_subkey)
