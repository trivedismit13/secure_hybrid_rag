"""
Migration utility: Plaintext to Encrypted Store.

Takes a PlaintextGraphStore, encrypts all nodes, type tags, and adjacency lists,
and populates an EncryptedStore.
"""

from typing import Optional
from shield_rag.schema.wire import EncryptedBucket
from shield_rag.graph_store.plaintext_store import PlaintextGraphStore
from shield_rag.graph_store.encrypted_store import EncryptedStore
from shield_rag.crypto.ada_ipfe import AdaIPFE, MasterPublicKey, Ciphertext
from shield_rag.crypto.prf import PRFGenerator
from shield_rag.crypto.type_tag_cipher import TypeTagCipher


class GraphMigrator:
    """Migrates a plaintext graph into an encrypted graph."""

    def __init__(
        self,
        ipfe: AdaIPFE,
        mpk: MasterPublicKey,
        prf: PRFGenerator,
        type_cipher: TypeTagCipher,
        session_salt: bytes
    ) -> None:
        self.ipfe = ipfe
        self.mpk = mpk
        self.prf = prf
        self.type_cipher = type_cipher
        self.session_salt = session_salt

    def migrate(self, pt_store: PlaintextGraphStore) -> EncryptedStore:
        """Encrypts the entire plaintext store and returns an EncryptedStore."""
        enc_store = EncryptedStore()

        # Step 1: Pre-compute PRF tokens for all node IDs
        # (Needed because adjacency lists refer to neighbor IDs)
        id_to_token: dict[str, bytes] = {}
        for node in pt_store.get_all_nodes():
            token = self.prf.get_token(self.session_salt, node.node_id)
            id_to_token[node.node_id] = token

        # Step 2: Encrypt each node
        for node in pt_store.get_all_nodes():
            token = id_to_token[node.node_id]
            
            # Encrypt embedding
            if not node.embedding:
                # If no embedding (e.g., during testing), use zeros
                embedding = [0.0] * len(self.mpk.h)
            else:
                embedding = node.embedding
            
            ct: Ciphertext = self.ipfe.encrypt(self.mpk, embedding)
            ct_bytes = self.ipfe.serialize_ciphertext(ct)
            
            # Encrypt type tag
            type_tag_ct = self.type_cipher.encrypt_type(node.node_type)
            
            # Encrypt adjacency list
            # The adjacency list is just a list of neighbor PRF tokens.
            # In a full production system, we'd encrypt the relation type as well,
            # but Component C requires fetching tokens, and relation filtering
            # happens via Component D's relation subkeys after decryption.
            # So the stored adjacency is just the neighbor tokens.
            adjacency_ct = []
            neighbors = pt_store.get_neighbors(node.node_id, direction="outgoing")
            for neighbor_node, relation in neighbors:
                neighbor_token = id_to_token[neighbor_node.node_id]
                adjacency_ct.append(neighbor_token)

            bucket = EncryptedBucket(
                token=token,
                ciphertext=ct_bytes,
                type_tag_ct=type_tag_ct,
                adjacency_ct=adjacency_ct
            )
            
            enc_store.add_bucket(bucket)

        return enc_store
