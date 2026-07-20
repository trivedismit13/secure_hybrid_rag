"""
Tests for Phase 2 cryptographic primitives and encrypted storage.
"""

import pytest
import os
import random

from shield_rag.schema.ontology import NodeType, GraphNode, GraphEdge, RelationType
from shield_rag.graph_store.plaintext_store import PlaintextGraphStore
from shield_rag.crypto.prf import PRFGenerator
from shield_rag.crypto.type_tag_cipher import TypeTagCipher
from shield_rag.crypto.ada_ipfe import AdaIPFE
from shield_rag.graph_store.encrypted_store import EncryptedStore
from shield_rag.graph_store.migrate import GraphMigrator


class TestPRFGenerator:
    def test_deterministic(self):
        prf = PRFGenerator(key=b"12345678901234567890123456789012")
        salt = b"salt1"
        t1 = prf.get_token(salt, "nodeA")
        t2 = prf.get_token(salt, "nodeA")
        assert t1 == t2
        assert len(t1) == 32

    def test_unlinkability(self):
        prf = PRFGenerator(key=b"12345678901234567890123456789012")
        t1 = prf.get_token(b"salt1", "nodeA")
        t2 = prf.get_token(b"salt2", "nodeA")
        assert t1 != t2


class TestTypeTagCipher:
    def test_deterministic_encryption(self):
        cipher = TypeTagCipher()
        ct1 = cipher.encrypt_type(NodeType.BLOCK)
        ct2 = cipher.encrypt_type(NodeType.BLOCK)
        assert ct1 == ct2
        
    def test_decryption(self):
        cipher = TypeTagCipher()
        ct = cipher.encrypt_type(NodeType.ACTION)
        pt = cipher.decrypt_type(ct)
        assert pt == NodeType.ACTION

    def test_different_types(self):
        cipher = TypeTagCipher()
        ct_block = cipher.encrypt_type(NodeType.BLOCK)
        ct_req = cipher.encrypt_type(NodeType.REQUIREMENT)
        assert ct_block != ct_req
        
        cid_block = cipher.get_cluster_id(ct_block)
        cid_req = cipher.get_cluster_id(ct_req)
        assert cid_block != cid_req


class TestGraphMigrator:
    def test_migration(self):
        # 1. Setup plaintext store
        pt_store = PlaintextGraphStore()
        
        dim = 4
        nodes = [
            GraphNode("n1", NodeType.BLOCK, "A", [0.1]*dim),
            GraphNode("n2", NodeType.ACTION, "B", [0.2]*dim),
            GraphNode("n3", NodeType.BLOCK, "C", [0.3]*dim),
        ]
        for n in nodes:
            pt_store.add_node(n)
            
        pt_store.add_edge(GraphEdge("n1", "n2", RelationType.TRACE))
        pt_store.add_edge(GraphEdge("n1", "n3", RelationType.PART_OF))
        
        # 2. Setup crypto context
        ipfe = AdaIPFE(key_size=512)
        mpk, msk = ipfe.setup(dimension=dim)
        
        prf = PRFGenerator()
        type_cipher = TypeTagCipher()
        salt = os.urandom(16)
        
        # 3. Migrate
        migrator = GraphMigrator(ipfe, mpk, prf, type_cipher, salt)
        enc_store = migrator.migrate(pt_store)
        
        assert enc_store.node_count() == 3
        
        # 4. Verify tokens and buckets
        t1 = prf.get_token(salt, "n1")
        t2 = prf.get_token(salt, "n2")
        t3 = prf.get_token(salt, "n3")
        
        b1 = enc_store.fetch(t1)
        assert b1 is not None
        assert b1.token == t1
        
        # Verify type cluster blind index
        block_ct = type_cipher.encrypt_type(NodeType.BLOCK)
        block_cid = type_cipher.get_cluster_id(block_ct)
        
        cluster_tokens = enc_store.get_type_cluster(block_cid)
        assert len(cluster_tokens) == 2
        assert set(cluster_tokens) == {t1, t3}
        
        # Verify adjacency
        assert len(b1.adjacency_ct) == 2
        assert set(b1.adjacency_ct) == {t2, t3}
        
        # Verify decrypt type
        assert type_cipher.decrypt_type(b1.type_tag_ct) == NodeType.BLOCK
