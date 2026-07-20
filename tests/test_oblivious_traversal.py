"""
Tests for Phase 3: Bounded-Decoy Oblivious Traversal.
"""

import pytest
import os
import random

from shield_rag.schema.ontology import NodeType, GraphNode, GraphEdge, RelationType, IntentLabel
from shield_rag.schema.wire import EncryptedBucket, TraversalRequest, HopResult
from shield_rag.graph_store.plaintext_store import PlaintextGraphStore
from shield_rag.crypto.prf import PRFGenerator
from shield_rag.crypto.type_tag_cipher import TypeTagCipher
from shield_rag.crypto.ada_ipfe import AdaIPFE
from shield_rag.graph_store.encrypted_store import EncryptedStore
from shield_rag.graph_store.migrate import GraphMigrator
from shield_rag.oblivious_traversal.engine import ObliviousTraversalEngine


class TestObliviousTraversalEngine:
    @pytest.fixture
    def setup_data(self):
        # 1. Plaintext Graph
        pt_store = PlaintextGraphStore()
        dim = 4
        
        # A simple valid chain: Block ->(Satisfy)-> Requirement ->(Trace)-> Action
        nodes = [
            GraphNode("block1", NodeType.BLOCK, "Block 1", [0.1]*dim),
            GraphNode("req1", NodeType.REQUIREMENT, "Req 1", [0.2]*dim),
            GraphNode("act1", NodeType.ACTION, "Action 1", [0.3]*dim),
            # Add some decoys
            GraphNode("req2", NodeType.REQUIREMENT, "Req 2", [0.4]*dim),
            GraphNode("req3", NodeType.REQUIREMENT, "Req 3", [0.5]*dim),
            GraphNode("act2", NodeType.ACTION, "Action 2", [0.6]*dim),
        ]
        for n in nodes:
            pt_store.add_node(n)
            
        pt_store.add_edge(GraphEdge("block1", "req1", RelationType.SATISFY))
        pt_store.add_edge(GraphEdge("req1", "act1", RelationType.TRACE))
        
        # 2. Crypto Setup
        ipfe = AdaIPFE(key_size=512)
        mpk, msk = ipfe.setup(dimension=dim)
        
        prf = PRFGenerator()
        type_cipher = TypeTagCipher()
        salt = os.urandom(16)
        
        # 3. Migrate to Encrypted Store
        migrator = GraphMigrator(ipfe, mpk, prf, type_cipher, salt)
        enc_store = migrator.migrate(pt_store)
        
        return {
            "enc_store": enc_store,
            "type_cipher": type_cipher,
            "ipfe": ipfe,
            "mpk": mpk,
            "msk": msk,
            "prf": prf,
            "salt": salt,
            "dim": dim
        }

    def test_expected_target_types(self, setup_data):
        engine = ObliviousTraversalEngine(
            store=setup_data["enc_store"],
            type_cipher=setup_data["type_cipher"],
            ipfe=setup_data["ipfe"],
            mpk=setup_data["mpk"]
        )
        # RelationType.SATISFY is BLOCK -> REQUIREMENT
        types = engine._get_expected_target_types(NodeType.BLOCK)
        assert NodeType.REQUIREMENT in types
        assert NodeType.PARAMETER in types

    def test_traverse_hop_with_decoys(self, setup_data):
        # We want to traverse to req1, which is a REQUIREMENT.
        # There are 3 REQUIREMENTS in the store (req1, req2, req3).
        # We expect traverse_hop to fetch req1 and some decoys.
        engine = ObliviousTraversalEngine(
            store=setup_data["enc_store"],
            type_cipher=setup_data["type_cipher"],
            ipfe=setup_data["ipfe"],
            mpk=setup_data["mpk"],
            k_decoys=2  # Request 2 total buckets (1 real + 1 decoy)
        )
        
        target_token = setup_data["prf"].get_token(setup_data["salt"], "req1")
        
        real_bucket, num_decoys = engine.traverse_hop(
            target_token=target_token,
            expected_type=NodeType.REQUIREMENT,
            hop_index=0
        )
        
        assert num_decoys == 1
        assert real_bucket is not None
        assert real_bucket.token == target_token

    def test_orchestrate_multi_hop(self, setup_data):
        engine = ObliviousTraversalEngine(
            store=setup_data["enc_store"],
            type_cipher=setup_data["type_cipher"],
            ipfe=setup_data["ipfe"],
            mpk=setup_data["mpk"],
            k_decoys=3
        )
        
        start_token = setup_data["prf"].get_token(setup_data["salt"], "block1")
        anchor = setup_data["enc_store"].fetch(start_token)
        
        intent = IntentLabel(
            target_type=NodeType.ACTION,
            allowed_relations={RelationType.SATISFY, RelationType.TRACE}
        )
        
        func_key = setup_data["ipfe"].keygen(setup_data["msk"], [0.1]*setup_data["dim"])
        
        collected, hop_results = engine.orchestrate(
            anchors=[anchor],
            intent=intent,
            query_func_key=func_key,
            max_hops=2,
            similarity_threshold=-1.0 # bypass similarity check for test
        )
        
        assert len(collected) == 3 # block1, req1, act1
        # Hops: block1 -> req1 (hop 0), req1 -> act1 (hop 1)
        assert len(hop_results) == 2
        
        # Verify k-anonymity
        for hop in hop_results:
            # We asked for k=3, but if there aren't enough decoys in the cluster, 
            # it might be smaller.
            # ACTION cluster has 2 items (act1, act2).
            # REQUIREMENT cluster has 3 items.
            assert hop.decoy_count + 1 > 1
            assert hop.decoy_count + 1 <= 3
