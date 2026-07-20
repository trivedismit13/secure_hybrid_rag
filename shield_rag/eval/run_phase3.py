"""
Phase 3 Runner — Benchmarks Bounded-Decoy Oblivious Traversal.

Measures the latency of traversing the graph with varying numbers
of decoys (K = 3, 5, 10). Saves results to phase3_traversal_bench.json.
"""

import json
import os
import time

from shield_rag.schema.ontology import NodeType, IntentLabel, RelationType
from shield_rag.crypto.ada_ipfe import AdaIPFE
from shield_rag.crypto.prf import PRFGenerator
from shield_rag.crypto.type_tag_cipher import TypeTagCipher
from shield_rag.graph_store.plaintext_store import PlaintextGraphStore
from shield_rag.graph_store.migrate import GraphMigrator
from shield_rag.oblivious_traversal.engine import ObliviousTraversalEngine
from shield_rag.eval.corpus_builder import build_corpus, CorpusConfig


def main() -> None:
    print("=" * 70)
    print("SHIELD-RAG Phase 3 — Oblivious Traversal Benchmarking")
    print("=" * 70)

    # 1. Setup Data & Crypto
    print("[1/3] Setting up corpus and encrypting graph (using fast 512-bit IPFE)...")
    corpus = build_corpus(CorpusConfig(seed=42))
    pt_store = PlaintextGraphStore()
    
    # We use a fast 512-bit modulus and small dim for benchmarking traversal overhead,
    # because the focus is on the multi-hop network orchestration logic.
    dim = 8
    ipfe = AdaIPFE(key_size=512, scale=100)
    mpk, msk = ipfe.setup(dimension=dim)
    
    prf = PRFGenerator()
    type_cipher = TypeTagCipher()
    salt = os.urandom(16)
    
    for node in corpus.nodes:
        node.embedding = [0.1] * dim  # dummy embedding
        pt_store.add_node(node)
    for edge in corpus.edges:
        try:
            pt_store.add_edge(edge)
        except KeyError:
            pass

    migrator = GraphMigrator(ipfe, mpk, prf, type_cipher, salt)
    t0 = time.perf_counter()
    enc_store = migrator.migrate(pt_store)
    print(f"  Graph encrypted in {(time.perf_counter()-t0):.2f}s")
    print(f"  Nodes: {enc_store.node_count()}")

    # 2. Setup Traversal
    print("\n[2/3] Benchmarking Traversal Latency (Varying K decoys)")
    # Find a block node that has at least one outgoing edge
    start_node = None
    for n in pt_store.get_nodes_by_type(NodeType.BLOCK):
        if pt_store.get_node_degree(n.node_id, direction="outgoing") > 0:
            start_node = n
            break
            
    if not start_node:
        start_node = pt_store.get_nodes_by_type(NodeType.BLOCK)[0]
        
    start_token = prf.get_token(salt, start_node.node_id)
    anchor_bucket = enc_store.fetch(start_token)
    
    intent = IntentLabel(
        target_type=NodeType.ACTION,
        allowed_relations={RelationType.SATISFY, RelationType.PART_OF, RelationType.HAS_PARAMETER, RelationType.TRACE, RelationType.ALLOCATE}
    )
    
    query_y = [0.1] * dim
    func_key = ipfe.keygen(msk, query_y)
    
    k_values = [1, 3, 5, 10]
    results = {}

    for k in k_values:
        engine = ObliviousTraversalEngine(
            store=enc_store,
            type_cipher=type_cipher,
            ipfe=ipfe,
            mpk=mpk,
            k_decoys=k
        )
        
        # Run a multi-hop traversal
        t0 = time.perf_counter()
        collected, hop_res = engine.orchestrate(
            anchors=[anchor_bucket],
            intent=intent,
            query_func_key=func_key,
            max_hops=2,
            similarity_threshold=-1.0 # explore all to maximize hops
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        # Calculate stats
        total_hops = len(hop_res)
        avg_hop_latency = elapsed_ms / total_hops if total_hops > 0 else 0
        total_buckets_fetched = sum(h.decoy_count + 1 for h in hop_res)
        
        print(f"\n  K = {k}")
        print(f"    Total Traversal Latency: {elapsed_ms:.2f} ms")
        print(f"    Total Hops (Edges Traversed): {total_hops}")
        print(f"    Avg Hop Latency: {avg_hop_latency:.2f} ms")
        print(f"    Buckets Fetched: {total_buckets_fetched} (Expected: {total_hops * k})")
        
        results[f"k={k}"] = {
            "total_latency_ms": elapsed_ms,
            "total_hops": total_hops,
            "avg_hop_latency_ms": avg_hop_latency,
            "total_buckets_fetched": total_buckets_fetched
        }

    # 3. Save Results
    print("\n[3/3] Saving results to eval/phase3_traversal_bench.json")
    os.makedirs("eval", exist_ok=True)
    with open("eval/phase3_traversal_bench.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("=" * 70)
    print("Phase 3 complete.")


if __name__ == "__main__":
    main()
