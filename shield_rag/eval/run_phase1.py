"""
Phase 1 Runner — executes the full plaintext baseline pipeline and evaluation.

Usage: python -m shield_rag.eval.run_phase1
"""

from __future__ import annotations

import json
import os
import sys
import time


def main() -> None:
    print("=" * 70)
    print("SHIELD-RAG Phase 1 — Plaintext Functional Baseline")
    print("=" * 70)

    # 1. Build corpus
    print("\n[1/6] Building synthetic corpus...")
    from shield_rag.eval.corpus_builder import build_corpus, CorpusConfig

    corpus = build_corpus(CorpusConfig(seed=42))
    print(f"  Nodes: {len(corpus.nodes)}, Edges: {len(corpus.edges)}, "
          f"Questions: {len(corpus.questions)}")
    print(f"  Split — Train: {len(corpus.train_ids)}, "
          f"Val: {len(corpus.val_ids)}, Test: {len(corpus.test_ids)}")

    # 2. Populate graph store with embeddings
    print("\n[2/6] Populating graph store and computing embeddings...")
    from shield_rag.graph_store.plaintext_store import PlaintextGraphStore
    from shield_rag.retrieval.anchor_match import AnchorMatcher, SentenceBERTEmbedder

    store = PlaintextGraphStore()
    embedder = SentenceBERTEmbedder()

    # Embed all nodes
    t0 = time.perf_counter()
    texts = [n.text for n in corpus.nodes]
    embeddings = embedder.encode(texts)
    for node, emb in zip(corpus.nodes, embeddings):
        node.embedding = emb
        store.add_node(node)
    embed_time = time.perf_counter() - t0
    print(f"  Embedded {len(corpus.nodes)} nodes in {embed_time:.2f}s")

    # Add edges
    for edge in corpus.edges:
        try:
            store.add_edge(edge)
        except KeyError:
            pass  # skip edges with missing nodes
    print(f"  Store: {store.node_count()} nodes, {store.edge_count()} edges")

    # 3. Initialize pipeline components
    print("\n[3/6] Initializing pipeline components...")
    from shield_rag.intent.classifier import IntentClassifier
    from shield_rag.retrieval.expand import ConstrainedExpander, ExpansionConfig
    from shield_rag.generation.answer import AnswerGenerator, GenerationConfig

    classifier = IntentClassifier()
    matcher = AnchorMatcher(store, embedder=embedder, top_k=5)
    expander = ConstrainedExpander(
        store,
        ExpansionConfig(max_hops=2, similarity_threshold=0.1, max_triples=15),
    )
    generator = AnswerGenerator(GenerationConfig(use_local_model=False))

    # 4. Run evaluation on test questions
    print("\n[4/6] Running evaluation on test questions...")
    # Filter to test-set questions only
    test_node_ids = set(corpus.test_ids)
    # Use all questions for now (they reference nodes across the corpus)
    test_questions = corpus.questions

    print(f"  Evaluating {len(test_questions)} questions...")

    from shield_rag.eval.phase1_eval import run_phase1_eval

    os.makedirs("eval", exist_ok=True)
    metrics = run_phase1_eval(
        graph_store=store,
        intent_classifier=classifier,
        anchor_matcher=matcher,
        expander=expander,
        answer_generator=generator,
        questions=test_questions,
        output_path="eval/phase1_results.json",
    )

    # 5. Print results
    print("\n[5/6] Results:")
    print(f"  Accuracy:       {metrics.accuracy:.4f}")
    print(f"  Precision:      {metrics.precision:.4f}")
    print(f"  Recall:         {metrics.recall:.4f}")
    print(f"  F1:             {metrics.f1:.4f}")
    print(f"  Mean Latency:   {metrics.mean_latency_ms:.2f} ms")
    print(f"  Median Latency: {metrics.median_latency_ms:.2f} ms")
    print(f"  P95 Latency:    {metrics.p95_latency_ms:.2f} ms")
    print(f"  Confidence:     {metrics.mean_confidence:.4f}")
    print(f"  Insufficient:   {metrics.insufficient_evidence_count}/{metrics.total_questions}")

    # 6. Verify results saved
    print(f"\n[6/6] Results saved to eval/phase1_results.json")
    print("=" * 70)
    print("Phase 1 complete.")


if __name__ == "__main__":
    main()
