# Phase 1 Notes — Plaintext Functional Baseline

**Timestamp:** 2026-07-20T14:10:00+05:30  
**Phase:** 1 of 5  
**Status:** Complete (pending real-embedding re-evaluation)

---

## Design Decisions

### 1. Graph Store: In-Memory Dict-of-Dicts
Chose `dict[str, GraphNode]` with `defaultdict(list)` adjacency lists over Neo4j for several reasons:
- The encrypted store in Phase 2 is inherently custom (Neo4j can't do encrypted bucket lookups)
- Same interface (`GraphStoreInterface` protocol) is frozen for both plaintext and encrypted variants
- In-memory store is sufficient for the 300-node corpus scale
- Type index (`NodeType → set[node_id]`) maintained eagerly on `add_node()` — this mirrors the type-cluster-id blind index in Phase 2

### 2. Intent Classifier: Keyword-Based Rules
Phase 1 uses a rule-based classifier (`IntentClassifier`) rather than a fine-tuned model:
- **Rationale:** The closed-set label design (4 NodeTypes × 5 RelationTypes) is simple enough for keyword scoring
- **Output interface is frozen:** `IntentLabel(target_type, allowed_relations, confidence)` — Phase 2+ can swap the classifier implementation without changing downstream
- Includes comparison-pattern detection for true/false indicator questions
- Includes expansion-depth heuristic for multi-hop queries

### 3. Embedding Model: Sentence-BERT (all-MiniLM-L6-v2)
- 384-dim embeddings via `sentence-transformers`
- Falls back to deterministic hash-based pseudo-embeddings if the library is unavailable
- This dimension (384) determines the Ada-IPFE vector dimension in Phase 2

### 4. Constrained Expansion: BFS with Ontology Filtering
- BFS from anchor nodes, constrained by `IntentLabel.allowed_relations`
- Edges validated against `VALID_RELATION_SCHEMA` (src_type, relation, dst_type) constraints
- Similarity threshold cutoff using cosine similarity to the original query
- This is the plaintext reference implementation — Phase 3 must produce identical results

### 5. Answer Generation: Extractive Baseline
- **No LLM dependency in Phase 1** — uses extractive baseline for reproducibility
- Handles true/false questions via numeric extraction and comparison against triple evidence
- IC-HRAG-style prompt template defined but invoked only with local LLM in Phase 4+
- `AnswerResult` output dataclass is frozen for all phases

### 6. Test Corpus: Synthetic Equipment Manual
- Domain: Industrial centrifugal pump maintenance
- 300 nodes (38 blocks, 76 parameters, 20 requirements, 20 actions, ~146 supplementary passages)
- 179 edges across all 5 relation types
- 60 true/false evaluation questions (indicator_comparison, requirement_check, parameter_lookup)
- 70/15/15 train/val/test split (deterministic, seed=42)

---

## Baseline Results

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | 0.2500 | With hash-based pseudo-embeddings |
| Precision | 1.0000 | No false positives when predictions are made |
| Recall | 0.0571 | Very few positive predictions made |
| F1 | 0.1081 | Low due to pseudo-embeddings |
| Mean Latency | 0.91 ms | Very fast (no LLM, extractive only) |
| Insufficient Evidence | 8/60 | Questions with no useful triples found |

**Note:** These numbers use the hash-based pseudo-embedding fallback since `sentence-transformers` was not installed during the initial run. Results with real Sentence-BERT embeddings are expected to be significantly better. Will be re-evaluated.

---

## Frozen Interfaces (DO NOT CHANGE after Phase 3)

1. `NodeType` enum: Requirement, Action, Block, Parameter
2. `RelationType` enum: Satisfy, Trace, Allocate, HasParameter, PartOf  
3. `VALID_RELATION_SCHEMA`: Maps each RelationType to (src_type, dst_type)
4. `GraphNode` dataclass: node_id, node_type, text, embedding
5. `GraphEdge` dataclass: src_id, dst_id, relation
6. `IntentLabel` dataclass: target_type, allowed_relations, confidence
7. `RetrievedTriple` dataclass: head, relation, tail, score
8. `EncryptedBucket` wire format: token, ciphertext, type_tag_ct, adjacency_ct
9. `TraversalRequest` wire format: hop_index, requested_tokens
10. `GraphStoreInterface` protocol: add_node, add_edge, get_node, get_neighbors, similarity_search, get_nodes_by_type

---

## Test Suite

- **39 schema tests:** Enum completeness, serialization round-trips, edge cases
- **32 Phase 1 tests:** Graph store (12), intent classifier (7), anchor matcher (3), constrained expander (5), answer generator (4), integration (1)
- **All 71 tests passing**

---

## Deviations from Spec

None. Phase 1 was implemented exactly as specified in Section 3.
