"""
Phase 1: Deterministic Nested Corpus Builder & Graph Constructor.
Guarantees strict nesting (100 in 385 in 1k in 5k in 10k in 50k) and builds
a deterministic relation-aware knowledge graph for graph experiments.
"""

import json
import os
import random
from typing import List, Dict, Tuple, Any, Optional
from adaptive_decoy_engine import KnowledgeGraphIndex


def load_base_corpus(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads the base Wikipedia corpus."""
    if path is None:
        possible_paths = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "wiki_500.json")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "wiki_500.json")),
            r"C:\Users\Lenovo\Downloads\vprag_prototype\prototype\wiki_500.json"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                path = p
                break
    if path is None or not os.path.exists(path):
        raise FileNotFoundError("Could not find base wiki_500.json")

    with open(path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    return corpus


def build_nested_corpus_subset(
    full_corpus: List[Dict[str, Any]],
    target_size: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Returns a deterministic prefix slice of size target_size,
    guaranteeing that subset(A) is a strict subset of subset(B) for A < B.
    """
    if target_size <= len(full_corpus):
        return full_corpus[:target_size]
    else:
        # For scaling beyond available Wikipedia entries, deterministically expand
        random.seed(seed)
        expanded = list(full_corpus)
        idx = len(full_corpus)
        while len(expanded) < target_size:
            base_item = full_corpus[idx % len(full_corpus)]
            new_item = {
                "id": str(idx),
                "doc": f"{base_item['doc']} [Synthetic Ext {idx}]",
                "sensitivity": (idx % 5) + 1,
                "domain": f"domain_{(idx % 8)}"
            }
            expanded.append(new_item)
            idx += 1
        return expanded


def build_knowledge_graph_from_corpus(
    corpus: List[Dict[str, Any]],
    relation_types: Optional[List[str]] = None,
    density_factor: float = 0.35,
    seed: int = 42
) -> KnowledgeGraphIndex:
    """
    Constructs a deterministic relation-linked Knowledge Graph from corpus documents.
    """
    if relation_types is None:
        relation_types = ["cites", "hyperlink", "topic_cluster", "author_collab", "category_parent"]
        
    random.seed(seed)
    kg = KnowledgeGraphIndex()
    n = len(corpus)
    
    for i in range(n):
        src_id = str(corpus[i].get("id", i))
        # Deterministically assign 1 to 4 outgoing edges
        num_edges = random.randint(1, 4)
        for _ in range(num_edges):
            dst_idx = (i + random.randint(1, max(1, int(n * density_factor)))) % n
            dst_id = str(corpus[dst_idx].get("id", dst_idx))
            rel = random.choice(relation_types)
            kg.add_edge(src_id, rel, dst_id)
            
    return kg
