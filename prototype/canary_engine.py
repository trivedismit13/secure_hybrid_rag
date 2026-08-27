"""
Novelty #5: Canary-Token Leakage Detection Engine.
Generates cryptographically salted, statistically rare canary markers,
inserts them into a calibrated 20% subset of ingestion documents,
and tracks assignment mappings.
"""

import hashlib
import json
import os
import random
import re
from typing import List, Dict, Tuple, Optional, Any

# Secret salt known only to KDC / audit subsystem
SECRET_SALT = "kdc_sec_salt_9841f3d8a7c2e0b1"
CANARY_REGEX_PATTERN = r"zx\d+q\d+v[0-9a-f]{8}"
CANARY_REGEX = re.compile(CANARY_REGEX_PATTERN)


def generate_canary(doc_id: str, sensitivity_level: int) -> str:
    """
    Constructs a statistically unique, semantically inert canary token:
    Format: zx{doc_id}q{sensitivity_level}v{checksum}
    """
    # Clean doc_id to digits if possible, or use positive hash integer
    if isinstance(doc_id, str) and doc_id.isdigit():
        clean_id = doc_id
    else:
        clean_id = str(abs(hash(str(doc_id))) % 1000000)
        
    raw_key = f"{clean_id}-{sensitivity_level}-{SECRET_SALT}".encode("utf-8")
    checksum = hashlib.sha256(raw_key).hexdigest()[:8]
    return f"zx{clean_id}q{sensitivity_level}v{checksum}"


def insert_canary_into_document(doc_text: str, doc_id: str, sensitivity_level: int) -> Tuple[str, str]:
    """
    Inserts canary marker as a trailing, low-salience reference tag:
    doc_text + f" [ref:{canary}]"
    Returns (modified_text, canary_token).
    """
    canary = generate_canary(doc_id, sensitivity_level)
    modified_text = f"{doc_text.rstrip()} [ref:{canary}]"
    return modified_text, canary


def assign_canaries_to_corpus(
    corpus: List[Dict[str, Any]],
    fraction: float = 0.20,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Embeds canary tokens into a random 20% subset of documents across sensitivity levels.
    Saves assignment map to results/canary_assignment.json.
    """
    random.seed(seed)
    canary_assignment: Dict[str, Dict[str, Any]] = {}
    modified_corpus = []

    for idx, item in enumerate(corpus):
        doc_id = str(item.get("id", idx))
        sensitivity = item.get("sensitivity", (idx % 5) + 1)
        doc_text = item.get("doc", item.get("text", ""))
        
        # Determine if this document is chosen for canary insertion (20% probability)
        is_canary = random.random() < fraction
        
        if is_canary:
            mod_text, canary_tok = insert_canary_into_document(doc_text, doc_id, sensitivity)
            canary_assignment[doc_id] = {
                "canary_token": canary_tok,
                "sensitivity_level": sensitivity,
                "original_length": len(doc_text),
                "modified_length": len(mod_text)
            }
            new_item = dict(item)
            new_item["doc"] = mod_text
            new_item["canary_token"] = canary_tok
            new_item["has_canary"] = True
            modified_corpus.append(new_item)
        else:
            new_item = dict(item)
            new_item["has_canary"] = False
            modified_corpus.append(new_item)

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    assignment_path = os.path.join(results_dir, "canary_assignment.json")
    
    with open(assignment_path, "w", encoding="utf-8") as f:
        json.dump(canary_assignment, f, indent=2)

    return modified_corpus, canary_assignment
