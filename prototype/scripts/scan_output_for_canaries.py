"""
Step 5.3: Output Scanner for Canary Token Leakage Detection.
Scans generation outputs for verbatim regex hits and fuzzy paraphrases
using standard difflib SequenceMatcher.
"""

import re
import difflib
from typing import List, Optional

CANARY_PATTERN = re.compile(r"zx\d+q\d+v[0-9a-f]{8}")


def scan_generation_output(generated_text: str) -> List[str]:
    """
    Returns a list of all canary tokens found verbatim in generated_text.
    """
    return CANARY_PATTERN.findall(generated_text)


def _sliding_windows(text: str, size: int):
    """Generates overlapping window substrings for fuzzy detection."""
    step = max(1, size // 2)
    for i in range(0, max(1, len(text) - size + 1), step):
        yield text[i:i + size]


def scan_for_fuzzy_canary_leakage(
    generated_text: str,
    known_canaries: List[str],
    ratio_threshold: float = 0.80
) -> List[str]:
    """
    Detects paraphrased or slightly corrupted canary tokens using
    Python's built-in difflib.SequenceMatcher.
    """
    hits = []
    for canary in known_canaries:
        # 1. Exact match check
        if canary in generated_text:
            if canary not in hits:
                hits.append(canary)
            continue
            
        # 2. Fuzzy sliding window check
        canary_len = len(canary)
        for window in _sliding_windows(generated_text, canary_len):
            ratio = difflib.SequenceMatcher(None, window, canary).ratio()
            if ratio >= ratio_threshold:
                if canary not in hits:
                    hits.append(canary)
                break
                
    return hits
