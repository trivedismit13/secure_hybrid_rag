"""
Intent Classifier — Component A.

Maps a user query string to an IntentLabel (target NodeType + allowed
RelationType expansion set). This defines the ontology-constrained
retrieval scope for the downstream retrieval pipeline.

Phase 1 implementation: keyword/pattern-based classifier using TF-IDF
features and a rule-augmented heuristic. This avoids requiring a fine-tuned
model for the baseline phase while still producing meaningful intent labels.

The closed-set label design is decided HERE and must not change after
Phase 3 begins — Component C's decoy pool logic depends on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from shield_rag.schema.ontology import (
    VALID_RELATION_SCHEMA,
    IntentLabel,
    NodeType,
    RelationType,
)


# ─── Keyword-to-intent mapping rules ───────────────────────────────────────
# These patterns are designed for the equipment-manual domain corpus.
# Each rule maps keyword patterns → (target_type, allowed_relations, base_confidence).

@dataclass
class IntentRule:
    """A single intent classification rule."""
    keywords: list[str]       # any of these keywords triggers the rule
    target_type: NodeType
    allowed_relations: set[RelationType]
    confidence_boost: float = 0.0  # added to base confidence


# Relation expansion maps: for each target NodeType, which relations can
# reach it (either as source or destination)?
EXPANSION_MAP: dict[NodeType, set[RelationType]] = {
    NodeType.REQUIREMENT: {
        RelationType.SATISFY,   # Block -> Requirement
        RelationType.TRACE,     # Requirement -> Action
    },
    NodeType.ACTION: {
        RelationType.TRACE,     # Requirement -> Action
        RelationType.ALLOCATE,  # Action -> Block
    },
    NodeType.BLOCK: {
        RelationType.SATISFY,   # Block -> Requirement
        RelationType.ALLOCATE,  # Action -> Block
        RelationType.PART_OF,   # Block -> Block
        RelationType.HAS_PARAMETER,  # Block -> Parameter
    },
    NodeType.PARAMETER: {
        RelationType.HAS_PARAMETER,  # Block -> Parameter
    },
}


# Keyword patterns for each NodeType
INTENT_RULES: list[IntentRule] = [
    # Requirement-targeting queries
    IntentRule(
        keywords=[
            "requirement", "must", "shall", "comply", "compliance",
            "standard", "regulation", "specification", "rating",
            "withstand", "tolerance", "limit", "threshold",
            "safety", "certified", "approval",
        ],
        target_type=NodeType.REQUIREMENT,
        allowed_relations=EXPANSION_MAP[NodeType.REQUIREMENT],
        confidence_boost=0.1,
    ),
    # Action-targeting queries
    IntentRule(
        keywords=[
            "procedure", "step", "action", "perform", "execute",
            "maintain", "maintenance", "replace", "install",
            "inspect", "test", "calibrate", "check", "verify",
            "lubricate", "clean", "repair", "overhaul",
            "how to", "when to", "frequency", "interval",
        ],
        target_type=NodeType.ACTION,
        allowed_relations=EXPANSION_MAP[NodeType.ACTION],
        confidence_boost=0.1,
    ),
    # Block-targeting queries
    IntentRule(
        keywords=[
            "component", "part", "assembly", "unit", "module",
            "pump", "impeller", "bearing", "seal", "shaft",
            "casing", "housing", "coupling", "motor", "valve",
            "system", "subsystem", "structure",
        ],
        target_type=NodeType.BLOCK,
        allowed_relations=EXPANSION_MAP[NodeType.BLOCK],
        confidence_boost=0.05,
    ),
    # Parameter-targeting queries
    IntentRule(
        keywords=[
            "parameter", "value", "measurement", "dimension",
            "pressure", "temperature", "flow", "speed", "rpm",
            "diameter", "weight", "capacity", "voltage", "power",
            "maximum", "minimum", "range", "operating",
            "psi", "gpm", "inch", "celsius", "fahrenheit",
            "how much", "what is the", "how many",
        ],
        target_type=NodeType.PARAMETER,
        allowed_relations=EXPANSION_MAP[NodeType.PARAMETER],
        confidence_boost=0.15,
    ),
]

# Comparison / indicator question patterns
COMPARISON_PATTERNS = [
    r"(is|does|are|do)\s+.+\s+(above|below|greater|less|more|fewer|exceed|over|under)",
    r"(higher|lower|larger|smaller|bigger)\s+than",
    r"(compare|comparison|versus|vs\.?)\s+",
    r"(true|false)\s*[:\?]",
]


class IntentClassifier:
    """Rule-based intent classifier for the equipment-manual domain.

    Maps a natural-language query to an IntentLabel containing:
    - target_type: which NodeType the query is primarily asking about
    - allowed_relations: which edge types constrained expansion may traverse
    - confidence: classification confidence in [0, 1]

    The classifier uses keyword matching with scoring heuristics.
    Phase 2+ may replace this with a fine-tuned model, but the OUTPUT
    interface (IntentLabel) is frozen.
    """

    def __init__(self) -> None:
        self._rules = INTENT_RULES
        self._comparison_patterns = [re.compile(p, re.IGNORECASE) for p in COMPARISON_PATTERNS]

    def classify(self, query: str) -> IntentLabel:
        """Classify a query string into an IntentLabel.

        Args:
            query: Natural language query string.

        Returns:
            IntentLabel with target_type, allowed_relations, and confidence.
        """
        query_lower = query.lower().strip()
        scores: dict[NodeType, float] = {nt: 0.0 for nt in NodeType}
        relation_sets: dict[NodeType, set[RelationType]] = {
            nt: set() for nt in NodeType
        }

        # Score each rule by counting keyword matches
        for rule in self._rules:
            match_count = 0
            for kw in rule.keywords:
                if kw.lower() in query_lower:
                    match_count += 1

            if match_count > 0:
                # Score = fraction of keywords matched + confidence boost
                score = (match_count / len(rule.keywords)) + rule.confidence_boost
                scores[rule.target_type] += score
                relation_sets[rule.target_type] |= rule.allowed_relations

        # Check for comparison/indicator patterns (boosts Parameter target)
        is_comparison = any(p.search(query_lower) for p in self._comparison_patterns)
        if is_comparison:
            scores[NodeType.PARAMETER] += 0.2
            scores[NodeType.REQUIREMENT] += 0.1

        # Find the best-scoring type
        best_type = max(scores, key=lambda nt: scores[nt])
        best_score = scores[best_type]

        # If no keywords matched at all, default to Block with broad expansion
        if best_score < 0.01:
            return IntentLabel(
                target_type=NodeType.BLOCK,
                allowed_relations=set(RelationType),  # allow all relations
                confidence=0.3,
            )

        # Compute confidence: normalize score to [0, 1]
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.3

        # Ensure we always include the expansion relations for the target type
        allowed = relation_sets[best_type] | EXPANSION_MAP.get(best_type, set())

        # For multi-hop support: if query seems to need cross-type traversal,
        # add relations that bridge to other high-scoring types
        secondary_types = sorted(
            [nt for nt in NodeType if nt != best_type and scores[nt] > 0.1],
            key=lambda nt: scores[nt],
            reverse=True,
        )
        for sec_type in secondary_types[:1]:  # at most one secondary
            allowed |= EXPANSION_MAP.get(sec_type, set())

        return IntentLabel(
            target_type=best_type,
            allowed_relations=allowed,
            confidence=min(confidence, 1.0),
        )

    def classify_with_expansion_depth(
        self, query: str, max_hops: int = 3
    ) -> tuple[IntentLabel, int]:
        """Classify and suggest hop depth based on query complexity.

        Returns:
            (IntentLabel, suggested_hop_count)
        """
        label = self.classify(query)

        # Heuristic: more relation types → more hops needed
        suggested_hops = min(max(1, len(label.allowed_relations) - 1), max_hops)

        # Comparison questions often need more hops (find parameter → through block → etc.)
        query_lower = query.lower()
        if any(p.search(query_lower) for p in self._comparison_patterns):
            suggested_hops = min(suggested_hops + 1, max_hops)

        return label, suggested_hops
