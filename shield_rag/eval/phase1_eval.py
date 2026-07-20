"""
Phase 1 Evaluation Harness.

Runs the full Phase 1 plaintext pipeline end-to-end and measures:
- Precision, Recall, F1 on true/false technical-indicator questions
- Mean latency per query
- Accuracy on the held-out test set

Results are saved to eval/phase1_results.json for cross-phase comparison.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from shield_rag.schema.ontology import IntentLabel, NodeType, RelationType, RetrievedTriple


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single evaluation run."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    total_questions: int = 0
    correct: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_confidence: float = 0.0
    insufficient_evidence_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionResult:
    """Result for a single evaluation question."""
    question_id: str = ""
    question: str = ""
    predicted_verdict: Optional[str] = None
    gold_answer: bool = False
    is_correct: bool = False
    confidence: float = 0.0
    latency_ms: float = 0.0
    reasoning: str = ""
    triples_used: int = 0


def compute_metrics(results: list[QuestionResult]) -> EvalMetrics:
    """Compute precision/recall/F1 from question results.

    For true/false classification:
    - TRUE prediction matching TRUE gold = True Positive
    - TRUE prediction matching FALSE gold = False Positive
    - FALSE prediction matching TRUE gold = False Negative
    - FALSE prediction matching FALSE gold = True Negative
    """
    import numpy as np

    metrics = EvalMetrics(total_questions=len(results))

    if not results:
        return metrics

    latencies = []
    confidences = []

    for r in results:
        latencies.append(r.latency_ms)
        confidences.append(r.confidence)

        if r.is_correct:
            metrics.correct += 1

        if r.predicted_verdict == "INSUFFICIENT_EVIDENCE":
            metrics.insufficient_evidence_count += 1
            continue

        predicted_true = r.predicted_verdict == "TRUE"
        gold_true = r.gold_answer

        if predicted_true and gold_true:
            metrics.true_positives += 1
        elif predicted_true and not gold_true:
            metrics.false_positives += 1
        elif not predicted_true and gold_true:
            metrics.false_negatives += 1
        else:
            metrics.true_negatives += 1

    # Accuracy
    metrics.accuracy = metrics.correct / len(results) if results else 0.0

    # Precision
    tp_fp = metrics.true_positives + metrics.false_positives
    metrics.precision = metrics.true_positives / tp_fp if tp_fp > 0 else 0.0

    # Recall
    tp_fn = metrics.true_positives + metrics.false_negatives
    metrics.recall = metrics.true_positives / tp_fn if tp_fn > 0 else 0.0

    # F1
    if metrics.precision + metrics.recall > 0:
        metrics.f1 = (
            2 * metrics.precision * metrics.recall
            / (metrics.precision + metrics.recall)
        )

    # Latency stats
    latencies_arr = np.array(latencies)
    metrics.mean_latency_ms = float(np.mean(latencies_arr))
    metrics.median_latency_ms = float(np.median(latencies_arr))
    metrics.p95_latency_ms = float(np.percentile(latencies_arr, 95))
    metrics.mean_confidence = float(np.mean(confidences)) if confidences else 0.0

    return metrics


def run_phase1_eval(
    graph_store,
    intent_classifier,
    anchor_matcher,
    expander,
    answer_generator,
    questions: list,  # list of EvalQuestion from corpus_builder
    output_path: str = "eval/phase1_results.json",
) -> EvalMetrics:
    """Run the full Phase 1 evaluation pipeline.

    Args:
        graph_store:       PlaintextGraphStore with populated corpus.
        intent_classifier: IntentClassifier instance.
        anchor_matcher:    AnchorMatcher instance.
        expander:          ConstrainedExpander instance.
        answer_generator:  AnswerGenerator instance.
        questions:         List of EvalQuestion objects.
        output_path:       Where to save results JSON.

    Returns:
        EvalMetrics with accuracy, P/R/F1, and latency stats.
    """
    results: list[QuestionResult] = []

    for q in questions:
        start = time.perf_counter()

        # Step 1: Classify intent
        intent = intent_classifier.classify(q.question)

        # Step 2: Find anchor nodes
        anchors = anchor_matcher.find_anchors(q.question, top_k=5)

        # Step 3: Expand from anchors
        query_emb = anchor_matcher._embedder.encode_single(q.question)
        triples = expander.expand(anchors, intent, query_emb)

        # Step 4: Generate answer
        answer = answer_generator.generate(q.question, triples)

        elapsed = (time.perf_counter() - start) * 1000

        # Evaluate
        predicted = answer.verdict
        gold = q.answer

        if predicted in ("TRUE", "FALSE"):
            is_correct = (predicted == "TRUE") == gold
        else:
            is_correct = False

        result = QuestionResult(
            question_id=q.question_id,
            question=q.question,
            predicted_verdict=predicted,
            gold_answer=gold,
            is_correct=is_correct,
            confidence=answer.confidence,
            latency_ms=elapsed,
            reasoning=answer.reasoning,
            triples_used=len(triples),
        )
        results.append(result)

    metrics = compute_metrics(results)

    # Save results
    output = {
        "phase": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": metrics.to_dict(),
        "per_question": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "predicted": r.predicted_verdict,
                "gold": r.gold_answer,
                "correct": r.is_correct,
                "confidence": r.confidence,
                "latency_ms": r.latency_ms,
                "triples_used": r.triples_used,
            }
            for r in results
        ],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return metrics
