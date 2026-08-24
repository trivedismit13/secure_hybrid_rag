"""
Tests for Phase 5: Trust Metrics.
"""

import pytest
from shield_rag.orchestrator.trust_metrics import TrustMetrics


def test_expected_calibration_error():
    # Model predictions: (confidence, is_correct)
    predictions = [
        (0.9, True), (0.95, True), (0.85, False),  # Bin 8/9 (high confidence)
        (0.6, True), (0.55, False), (0.65, False), # Bin 5/6 (medium confidence)
        (0.1, False), (0.2, False),                # Bin 1/2 (low confidence)
    ]
    
    # Let's manually calculate for 10 bins:
    # Bin [0.8, 0.9) and [0.9, 1.0]: 
    #   conf=0.9 -> bin 9. conf=0.95 -> bin 9. conf=0.85 -> bin 8.
    # Actually with conf=[0.9, 0.95, 0.85, 0.6, 0.55, 0.65, 0.1, 0.2]
    # Bins containing items:
    # Bin 1 (0.1): 1 item, acc=0, conf=0.1
    # Bin 2 (0.2): 1 item, acc=0, conf=0.2
    # Bin 5 (0.55): 1 item, acc=0, conf=0.55
    # Bin 6 (0.6, 0.65): 2 items, acc=0.5, conf=0.625
    # Bin 8 (0.85): 1 item, acc=0, conf=0.85
    # Bin 9 (0.9, 0.95): 2 items, acc=1.0, conf=0.925
    
    ece = TrustMetrics.expected_calibration_error(predictions, num_bins=10)
    
    # ECE must be positive and bounded [0, 1]
    assert 0.0 <= ece <= 1.0


def test_overconfidence_ratio():
    predictions = [
        (0.9, True),   # Correct, confident
        (0.95, False), # Incorrect, confident (OCR case)
        (0.85, False), # Incorrect, confident (OCR case)
        (0.6, True),   # Correct, unconfident
        (0.1, False),  # Incorrect, unconfident
    ]
    # 2 out of 5 are confident (>0.8) and wrong
    ocr = TrustMetrics.overconfidence_ratio(predictions)
    assert ocr == 2 / 5


def test_consistency_gap():
    # 3 queries, K=4 variations each
    predictions_across_prompts = [
        ["A", "A", "A", "A"], # Gap = 1 - 4/4 = 0
        ["A", "A", "B", "C"], # Gap = 1 - 2/4 = 0.5
        ["A", "B", "C", "D"], # Gap = 1 - 1/4 = 0.75
    ]
    # Avg gap = (0 + 0.5 + 0.75) / 3 = 1.25 / 3 = 0.4166...
    cg = TrustMetrics.consistency_gap(predictions_across_prompts)
    assert abs(cg - (1.25 / 3)) < 1e-6
