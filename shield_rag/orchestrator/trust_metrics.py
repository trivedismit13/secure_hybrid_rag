"""
Trust Calibration Metrics (Component E).

Implements the Roumeliotis trust calibration metrics:
- Expected Calibration Error (ECE)
- Overconfidence Ratio (OCR)
- Consistency Gap (CG)

These metrics evaluate the reliability of the LLM's reasoning over the
encrypted structural graph, specifically detecting "consensus failure modes".
"""

from typing import List, Tuple, Any
from collections import Counter


class TrustMetrics:
    """Computes confidence and consistency metrics for LLM generation."""
    
    @staticmethod
    def expected_calibration_error(
        predictions: List[Tuple[float, bool]], 
        num_bins: int = 10
    ) -> float:
        """
        Calculates Expected Calibration Error (ECE).
        
        Args:
            predictions: List of (confidence_score, is_correct) tuples.
            num_bins: Number of probability bins (default 10 for [0,1]).
            
        Returns:
            The ECE score.
        """
        if not predictions:
            return 0.0
            
        bins = [[] for _ in range(num_bins)]
        
        # Assign predictions to bins
        for conf, is_correct in predictions:
            # Ensure conf is in [0, 1]
            conf = max(0.0, min(1.0, conf))
            bin_idx = int(conf * num_bins)
            if bin_idx == num_bins:
                bin_idx -= 1
            bins[bin_idx].append((conf, is_correct))
            
        ece = 0.0
        n = len(predictions)
        
        for b in bins:
            if not b:
                continue
            
            bin_size = len(b)
            avg_conf = sum(conf for conf, _ in b) / bin_size
            avg_acc = sum(1.0 for _, is_correct in b if is_correct) / bin_size
            
            ece += (bin_size / n) * abs(avg_acc - avg_conf)
            
        return ece

    @staticmethod
    def overconfidence_ratio(predictions: List[Tuple[float, bool]]) -> float:
        """
        Calculates the Overconfidence Ratio (OCR).
        OCR is the ratio of instances where the model is highly confident but wrong.
        For this implementation, we define 'highly confident' as > 0.8.
        
        Args:
            predictions: List of (confidence_score, is_correct) tuples.
            
        Returns:
            The OCR score [0, 1].
        """
        if not predictions:
            return 0.0
            
        confident_and_wrong = sum(1 for conf, is_correct in predictions if conf > 0.8 and not is_correct)
        return confident_and_wrong / len(predictions)

    @staticmethod
    def consistency_gap(predictions_across_prompts: List[List[Any]]) -> float:
        """
        Calculates the Consistency Gap (CG).
        Delta_cons = 1 - max_y (1/K * sum I[y_k == y])
        
        Args:
            predictions_across_prompts: A list of K runs. Each run is a list of N predictions.
                                        Or, simpler for our use case: A list where each element
                                        is a list of K responses for a single query.
                                        
        Returns:
            The average Consistency Gap across all queries.
        """
        if not predictions_across_prompts:
            return 0.0
            
        total_gap = 0.0
        num_queries = len(predictions_across_prompts)
        
        for query_responses in predictions_across_prompts:
            if not query_responses:
                continue
                
            k = len(query_responses)
            counts = Counter(query_responses)
            # Find the most frequent prediction frequency
            max_freq = counts.most_common(1)[0][1]
            
            gap = 1.0 - (max_freq / k)
            total_gap += gap
            
        return total_gap / num_queries
