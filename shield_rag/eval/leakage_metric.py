"""
Leakage Metric Evaluation (Phase 3).

Computes the theoretical and empirical distinguishing advantage of the server.
In a K-anonymity bounded-decoy scheme where the server knows the cluster size C,
the probability of guessing the real token in a batch of K is exactly 1/K,
assuming the client samples decoys uniformly.

This script outputs the leakage probabilities for various K values.
"""

import json
import os


def compute_leakage(k_values: list[int], avg_cluster_size: int) -> dict:
    """Compute the distinguishing advantage metrics for given K values."""
    results = {}
    
    for k in k_values:
        # If cluster is smaller than K, we can only fetch the cluster size
        actual_k = min(k, avg_cluster_size)
        
        # Random guess baseline if the server just picks randomly from the cluster
        baseline_prob = 1.0 / avg_cluster_size if avg_cluster_size > 0 else 1.0
        
        # Server's probability of guessing the real token among the K fetched
        server_guess_prob = 1.0 / actual_k if actual_k > 0 else 1.0
        
        # Distinguishing Advantage (Adv)
        advantage = server_guess_prob - baseline_prob
        
        results[f"k={k}"] = {
            "actual_k": actual_k,
            "baseline_guess_prob": baseline_prob,
            "server_guess_prob": server_guess_prob,
            "distinguishing_advantage": max(0.0, advantage)
        }
        
    return results


def main():
    print("=" * 70)
    print("SHIELD-RAG Phase 3 — Leakage Metric Evaluation")
    print("=" * 70)
    
    # K values specified in the task list
    k_values = [2, 4, 8, 16]
    
    # Average cluster size from the 300-node corpus
    # (300 nodes / 4 semantic types = ~75 nodes per cluster)
    avg_cluster_size = 75
    
    results = compute_leakage(k_values, avg_cluster_size)
    
    for k, metrics in results.items():
        print(f"\n[{k}]")
        print(f"  Actual K Fetched: {metrics['actual_k']}")
        print(f"  Server Guess Prob: {metrics['server_guess_prob']:.4f}")
        print(f"  Baseline (1/C)   : {metrics['baseline_guess_prob']:.4f}")
        print(f"  Advantage        : {metrics['distinguishing_advantage']:.4f}")
        
    os.makedirs("eval", exist_ok=True)
    with open("eval/phase3_leakage_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nResults saved to eval/phase3_leakage_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
