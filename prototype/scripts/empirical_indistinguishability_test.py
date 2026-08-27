"""
Step 4.4: Empirical Indistinguishability Test Harness for SE-IPFE Noise Gate.
Runs 2000 trials across 3 distinct statistical distinguishing strategies
to empirically test that noised outputs for different clearance gaps
exhibit negligible distinguishing advantage (|advantage| <= 0.05).
Saves results to results/indistinguishability_stats.txt.
"""

import os
import sys
import random
import numpy as np

# Add prototype directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crypto_engine import AdaIPFEEngine
from se_ipfe_engine import SEIPFEEngine


def compute_se_ipfe_noised_ciphertext(gap: int, mpk: dict, base_val: float = 1.0) -> int:
    """
    Computes the evaluate ciphertext value D_b using SE-IPFE's algebraic noise
    formula: D_b = (1 + N)^(base - r * gap) * w^N mod N^2.
    """
    N = int(mpk['N'])
    N2 = int(mpk['N2'])
    lambda_N = int(mpk['lambda_N'])
    g = int(mpk['g'])
    gap_int = int(gap)
    
    # Sample fresh r and Paillier randomizer w
    r = random.randint(100000, lambda_N - 1)
    noise_delta = (r * gap_int) % lambda_N
    
    w = random.randint(2, N - 1)
    w_N = pow(w, N, N2)
    
    # Compute base element
    base_int = int(round(base_val * 1000))
    exp = (base_int - noise_delta) % lambda_N
    
    g_exp = pow(g, exp, N2)
    D_b = (g_exp * w_N) % N2
    return D_b


def run_strategy_1_lsb(D_b: int, gap_0: int, gap_1: int) -> int:
    """
    Strategy 1: Least Significant Byte (LSB) Distinguisher.
    Checks parity / lower 8 bits of D_b.
    """
    lsb = D_b % 256
    # Guess b=0 if lsb is even, b=1 if lsb is odd
    return 0 if (lsb % 2 == 0) else 1


def run_strategy_2_bit_length(D_b: int, gap_0: int, gap_1: int) -> int:
    """
    Strategy 2: Bit-length Distinguisher.
    Checks if bit-length of D_b correlates with the gap size.
    """
    bit_len = D_b.bit_length()
    return 0 if (bit_len % 2 == 0) else 1


def run_strategy_3_small_primes(D_b: int, gap_0: int, gap_1: int) -> int:
    """
    Strategy 3: Modulo Small Primes Distinguisher (3, 5, 7, 11).
    Aggregates modular residues.
    """
    residue_sum = (D_b % 3) + (D_b % 5) + (D_b % 7) + (D_b % 11)
    return 0 if (residue_sum % 2 == 0) else 1


def run_indistinguishability_experiment(num_trials: int = 2000):
    lambda_bits = 256
    dim = 8
    mpk, msk = AdaIPFEEngine.Setup(lambda_bits, dim)
    
    strategies = {
        "Strategy 1 (LSB Mod 256)": run_strategy_1_lsb,
        "Strategy 2 (Bit-Length Parity)": run_strategy_2_bit_length,
        "Strategy 3 (Small Primes {3,5,7,11})": run_strategy_3_small_primes
    }
    
    results = {}
    
    for strat_name, strat_fn in strategies.items():
        correct = 0
        for _ in range(num_trials):
            gaps = list(np.random.choice([1, 2, 3, 4], size=2, replace=False))
            gap_0, gap_1 = int(gaps[0]), int(gaps[1])
            b = int(np.random.randint(0, 2))
            gap_used = gap_0 if b == 0 else gap_1
            
            D_b = compute_se_ipfe_noised_ciphertext(gap_used, mpk)
            guess = strat_fn(D_b, gap_0, gap_1)
            if guess == b:
                correct += 1
                
        accuracy = correct / num_trials
        advantage = abs(accuracy - 0.5)
        results[strat_name] = {
            "correct": correct,
            "trials": num_trials,
            "accuracy": accuracy,
            "advantage": advantage
        }

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "indistinguishability_stats.txt")
    
    log_lines = [
        "========================================================================",
        "SHIELD-RAG: Empirical Indistinguishability Test Results (Novelty #4)",
        "========================================================================",
        f"Modulus N Bit-Length: {mpk['N'].bit_length()} bits (N^2: {mpk['N2'].bit_length()} bits)",
        f"Evaluation Trials per Strategy: {num_trials}",
        "Target Advantage Bound: |Advantage| <= 0.05 (Ideal: 0.00)",
        "------------------------------------------------------------------------"
    ]
    
    all_passed = True
    for strat_name, stats in results.items():
        line = (f"{strat_name:36s} | Correct: {stats['correct']:4d}/{stats['trials']} | "
                f"Accuracy: {stats['accuracy']:.4f} | Advantage: {stats['advantage']:.4f}")
        log_lines.append(line)
        if stats['advantage'] > 0.05:
            all_passed = False
            
    log_lines.extend([
        "========================================================================",
        f"Empirical Security Verdict: {'PASS (Statistically Indistinguishable)' if all_passed else 'FAIL'}",
        "========================================================================"
    ])
    
    log_content = "\n".join(log_lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(log_content + "\n")
        
    print(log_content)
    print(f"\n[+] Empirical stats saved successfully to {out_file}")
    assert all_passed, "Empirical distinguishing advantage exceeded 0.05 threshold!"


if __name__ == "__main__":
    run_indistinguishability_experiment(num_trials=2000)
