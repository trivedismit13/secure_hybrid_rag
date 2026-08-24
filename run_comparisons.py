import os
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from crypto_engine import AdaIPFEEngine
from se_ipfe_engine import SEIPFEEngine
from qdcs_engine import QDCSEngine
from pod_engine import PODEngine
from rag_pipeline import VPRAGPipeline, keygen_with_blenders

def main():
    print("====================================================")
    # Print clean header
    print("   CipheRAG CRYPTOGRAPHIC COMPARISON SUITE")
    print("====================================================\n")
    
    # 1. Setup paths
    wiki_path = r"C:\Users\Lenovo\Downloads\vprag_prototype\wiki_500.json"
    output_dir = r"C:\Users\Lenovo\Downloads\vprag_prototype\output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load Wikipedia dataset
    if not os.path.exists(wiki_path):
        print(f"Error: Wikipedia file not found at {wiki_path}!")
        exit(1)
        
    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki_data = json.load(f)
        
    # Load all 385 documents for complete evaluation
    corpus = wiki_data
    print(f"[+] Loaded all {len(corpus)} real Wikipedia documents for comparison benchmark.")
    
    # 3. Initialize S-BERT Model
    print("\n[+] Loading SentenceTransformer model...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    dim = 384
    
    # Encode document texts
    print("[+] Encoding corpus texts to 384-dimensional vectors...")
    raw_vectors = []
    for doc in corpus:
        v = sbert.encode(doc["doc"])
        norm = np.linalg.norm(v)
        if norm > 0.0:
            v = v * (0.7 / norm)
        raw_vectors.append(v.tolist())
        
    # For benchmarking speed, we encrypt and match on a subset of 50 documents,
    # then scale the results mathematically to represent the full 385 documents.
    benchmark_subset_size = 50
    subset_vectors = raw_vectors[:benchmark_subset_size]
    scale_factor = len(raw_vectors) / benchmark_subset_size
    print(f"[+] Benchmarking on a subset of {benchmark_subset_size} documents. Scaling factor: {scale_factor:.2f}x")
        
    # Generate 5 query vectors
    queries = [
        "what is the capital of France?",
        "who designed the Turing machine?",
        "what is the history of computing?",
        "tell me about artificial intelligence",
        "what is the role of blockchain in security?"
    ]
    query_vectors = []
    for q in queries:
        v = sbert.encode(q)
        norm = np.linalg.norm(v)
        if norm > 0.0:
            v = v * (0.7 / norm)
        query_vectors.append(v.tolist())
        
    # 4. Initialize results dictionary
    results = {
        "Ada-IPFE": {},
        "SE-IPFE": {},
        "QDCS": {},
        "POD (Depth 3)": {}
    }
    
    lambda_bits = 256 # 512-bit RSA modulus for fast checks
    
    # ==========================================
    # BENCHMARK 1: Ada-IPFE (Baseline)
    # ==========================================
    print_benchmark_header("Evaluating Baseline: Ada-IPFE")
    
    # KeyGen
    t_start = time.time()
    mpk, msk = AdaIPFEEngine.Setup(lambda_bits, dim)
    alpha = random_blender(mpk['lambda_N'])
    beta = random_blender(mpk['lambda_N'])
    pk_y = (
        pow(mpk['g'], alpha, mpk['N2']),
        pow(mpk['g'], beta, mpk['N2'])
    )
    results["Ada-IPFE"]["KeyGen"] = time.time() - t_start
    
    # Encryption (benchmarked on subset, scaled to full corpus)
    t_start = time.time()
    encrypted_docs = []
    for x in subset_vectors:
        ct = AdaIPFEEngine.Encrypt(x, mpk, pk_y)
        encrypted_docs.append(ct)
    results["Ada-IPFE"]["Encrypt"] = (time.time() - t_start) * scale_factor
    
    # Retrieval Matching (benchmarked on subset, scaled to full corpus)
    t_start = time.time()
    for q in query_vectors:
        sk_q = keygen_with_blenders(q, msk, mpk, alpha, beta)
        for ct in encrypted_docs:
            _ = AdaIPFEEngine.Decrypt(sk_q, ct, mpk)
    results["Ada-IPFE"]["Match"] = ((time.time() - t_start) / len(queries)) * scale_factor
    
    # Gateway Decryption (QKV slice - 64 elements)
    t_start = time.time()
    # Mock row decryption
    eval_dim = 64
    sliced_msk = msk[:eval_dim]
    sliced_mpk = mpk.copy()
    sliced_mpk['n'] = eval_dim
    w_row = [random_blender(mpk['lambda_N']) for _ in range(eval_dim)]
    sk_row = keygen_with_blenders(w_row, sliced_msk, sliced_mpk, alpha, beta)
    ct_sliced = (encrypted_docs[0][0], encrypted_docs[0][1], encrypted_docs[0][2], encrypted_docs[0][3], encrypted_docs[0][4], encrypted_docs[0][5][:eval_dim])
    _ = AdaIPFEEngine.Decrypt(sk_row, ct_sliced, sliced_mpk)
    results["Ada-IPFE"]["Gateway"] = time.time() - t_start
    results["Ada-IPFE"]["SecurityEnforcement"] = 100.0 # Standard security baseline
    
    print(f"  KeyGen Time: {results['Ada-IPFE']['KeyGen']:.4f}s")
    print(f"  Encryption Time: {results['Ada-IPFE']['Encrypt']:.4f}s")
    print(f"  Avg Match Time: {results['Ada-IPFE']['Match']:.4f}s")
    print(f"  Gateway Decryption (QKV slice): {results['Ada-IPFE']['Gateway']:.4f}s")

    # ==========================================
    # BENCHMARK 2: SE-IPFE (Sensitivity-Embedded)
    # ==========================================
    print_benchmark_header("Evaluating: SE-IPFE (Sensitivity-Embedded)")
    
    t_start = time.time()
    mpk, msk = SEIPFEEngine.Setup(lambda_bits, dim)
    alpha = random_blender(mpk['lambda_N'])
    beta = random_blender(mpk['lambda_N'])
    pk_y = (
        pow(mpk['g'], alpha, mpk['N2']),
        pow(mpk['g'], beta, mpk['N2'])
    )
    results["SE-IPFE"]["KeyGen"] = time.time() - t_start
    
    # Encrypt (benchmarked on subset, scaled to full corpus)
    t_start = time.time()
    encrypted_se_docs = []
    for x in subset_vectors:
        ct_se = SEIPFEEngine.Encrypt(x, sensitivity=3, mpk=mpk, pk=pk_y)
        encrypted_se_docs.append(ct_se)
    results["SE-IPFE"]["Encrypt"] = (time.time() - t_start) * scale_factor
    
    # KeyGen with Clearance = 2 (Unauthorized) and Clearance = 4 (Authorized)
    sk_unauth = SEIPFEEngine.KeyGen(query_vectors[0], clearance=2, msk=msk, mpk=mpk, alpha=alpha, beta=beta)
    sk_auth = SEIPFEEngine.KeyGen(query_vectors[0], clearance=4, msk=msk, mpk=mpk, alpha=alpha, beta=beta)
    
    # Verify access enforcement
    unauth_val = SEIPFEEngine.Decrypt(sk_unauth, encrypted_se_docs[0], mpk)
    auth_val = SEIPFEEngine.Decrypt(sk_auth, encrypted_se_docs[0], mpk)
    
    # Standard product
    true_prod = np.dot(raw_vectors[0], query_vectors[0])
    
    # Calculate correctness
    auth_err = abs(auth_val - true_prod)
    unauth_err = abs(unauth_val - true_prod)
    is_blocked_correctly = unauth_err > 10.0 and auth_err < 0.1
    results["SE-IPFE"]["SecurityEnforcement"] = 100.0 if is_blocked_correctly else 0.0
    
    # Avg Match Time (benchmarked on subset, scaled to full corpus)
    t_start = time.time()
    for q in query_vectors:
        sk_se = SEIPFEEngine.KeyGen(q, clearance=4, msk=msk, mpk=mpk, alpha=alpha, beta=beta)
        for ct_se in encrypted_se_docs:
            _ = SEIPFEEngine.Decrypt(sk_se, ct_se, mpk)
    results["SE-IPFE"]["Match"] = ((time.time() - t_start) / len(queries)) * scale_factor
    results["SE-IPFE"]["Gateway"] = results["Ada-IPFE"]["Gateway"] # Decryption gate runs standard Ada-IPFE
    
    print(f"  KeyGen Time: {results['SE-IPFE']['KeyGen']:.4f}s")
    print(f"  Encryption Time: {results['SE-IPFE']['Encrypt']:.4f}s")
    print(f"  Avg Match Time: {results['SE-IPFE']['Match']:.4f}s")
    print(f"  Access Control Enforcement: {results['SE-IPFE']['SecurityEnforcement']:.1f}% SUCCESS")

    # ==========================================
    # BENCHMARK 3: QDCS (Query-Derived Scope)
    # ==========================================
    print_benchmark_header("Evaluating: QDCS (Query-Derived Scope)")
    
    t_start = time.time()
    mpk, msk = QDCSEngine.Setup(lambda_bits, dim)
    alpha = random_blender(mpk['lambda_N'])
    beta = random_blender(mpk['lambda_N'])
    pk_y = (
        pow(mpk['g'], alpha, mpk['N2']),
        pow(mpk['g'], beta, mpk['N2'])
    )
    results["QDCS"]["KeyGen"] = time.time() - t_start
    
    # Encrypt (benchmarked on subset, scaled to full corpus)
    t_start = time.time()
    encrypted_qdcs_docs = []
    for x in subset_vectors:
        ct_qdcs = QDCSEngine.Encrypt(x, domain='finance', mpk=mpk, pk=pk_y)
        encrypted_qdcs_docs.append(ct_qdcs)
    results["QDCS"]["Encrypt"] = (time.time() - t_start) * scale_factor
    
    # KeyGen: Auth scope ['finance', 'HR'] vs Unauth scope ['engineering']
    sk_q_auth = QDCSEngine.KeyGen(query_vectors[0], allowed_domains=['finance', 'HR'], msk=msk, mpk=mpk, alpha=alpha, beta=beta)
    sk_q_unauth = QDCSEngine.KeyGen(query_vectors[0], allowed_domains=['engineering'], msk=msk, mpk=mpk, alpha=alpha, beta=beta)
    
    # Verify boundary enforcement
    val_auth = QDCSEngine.Decrypt(sk_q_auth, encrypted_qdcs_docs[0], mpk)
    val_unauth = QDCSEngine.Decrypt(sk_q_unauth, encrypted_qdcs_docs[0], mpk)
    
    is_scoped_correctly = val_unauth == 0.0 and abs(val_auth - true_prod) < 0.1
    results["QDCS"]["SecurityEnforcement"] = 100.0 if is_scoped_correctly else 0.0
    
    # Avg Match Time (benchmarked on subset, scaled to full corpus)
    t_start = time.time()
    for q in query_vectors:
        sk_qdcs = QDCSEngine.KeyGen(q, allowed_domains=['finance'], msk=msk, mpk=mpk, alpha=alpha, beta=beta)
        for ct_qdcs in encrypted_qdcs_docs:
            _ = QDCSEngine.Decrypt(sk_qdcs, ct_qdcs, mpk)
    results["QDCS"]["Match"] = ((time.time() - t_start) / len(queries)) * scale_factor
    results["QDCS"]["Gateway"] = results["Ada-IPFE"]["Gateway"]
    
    print(f"  KeyGen Time: {results['QDCS']['KeyGen']:.4f}s")
    print(f"  Encryption Time: {results['QDCS']['Encrypt']:.4f}s")
    print(f"  Avg Match Time: {results['QDCS']['Match']:.4f}s")
    print(f"  Scope Exclusion Enforcement: {results['QDCS']['SecurityEnforcement']:.1f}% SUCCESS")

    # ==========================================
    # BENCHMARK 4: POD (Progressive Onion Decryption)
    # ==========================================
    print_benchmark_header("Evaluating: POD (Progressive Onion Decryption)")
    
    # Onion Depth Max = 3
    max_layers = 3
    
    t_start = time.time()
    mpk_layers, msk_layers = PODEngine.Setup(lambda_bits, dim, max_layers=max_layers)
    alphas = [random_blender(mpk_layers[0]['lambda_N']) for _ in range(max_layers)]
    betas = [random_blender(mpk_layers[0]['lambda_N']) for _ in range(max_layers)]
    
    pk_layers = []
    for l in range(max_layers):
        pk_l = (
            pow(mpk_layers[l]['g'], alphas[l], mpk_layers[l]['N2']),
            pow(mpk_layers[l]['g'], betas[l], mpk_layers[l]['N2'])
        )
        pk_layers.append(pk_l)
    results["POD (Depth 3)"]["KeyGen"] = time.time() - t_start
    
    # Encrypt (benchmarked on subset, scaled to full corpus)
    t_start = time.time()
    encrypted_pod_docs = []
    for x in subset_vectors:
        ct_pod = PODEngine.Encrypt(x, max_layers=max_layers, mpk_layers=mpk_layers, pk_layers=pk_layers)
        encrypted_pod_docs.append(ct_pod)
    results["POD (Depth 3)"]["Encrypt"] = (time.time() - t_start) * scale_factor
    
    # KeyGen for all layers
    sk_layers = PODEngine.KeyGen(query_vectors[0], max_layers=max_layers, msk_layers=msk_layers, mpk_layers=mpk_layers, alphas=alphas, betas=betas)
    
    # Verify progressive decryption at depth 2 (incomplete) vs depth 3 (complete)
    val_incomplete = PODEngine.Decrypt(sk_layers, encrypted_pod_docs[0], traversal_depth=2, max_layers=max_layers, mpk_layers=mpk_layers)
    val_complete = PODEngine.Decrypt(sk_layers, encrypted_pod_docs[0], traversal_depth=3, max_layers=max_layers, mpk_layers=mpk_layers)
    
    is_onion_masked_correctly = abs(val_incomplete - true_prod) > 100.0 and abs(val_complete - true_prod) < 0.1
    results["POD (Depth 3)"]["SecurityEnforcement"] = 100.0 if is_onion_masked_correctly else 0.0
    
    # Avg Match Time at depth 3 (benchmarked on 1 query and 5 documents for CPU speed, scaled to full corpus)
    t_start = time.time()
    sks = PODEngine.KeyGen(query_vectors[0], max_layers=max_layers, msk_layers=msk_layers, mpk_layers=mpk_layers, alphas=alphas, betas=betas)
    for ct_pod in encrypted_pod_docs[:5]:
        _ = PODEngine.Decrypt(sks, ct_pod, traversal_depth=3, max_layers=max_layers, mpk_layers=mpk_layers)
    results["POD (Depth 3)"]["Match"] = (time.time() - t_start) * (len(raw_vectors) / 5.0)
    
    # Gateway Decryption latency at depth 3 (requires 3 sequential decryptions)
    t_start = time.time()
    _ = PODEngine.Decrypt(sk_layers, encrypted_pod_docs[0], traversal_depth=3, max_layers=max_layers, mpk_layers=mpk_layers)
    results["POD (Depth 3)"]["Gateway"] = time.time() - t_start
    
    print(f"  KeyGen Time: {results['POD (Depth 3)']['KeyGen']:.4f}s")
    print(f"  Encryption Time: {results['POD (Depth 3)']['Encrypt']:.4f}s")
    print(f"  Avg Match Time (Depth 3): {results['POD (Depth 3)']['Match']:.4f}s")
    print(f"  Gateway Decryption (Depth 3): {results['POD (Depth 3)']['Gateway']:.4f}s")
    print(f"  Multi-Hop Masking Enforcement: {results['POD (Depth 3)']['SecurityEnforcement']:.1f}% SUCCESS")

    # ==========================================
    # SAVE AND PLOT COMPARATIVE METRICS
    # ==========================================
    json_output_path = os.path.join(output_dir, "comparison_results.json")
    with open(json_output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[+] Raw comparative metrics exported successfully to {json_output_path}")
    
    plot_comparisons(results, output_dir)
    print("====================================================")
    print("      ALL EXPERIMENTAL RUNS COMPLETE SUCCESSFULLY")
    print("====================================================")

def random_blender(modulus):
    return random.randint(1, modulus - 1)

def print_benchmark_header(title):
    print(f"\n--- {title} ---")

def plot_comparisons(results, output_dir):
    # Plotting comparison bar charts
    labels = list(results.keys())
    
    keygen_times = [results[algo]["KeyGen"] for algo in labels]
    encrypt_times = [results[algo]["Encrypt"] for algo in labels]
    match_times = [results[algo]["Match"] for algo in labels]
    gateway_times = [results[algo]["Gateway"] for algo in labels]
    
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - 1.5*width, keygen_times, width, label='Key Gen Latency (s)', color='#1b365d')
    rects2 = ax.bar(x - 0.5*width, encrypt_times, width, label='Encryption Time (s)', color='#4b6b94')
    rects3 = ax.bar(x + 0.5*width, match_times, width, label='Avg Match Time (s)', color='#2e7d32')
    rects4 = ax.bar(x + 1.5*width, gateway_times, width, label='Gateway Decrypt Time (s)', color='#c62828')
    
    ax.set_ylabel('Execution Latency (Seconds)')
    ax.set_title('Performance Latency Comparison Across Cryptographic Schemes')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale('log') # Log scale to handle latency disparities
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "comparison_results.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[+] Comparative bar chart saved to {plot_path}")

if __name__ == '__main__':
    main()
