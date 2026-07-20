"""
Phase 2 Runner — Benchmarks Ada-IPFE and Crypto primitives.

Generates setup times, keygen times, encrypt times, and decrypt times
to prove latency feasibility for the IPFE layer.
Saves results to eval/phase2_crypto_bench.json.
"""

import json
import os
import time
import random
import statistics

from shield_rag.crypto.ada_ipfe import AdaIPFE
from shield_rag.crypto.prf import PRFGenerator
from shield_rag.crypto.type_tag_cipher import TypeTagCipher
from shield_rag.schema.ontology import NodeType


def main() -> None:
    print("=" * 70)
    print("SHIELD-RAG Phase 2 — Crypto Primitives Benchmarking")
    print("=" * 70)

    # We use 1024-bit key size for the IPFE (standard for fast prototype, 
    # though 2048 is more secure in production, 1024 proves the concept).
    # The vector dimension is 384 (all-MiniLM-L6-v2 embedding size).
    key_size = 1024
    dimension = 384
    iterations = 50

    print(f"IPFE Parameters:")
    print(f"  Key Size:  {key_size} bits")
    print(f"  Dimension: {dimension}")
    print(f"  Iterations: {iterations}")

    ipfe = AdaIPFE(key_size=key_size, scale=1000)

    # 1. Setup Time
    print("\n[1/5] Benchmarking Setup...")
    t0 = time.perf_counter()
    mpk, msk = ipfe.setup(dimension=dimension)
    setup_time_ms = (time.perf_counter() - t0) * 1000
    print(f"  Setup Time: {setup_time_ms:.2f} ms")

    # 2. KeyGen Time
    print("\n[2/5] Benchmarking KeyGen...")
    keygen_times = []
    for _ in range(iterations):
        y = [random.random() * 2 - 1 for _ in range(dimension)]
        t0 = time.perf_counter()
        _ = ipfe.keygen(msk, y)
        keygen_times.append((time.perf_counter() - t0) * 1000)
    
    print(f"  Mean KeyGen Time: {statistics.mean(keygen_times):.2f} ms")

    # 3. Encrypt Time
    print("\n[3/5] Benchmarking Encrypt...")
    encrypt_times = []
    ciphertexts = []
    for _ in range(iterations):
        x = [random.random() * 2 - 1 for _ in range(dimension)]
        t0 = time.perf_counter()
        ct = ipfe.encrypt(mpk, x)
        encrypt_times.append((time.perf_counter() - t0) * 1000)
        ciphertexts.append(ct)
        
    print(f"  Mean Encrypt Time: {statistics.mean(encrypt_times):.2f} ms")

    # 4. Decrypt Time
    print("\n[4/5] Benchmarking Decrypt...")
    decrypt_times = []
    for ct in ciphertexts:
        # random query
        y = [random.random() * 2 - 1 for _ in range(dimension)]
        sk_y = ipfe.keygen(msk, y)
        
        t0 = time.perf_counter()
        _ = ipfe.decrypt(mpk, sk_y, ct)
        decrypt_times.append((time.perf_counter() - t0) * 1000)
        
    print(f"  Mean Decrypt Time: {statistics.mean(decrypt_times):.2f} ms")

    # 5. Serialization Size
    print("\n[5/5] Serialization Sizes...")
    ct_bytes = ipfe.serialize_ciphertext(ciphertexts[0])
    print(f"  Ciphertext Size: {len(ct_bytes) / 1024:.2f} KB")

    # Symmetric crypto
    prf = PRFGenerator()
    t_prf = prf.get_token(os.urandom(16), "test")
    
    type_cipher = TypeTagCipher()
    t_type = type_cipher.encrypt_type(NodeType.BLOCK)

    # Save to JSON
    results = {
        "key_size_bits": key_size,
        "vector_dimension": dimension,
        "iterations": iterations,
        "setup_latency_ms": setup_time_ms,
        "mean_keygen_latency_ms": statistics.mean(keygen_times),
        "mean_encrypt_latency_ms": statistics.mean(encrypt_times),
        "mean_decrypt_latency_ms": statistics.mean(decrypt_times),
        "ciphertext_size_bytes": len(ct_bytes),
        "prf_token_size_bytes": len(t_prf),
        "type_tag_ct_size_bytes": len(t_type),
    }

    os.makedirs("eval", exist_ok=True)
    out_path = "eval/phase2_crypto_bench.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {out_path}")
    print("=" * 70)
    print("Phase 2 complete.")


if __name__ == "__main__":
    main()
