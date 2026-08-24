"""
Phase 4 Runner — Benchmarks Structured In-Model Decryption.

Measures the generation latency overhead of the PyTorch attention layer hook.
Saves results to phase4_generation_latency.json.
"""

import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shield_rag.schema.ontology import RelationType, GraphNode, IntentLabel
from shield_rag.crypto.relation_subkeys import RelationKeyManager
from shield_rag.decrypt_attn.hook import AttentionDecryptionHook
from shield_rag.generation.structured_prompt import StructuredPromptBuilder


def main() -> None:
    print("=" * 70)
    print("SHIELD-RAG Phase 4 — Structured In-Model Decryption Benchmark")
    print("=" * 70)
    
    # We use a small proxy model for benchmarking the hook overhead locally
    # to avoid OOM, as approved in the Implementation Plan.
    model_name = "HuggingFaceTB/SmolLM-135M"
    print(f"[1/4] Loading proxy model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to("cpu")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have internet access or the model is cached.")
        return

    print("\n[2/4] Setting up Cryptographic Keys & Prompt...")
    master_key = os.urandom(32)
    key_manager = RelationKeyManager(master_key)
    builder = StructuredPromptBuilder(tokenizer, key_manager)
    
    # Mock some retrieved data
    nodes = [
        GraphNode("n1", None, "High Pressure Pump Housing", []),
        GraphNode("n2", None, "Pressure Rating 500 PSI", []),
        GraphNode("n3", None, "Hydrostatic Test Protocol", []),
    ]
    triples = [
        ("n1", RelationType.SATISFY, "n2"),
        ("n2", RelationType.TRACE, "n3")
    ]
    
    # Client intent only authorizes SATISFY, not TRACE
    intent = IntentLabel(
        target_type=None,
        allowed_relations={RelationType.SATISFY}
    )
    authorized_subkeys = list(key_manager.get_authorized_subkeys(intent.allowed_relations).values())
    
    prompt, token_reqs = builder.build_prompt("What is the pressure rating?", nodes, triples)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("\n[3/4] Benchmarking Generation Latency...")
    num_runs = 5
    max_new_tokens = 20
    
    # 1. Baseline (No Hook)
    baseline_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        baseline_times.append(time.perf_counter() - t0)
    avg_baseline = sum(baseline_times) / num_runs
    print(f"  Baseline Avg Latency: {avg_baseline:.3f} s")
    
    # 2. Hooked Generation
    hook = AttentionDecryptionHook()
    hook.register(model)
    hook.set_context(authorized_subkeys, token_reqs)
    
    hooked_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        hooked_times.append(time.perf_counter() - t0)
    avg_hooked = sum(hooked_times) / num_runs
    print(f"  Hooked Avg Latency:   {avg_hooked:.3f} s")
    
    overhead_pct = ((avg_hooked - avg_baseline) / avg_baseline) * 100
    print(f"  Overhead:             {overhead_pct:.2f}%")
    
    hook.remove()
    
    print("\n[4/4] Saving results...")
    results = {
        "model": model_name,
        "max_new_tokens": max_new_tokens,
        "runs": num_runs,
        "baseline_latency_s": avg_baseline,
        "hooked_latency_s": avg_hooked,
        "overhead_percent": overhead_pct
    }
    
    os.makedirs("eval", exist_ok=True)
    with open("eval/phase4_generation_latency.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("=" * 70)
    print("Phase 4 complete.")


if __name__ == "__main__":
    main()
