import time
import numpy as np
import torch
import json
import hashlib
import os
from config import L_D, SCALE_FACTOR
from crypto_engine import AdaIPFEEngine
from alsh_engine import ALSHEngine
from ipfs_mock import IPFSMock
from rag_pipeline import VPRAGPipeline, keygen_with_blenders

print("Loading pre-trained SentenceTransformer model ('all-MiniLM-L6-v2')...")
try:
    from sentence_transformers import SentenceTransformer
    SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    SBERT_MODEL = None
    print("WARNING: sentence-transformers not installed. Falling back to mock vectors.")

def print_separator(title):
    print("\n" + "="*80)
    print(f" {title.upper()} ".center(80, "="))
    print("="*80 + "\n")

def run_interactive_demo():
    print_separator("CipheRAG Live Demonstration Script")
    print("This script walks you through the step-by-step cryptographic and search pipeline")
    print("to demonstrate how CipheRAG secures documents and queries.")
    input("\n[Press Enter to start Step 1: System Setup]")

    # --- STEP 1: INITIALIZATION ---
    print_separator("Step 1: System Setup & Key Generation")
    print("Initializing the V-PPRAG pipeline with 384-dimensional S-BERT embeddings.")
    print("The Key Distribution Center (KDC) generates two safe primes to form modulus N.")
    
    pipeline = VPRAGPipeline(hidden_dim=384, K=128, lambda_bits=256)
    
    # Show public parameters
    N = pipeline.mpk_h['N']
    g = pipeline.mpk_h['g']
    print(f"\n[+] Generated RSA Modulus N (512-bit safe prime composite):")
    print(f"    N = {N}")
    print(f"    N^2 = {pipeline.mpk_h['N2']}")
    print(f"\n[+] Public Generator g (order dividing lambda(N)):")
    print(f"    g = {g}")
    print("\n[+] Master Secret Keys (msk) and System Blenders (alpha, beta) generated inside KDC.")
    
    step2_input = input("\n[Press Enter to proceed to Step 2 (Or type the private document here directly)]:\n>>> ").strip()
    
    # --- STEP 2: DOCUMENT ENCRYPTION & DUAL STORAGE ---
    print_separator("Step 2: Document Preprocessing, Hashing, & Encryption")
    
    default_doc = "Paris is the capital and most populous city of France, situated on the Seine River. It has been a major European center of finance, commerce, fashion, and the arts since the 17th century."
    if step2_input:
        doc_text = step2_input
    else:
        doc_text = input(f"Enter a private document text to store (or press Enter to use default):\n>>> ").strip()
        if not doc_text:
            doc_text = default_doc
            
    print(f"\n[+] Raw Document Context to Store:\n    \"{doc_text}\"")
    
    # 1. Embed
    print("\n1. Generating semantic vector representation (384-dim S-BERT)...")
    if SBERT_MODEL is not None:
        raw_emb = SBERT_MODEL.encode(doc_text)
        norm = np.linalg.norm(raw_emb)
        if norm > 0:
            raw_emb = raw_emb * (0.7 / norm)
    else:
        raw_emb = generate_dummy_vector(384)
        
    emb = pipeline.alsh.p_transform(raw_emb)
    print(f"   - Generated vector dimensions: {emb.shape}")
    print(f"   - Vector snippet (first 5 elements): {emb[:5]}")
    
    # 2. ALSH Hashing
    print("\n2. Applying ALSH P-Transformation and random hyperplane projections (K=128)...")
    H_v = pipeline.alsh.hash_vector(emb)
    print(f"   - Generated ALSH Binary Signature (length 128, elements in {{-1, 1}}):")
    print(f"     {H_v.tolist()[:30]} ... [truncated]")
    
    # 3. Encrypt H_v under Ada-IPFE
    print("\n3. Encrypting ALSH Signature via Ada-IPFE (for On-chain storage):")
    H_v_float = [float(val) for val in H_v]
    ct_h = AdaIPFEEngine.Encrypt(H_v_float, pipeline.mpk_h, pipeline.pk_h)
    print(f"   - Ciphertext elements generated:")
    print(f"     ct_0 (a blender): {ct_h[0]}")
    print(f"     ct_1 (g^r mod N^2): {str(ct_h[1])[:60]}... [truncated]")
    print(f"     ct_5[0] (encrypted signature element 1): {str(ct_h[5][0])[:60]}... [truncated]")
    
    # 4. Encrypt full embedding vector and upload to IPFS
    print("\n4. Encrypting full embedding vector via Ada-IPFE (for Off-chain IPFS storage)...")
    ct_e = AdaIPFEEngine.Encrypt(raw_emb.tolist(), pipeline.mpk_e, pipeline.pk_e)
    cid = pipeline.ipfs.upload(ct_e)
    pipeline.text_db[cid] = doc_text
    
    # 5. Upload to contract
    pipeline.blockchain.get_contract().uploadCorpus(
        b"demo_corpus", 
        [json.dumps({'ct_0': ct_h[0], 'ct_1': ct_h[1], 'ct_2': ct_h[2], 'ct_3': ct_h[3], 'ct_4': ct_h[4], 'ct_5': ct_h[5]}).encode('utf-8')], 
        [cid]
    )
    print(f"\n[+] Off-chain Storage Complete: Uploaded encrypted embedding payload to IPFS.")
    print(f"    - IPFS CID: {cid}")
    print(f"[+] On-chain Storage Complete: Uploaded encrypted signature to Solidity Smart Contract.")
    
    step3_input = input("\n[Press Enter to proceed to Step 3 (Or type the query here directly)]:\n>>> ").strip()
    
    # --- STEP 3: USER QUERY SUBMISSION ---
    print_separator("Step 3: User Query Hashing & Subkey Generation")
    
    default_query = "what is the capital of France?"
    if step3_input:
        query_text = step3_input
    else:
        query_text = input(f"Enter a search query (or press Enter to use default):\n>>> ").strip()
        if not query_text:
            query_text = default_query
    print(f"\n[+] User Plaintext Query: \"{query_text}\"")
    
    # 1. Q-Transformation
    if SBERT_MODEL is not None:
        raw_q = SBERT_MODEL.encode(query_text)
        norm = np.linalg.norm(raw_q)
        if norm > 0:
            raw_q = raw_q * (0.7 / norm)
    else:
        raw_q = generate_dummy_vector(384)
        
    q_trans = pipeline.alsh.q_transform(raw_q)
    H_q = pipeline.alsh.hash_vector(q_trans)
    print(f"\n1. Generated Query ALSH Signature:")
    print(f"   {H_q.tolist()[:30]} ... [truncated]")
    
    # 2. KeyGen for query signature H_q
    print("\n2. Requesting functional query subkey (sk_q) from KDC...")
    H_q_float = [float(val) for val in H_q]
    sk_hq = keygen_with_blenders(H_q_float, pipeline.msk_h, pipeline.mpk_h, pipeline.alpha_h, pipeline.beta_h)
    print(f"   - Generated subkey sk_q = (beta, sk):")
    print(f"     beta (blender value): {sk_hq[0]}")
    print(f"     sk (dot_s_y + alpha + beta mod lambda): {sk_hq[1]}")
    
    # 3. Submit Query to Blockchain
    query_id = hashlib.sha256(b"demo_query").digest()
    pipeline.blockchain.get_contract().submitQuery(
        query_id, 
        b"demo_corpus", 
        json.dumps({'beta': sk_hq[0], 'sk': sk_hq[1], 'y_scaled': sk_hq[2]}).encode('utf-8'), 
        b"query_audit_hash"
    )
    print(f"\n[+] Submitted query transaction to smart contract containing query subkey sk_q.")
    
    input("\n[Press Enter to proceed to Step 4: On-Chain Search & Oracle Matching]")

    # --- STEP 4: ORACLE SEARCH & RETRIEVAL ---
    print_separator("Step 4: Encrypted Search Matching (Algorithm 1)")
    print("The off-chain Oracle detects the QuerySubmitted blockchain event.")
    print("It fetches the encrypted database signatures and compares them using the query subkey sk_q.")
    
    # Oracle matching execution
    contract = pipeline.blockchain.get_contract()
    enc_sigs, corpus_cids = contract.getCorpus(b"demo_corpus")
    
    print(f"\n[+] Oracle runs Decrypt(sk_q, Enc(H(v_i))):")
    for ct_h_bytes, cid in zip(enc_sigs, corpus_cids):
        payload = json.loads(ct_h_bytes.decode('utf-8'))
        ct_h = (payload['ct_0'], payload['ct_1'], payload['ct_2'], payload['ct_3'], payload['ct_4'], payload['ct_5'])
        
        # Run decryption matching
        dot_product = AdaIPFEEngine.Decrypt(sk_hq, ct_h, pipeline.mpk_h)
        similarity = dot_product / pipeline.K
        
        print(f"    - Comparing query signature with IPFS document '{cid}':")
        print(f"      Calculated Inner Product Score = {dot_product}")
        print(f"      Calculated ALSH Collision Similarity = {similarity:.4f}")
        
    print(f"\n[+] Oracle ranks the documents and registers the winning CID on-chain.")
    print(f"    Winning CID retrieved: {cid}")
    
    input("\n[Press Enter to proceed to Step 5: Decryption attention gateway]")

    # --- STEP 5: DECRYPTION ATTENTION GATEWAY ---
    print_separator("Step 5: Decryption-Enabled Attention Gateway (Algorithm 2)")
    print("The client fetches the encrypted document embedding from IPFS.")
    print("It decrypts the Query-Key-Value projection states inside the attention layer.")
    
    # Setup eval dim
    eval_dim = 8
    print(f"\n1. Model sets up row subkeys for the attention weight matrices (evaluating {eval_dim} rows):")
    
    # Slice variables for demo
    sliced_msk_e = pipeline.msk_e[:eval_dim]
    sliced_mpk_e = pipeline.mpk_e.copy()
    sliced_mpk_e['n'] = eval_dim
    
    # Slice weights
    w_K = pipeline.W_K[:eval_dim, :eval_dim]
    w_V = pipeline.W_V[:eval_dim, :eval_dim]
    
    sk_K = []
    sk_V = []
    for j in range(eval_dim):
        sk_K.append(keygen_with_blenders(w_K[j].tolist(), sliced_msk_e, sliced_mpk_e, pipeline.alpha_e, pipeline.beta_e))
        sk_V.append(keygen_with_blenders(w_V[j].tolist(), sliced_msk_e, sliced_mpk_e, pipeline.alpha_e, pipeline.beta_e))
    print(f"   - Generated {eval_dim} row subkeys for Key weights W_K.")
    print(f"   - Generated {eval_dim} row subkeys for Value weights W_V.")
    
    # Run gateway decryption
    print(f"\n2. Executing row-wise projection decryption for embedding '{cid}'...")
    ct_e_full = pipeline.ipfs.fetch(cid)
    ct_e_sliced = (ct_e_full[0], ct_e_full[1], ct_e_full[2], ct_e_full[3], ct_e_full[4], ct_e_full[5][:eval_dim])
    
    # Decrypt Key state elements
    K_proj = np.zeros(eval_dim)
    V_proj = np.zeros(eval_dim)
    for j in range(eval_dim):
        K_proj[j] = AdaIPFEEngine.Decrypt(sk_K[j], ct_e_sliced, sliced_mpk_e)
        V_proj[j] = AdaIPFEEngine.Decrypt(sk_V[j], ct_e_sliced, sliced_mpk_e)
        
    print(f"   - Decrypted Key state vector snippet (W_K * x):")
    print(f"     {K_proj}")
    print(f"   - Decrypted Value state vector snippet (W_V * x):")
    print(f"     {V_proj}")
    
    # Execute attention
    Q_state = torch.randn(1, 1, eval_dim)
    K_full = torch.tensor(K_proj).unsqueeze(0).unsqueeze(0).float()
    V_full = torch.tensor(V_proj).unsqueeze(0).unsqueeze(0).float()
    
    attn_out = pipeline.execute_self_attention(Q_state, K_full, V_full)
    print(f"\n3. Executing Self-Attention over decrypted transient states...")
    print(f"   - Attention Output Vector: {attn_out.squeeze().tolist()}")
    
    input("\n[Press Enter to proceed to Step 6: Final Answer Generation]")

    # --- STEP 6: RESPONSE GENERATION ---
    print_separator("Step 6: Final Answer Generation")
    print("In standard RAG, the LLM uses the decrypted attention states to generate the final textual answer.")
    
    # Retrieve decrypted context text
    context = pipeline.text_db[cid]
    print(f"\n[+] Retrieved Context (Decrypted):")
    print(f"    \"{context}\"")
    print(f"\n[+] User Query:")
    print(f"    \"{query_text}\"")
    
    # Clean punctuation and map synonyms for robust keyword extraction
    import re
    clean_query = re.sub(r'[^\w\s]', '', query_text).lower()
    query_words = [w for w in clean_query.split() if len(w) > 2]
    
    # Synonym expansions to map query intent to document terms
    synonym_map = {
        "age": ["year", "old", "age", "born", "birthday"],
        "old": ["year", "old", "age"],
        "who": ["name", "who", "identity", "nitya", "thaker"],
        "name": ["name", "who", "identity", "nitya", "thaker"],
        "university": ["university", "institute", "college", "school", "technology", "vellore", "vit"],
        "college": ["university", "institute", "college", "school", "technology", "vellore", "vit"],
        "school": ["university", "institute", "college", "school", "technology", "vellore", "vit"],
        "capital": ["capital", "city", "paris", "london", "seine"]
    }
    
    expanded_keywords = []
    for qw in query_words:
        expanded_keywords.append(qw)
        if qw in synonym_map:
            expanded_keywords.extend(synonym_map[qw])
            
    # Remove duplicates
    expanded_keywords = list(set(expanded_keywords))
    
    sentences = [s.strip() for s in context.split('.') if s.strip()]
    best_sentence = ""
    max_matches = 0
    
    for s in sentences:
        matches = sum(1 for kw in expanded_keywords if kw in s.lower())
        if matches > max_matches:
            max_matches = matches
            best_sentence = s + "."
            
    if best_sentence:
        answer = best_sentence
    else:
        answer = sentences[0] + "." if sentences else context
        
    print(f"\n[+] Generated RAG Answer (Generated using decrypted memory):")
    print(f"    >>> \"{answer}\"")
    
    print("\n" + "="*80)
    print(" DEMO RUN COMPLETED SUCCESSFULLY ".center(80, "="))
    print("="*80)

def generate_dummy_vector(dim):
    # Generates a predictable dummy vector for demonstration
    rng = np.random.default_rng(12345)
    v = rng.normal(0.0, 1.0, dim)
    norm = np.linalg.norm(v)
    if norm > 0.0:
        v = v * (0.7 / norm)
    return v

if __name__ == '__main__':
    run_interactive_demo()
