import numpy as np
import torch
import math
import torch.nn as nn
import json
import hashlib
from typing import List, Tuple, Dict, Any
from crypto_engine import AdaIPFEEngine
from alsh_engine import ALSHEngine
from ipfs_mock import IPFSMock
from blockchain.contract_helper import BlockchainSimulator

class VPRAGPipeline:
    def __init__(self, hidden_dim: int = 768, K: int = 128, lambda_bits: int = 256):
        """
        Efficient Vector-Multiplicative Privacy-Preserving RAG Pipeline.
        hidden_dim: Hidden dimension of the LLM (e.g., 768 for GPT-2).
        K: ALSH signature dimension (number of hyperplanes).
        lambda_bits: Bit size of safe primes for Ada-IPFE setup.
        """
        self.hidden_dim = hidden_dim
        self.K = K
        self.lambda_bits = lambda_bits
        
        print(f"Initializing V-PPRAG Pipeline (Dim: {hidden_dim}, ALSH-K: {K})...")
        
        # 1. Initialize Cryptographic Engines
        # Setup for ALSH Hash signatures (dimension K)
        self.mpk_h, self.msk_h = AdaIPFEEngine.Setup(lambda_bits, K)
        # Setup for full embeddings (dimension hidden_dim)
        self.mpk_e, self.msk_e = AdaIPFEEngine.Setup(lambda_bits, hidden_dim)
        
        # System-wide blenders and public keys (for database pre-encryption)
        # Hash blenders
        self.alpha_h = random_blender(self.mpk_h['lambda_N'])
        self.beta_h = random_blender(self.mpk_h['lambda_N'])
        self.pk_h = (
            pow(self.mpk_h['g'], self.alpha_h, self.mpk_h['N2']),
            pow(self.mpk_h['g'], self.beta_h, self.mpk_h['N2'])
        )
        
        # Embedding blenders
        self.alpha_e = random_blender(self.mpk_e['lambda_N'])
        self.beta_e = random_blender(self.mpk_e['lambda_N'])
        self.pk_e = (
            pow(self.mpk_e['g'], self.alpha_e, self.mpk_e['N2']),
            pow(self.mpk_e['g'], self.beta_e, self.mpk_e['N2'])
        )
        
        # 2. Initialize ALSH Engine
        self.alsh = ALSHEngine(d=hidden_dim, K=K)
        
        # 3. Initialize Storage
        self.ipfs = IPFSMock()
        self.blockchain = BlockchainSimulator(oracle_address="0xOracleAddress")
        
        # Text database mapping CID to raw text for generation lookup
        self.text_db: Dict[str, str] = {}
        
        # Precompute Attention Gateway Subkeys (Algorithm 2 setup)
        self._init_attention_weights()

    def _init_attention_weights(self):
        """Simulates LLM self-attention weight matrices (W_Q, W_K, W_V)."""
        # Create random orthogonal-like weight matrices for early layers
        # Shape: hidden_dim x hidden_dim
        self.W_Q = np.random.normal(0.0, 0.02, (self.hidden_dim, self.hidden_dim))
        self.W_K = np.random.normal(0.0, 0.02, (self.hidden_dim, self.hidden_dim))
        self.W_V = np.random.normal(0.0, 0.02, (self.hidden_dim, self.hidden_dim))
        
        # Generate row-wise functional subkeys for W_K and W_V (precomputed by KDC)
        # Each row of W_K and W_V is a query vector in the Ada-IPFE KeyGen sense.
        print("Pre-generating attention weight functional subkeys...")
        self.sk_W_K = []
        self.sk_W_V = []
        
        # Decryptor needs subkeys to compute y_j = <x, w_j>
        # To speed up initialization, we can generate a subset or compute on the fly,
        # but here we generate them for the hidden_dim.
        # Note: In production, these are loaded from secure enclave.
        # We generate them using the master secret key (msk_e) and embedding parameters.
        # To avoid slow startup, we generate on-demand or mock them in benchmarks,
        # but let's provide the generator method.

    def generate_attention_subkeys(self, num_rows: int) -> Tuple[List[Any], List[Any]]:
        """Generates row functional subkeys for W_K and W_V up to num_rows."""
        sk_K = []
        sk_V = []
        for j in range(num_rows):
            # Row weights w_j
            w_K_j = self.W_K[j].tolist()
            w_V_j = self.W_V[j].tolist()
            
            # KeyGen(y, msk, mpk)
            # Use the same blenders alpha_e, beta_e so it matches pk_e
            sk_K_j = keygen_with_blenders(w_K_j, self.msk_e, self.mpk_e, self.alpha_e, self.beta_e)
            sk_V_j = keygen_with_blenders(w_V_j, self.msk_e, self.mpk_e, self.alpha_e, self.beta_e)
            
            sk_K.append(sk_K_j)
            sk_V.append(sk_V_j)
        return sk_K, sk_V

    def upload_knowledge_base(self, corpus_id: str, doc_embeddings: List[np.ndarray], doc_texts: List[str]):
        """
        Preprocesses, encrypts, and uploads knowledge vectors to IPFS and blockchain.
        """
        assert len(doc_embeddings) == len(doc_texts), "Mismatched embeddings and text size"
        print(f"Uploading corpus '{corpus_id}' ({len(doc_texts)} items) to dual storage...")
        
        corpus_id_bytes = corpus_id.encode('utf-8')
        encrypted_signatures = []
        cids = []
        
        for i, (emb, text) in enumerate(zip(doc_embeddings, doc_texts)):
            # 1. ALSH P-Transformation & hashing
            p_emb = self.alsh.p_transform(emb)
            H_v = self.alsh.hash_vector(p_emb)
            
            # 2. Encrypt hash signature H_v under Ada-IPFE
            # H_v components are in {-1, 1}
            H_v_float = [float(val) for val in H_v]
            ct_h = AdaIPFEEngine.Encrypt(H_v_float, self.mpk_h, self.pk_h)
            
            # Serialize ct_h to bytes for Solidity contract
            ct_h_bytes = serialize_ciphertext(ct_h)
            encrypted_signatures.append(ct_h_bytes)
            
            # 3. Encrypt full embedding vector under Ada-IPFE
            emb_float = [float(val) for val in emb]
            ct_e = AdaIPFEEngine.Encrypt(emb_float, self.mpk_e, self.pk_e)
            
            # 4. Upload full embedding to IPFS
            cid = self.ipfs.upload(ct_e)
            cids.append(cid)
            self.text_db[cid] = text
            
        # 5. Upload encrypted signatures and CIDs to Solidity Contract
        contract = self.blockchain.get_contract()
        contract.uploadCorpus(corpus_id_bytes, encrypted_signatures, cids)
        print(f"Corpus '{corpus_id}' successfully uploaded.")

    def query(self, corpus_id: str, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        ALGORITHM 1: Encrypted Hash Knowledge Retrieval
        Preprocesses query, submits to blockchain, runs oracle match, and fetches Top-k results.
        """
        # --- Client Node (Trusted) ---
        # 1. ALSH Q-Transformation
        q_trans = self.alsh.q_transform(query_vector)
        H_q = self.alsh.hash_vector(q_trans)
        H_q_float = [float(val) for val in H_q]
        
        # 2. KeyGen for query signature H_q using same system blenders alpha_h, beta_h
        sk_hq = keygen_with_blenders(H_q_float, self.msk_h, self.mpk_h, self.alpha_h, self.beta_h)
        
        # Serialize parameters to submit to contract
        query_id = hashlib.sha256(np.random.bytes(16)).digest()
        corpus_id_bytes = corpus_id.encode('utf-8')
        sk_hq_bytes = serialize_subkey(sk_hq)
        encrypted_query_hash_bytes = json.dumps(H_q_float).encode('utf-8') # dummy audit tag
        
        # 3. Submit Query to Blockchain
        contract = self.blockchain.get_contract()
        contract.submitQuery(query_id, corpus_id_bytes, sk_hq_bytes, encrypted_query_hash_bytes)
        
        # --- Off-chain Oracle Worker (Honest-but-Curious) ---
        # 4. Oracle detects QuerySubmitted event
        events = self.blockchain.get_events()
        query_event = [e for e in events if e['event'] == 'QuerySubmitted'][-1]
        
        # Extract query details
        evt_query_id = bytes.fromhex(query_event['args']['queryId'])
        evt_corpus_id = bytes.fromhex(query_event['args']['corpusId'])
        evt_sk_hq = deserialize_subkey(bytes.fromhex(query_event['args']['sk_q']))
        
        # Retrieve encrypted corpus signatures from contract
        enc_sigs, corpus_cids = contract.getCorpus(evt_corpus_id)
        
        # Perform encrypted hash signature matching
        scores = []
        for ct_h_bytes, cid in zip(enc_sigs, corpus_cids):
            ct_h = deserialize_ciphertext(ct_h_bytes)
            # Decrypt inner product s_i = <H(v_i), H(q)>
            dot_product = AdaIPFEEngine.Decrypt(evt_sk_hq, ct_h, self.mpk_h)
            # Compute ALSH similarity
            similarity = dot_product / self.K
            scores.append((cid, similarity))
            
        # Rank Top-k results
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k_results = scores[:top_k]
        top_k_cids = [item[0] for item in top_k_results]
        
        # Commit result audit hash to blockchain
        audit_string = "".join(top_k_cids).encode('utf-8')
        audit_hash = hashlib.sha256(audit_string).digest()
        
        contract.submitTopKResults("0xOracleAddress", evt_query_id, top_k_cids, audit_hash)
        
        # --- Client Node (Trusted) ---
        # 5. Retrieve result CIDs from contract
        processed, result_cids, contract_audit = contract.getQueryResult(query_id)
        assert processed, "Query was not processed by Oracle!"
        assert contract_audit == audit_hash, "Audit hash mismatch!"
        
        # Return CIDs and their match scores
        return [(cid, score) for cid, score in scores[:top_k]]

    def decrypt_embedding_matrix(self, cids: List[str], sk_rows_K: List[Any], sk_rows_V: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        ALGORITHM 2: Decryption-Enabled Self-Attention Gateway
        Decrypts the IPFS-stored ciphertexts into Query-Key-Value projection states.
        cids: list of CIDs for retrieved documents.
        sk_rows_K: row-wise functional subkeys for W_K.
        sk_rows_V: row-wise functional subkeys for W_V.
        """
        k = len(cids)
        d = self.hidden_dim
        
        # Initialize key and value state matrices
        K_proj = np.zeros((k, d))
        V_proj = np.zeros((k, d))
        
        # Process each retrieved ciphertext vector from IPFS
        for i, cid in enumerate(cids):
            ct_e = self.ipfs.fetch(cid)
            
            # Row-wise projection decryption
            for j in range(d):
                # K_proj[i, j] = <x_i, w_K_j> = Dec(sk_K_j, Enc(x_i))
                K_proj[i, j] = AdaIPFEEngine.Decrypt(sk_rows_K[j], ct_e, self.mpk_e)
                # V_proj[i, j] = <x_i, w_V_j> = Dec(sk_V_j, Enc(x_i))
                V_proj[i, j] = AdaIPFEEngine.Decrypt(sk_rows_V[j], ct_e, self.mpk_e)
                
        return K_proj, V_proj

    def execute_self_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Executes standard self-attention over the decrypted transient states.
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
        """
        # Q: Batch x Seq_Q x Dim
        # K: Batch x Seq_K x Dim
        # V: Batch x Seq_K x Dim
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output

# Helper Cryptographic Functions

def random_blender(modulus: int) -> int:
    return random_integer_in_range(1, modulus - 1)

def random_integer_in_range(low: int, high: int) -> int:
    return random_integer(high - low) + low

def random_integer(limit: int) -> int:
    import random
    return random.randint(0, limit)

def keygen_with_blenders(y: List[float], msk: List[int], mpk: Dict[str, Any], alpha: int, beta: int) -> Tuple[int, int, List[int]]:
    """Helper KeyGen using pre-defined blenders (to align database and query keys)."""
    lambda_N = mpk['lambda_N']
    from config import SCALE_FACTOR
    y_scaled = [round(val * SCALE_FACTOR) for val in y]
    dot_s_y = sum(s_i * y_i for s_i, y_i in zip(msk, y_scaled))
    sk = (dot_s_y + alpha + beta) % lambda_N
    return (beta, sk, y_scaled)

# Serialization Utilities for simulated smart contract interaction

def serialize_ciphertext(ct: Tuple[int, int, int, int, int, List[int]]) -> bytes:
    payload = {
        'ct_0': ct[0],
        'ct_1': ct[1],
        'ct_2': ct[2],
        'ct_3': ct[3],
        'ct_4': ct[4],
        'ct_5': ct[5]
    }
    return json.dumps(payload).encode('utf-8')

def deserialize_ciphertext(b: bytes) -> Tuple[int, int, int, int, int, List[int]]:
    payload = json.loads(b.decode('utf-8'))
    return (
        payload['ct_0'],
        payload['ct_1'],
        payload['ct_2'],
        payload['ct_3'],
        payload['ct_4'],
        payload['ct_5']
    )

def serialize_subkey(sk: Tuple[int, int, List[int]]) -> bytes:
    payload = {
        'beta': sk[0],
        'sk': sk[1],
        'y_scaled': sk[2]
    }
    return json.dumps(payload).encode('utf-8')

def deserialize_subkey(b: bytes) -> Tuple[int, int, List[int]]:
    payload = json.loads(b.decode('utf-8'))
    return (
        payload['beta'],
        payload['sk'],
        payload['y_scaled']
    )
