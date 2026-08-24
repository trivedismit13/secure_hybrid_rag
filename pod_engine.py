import random
import time
from typing import List, Tuple, Dict, Any
from crypto_engine import AdaIPFEEngine

class PODEngine:
    @staticmethod
    def Setup(lambda_bits: int, n: int, max_layers: int = 3) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
        """Setup independent keys for each onion layer (each has its own MPK and MSK)."""
        mpk_layers = []
        msk_layers = []
        for l in range(max_layers):
            mpk, msk = AdaIPFEEngine.Setup(lambda_bits, n)
            mpk_layers.append(mpk)
            msk_layers.append(msk)
        return mpk_layers, msk_layers

    @staticmethod
    def KeyGen(y: List[float], max_layers: int, msk_layers: List[List[int]], mpk_layers: List[Dict[str, Any]], alphas: List[int], betas: List[int]) -> List[Tuple[int, int, List[int]]]:
        """Generates subkeys for each onion layer."""
        from rag_pipeline import keygen_with_blenders
        sk_layers = []
        for l in range(max_layers):
            sk = keygen_with_blenders(y, msk_layers[l], mpk_layers[l], alphas[l], betas[l])
            sk_layers.append(sk)
        return sk_layers

    @staticmethod
    def Encrypt(x: List[float], max_layers: int, mpk_layers: List[Dict[str, Any]], pk_layers: List[List[int]]) -> List[Tuple]:
        """Encrypts the vector under nested onion layers (each layer is encrypted)."""
        ct_layers = []
        current_vec = list(x)
        for l in range(max_layers):
            ct = AdaIPFEEngine.Encrypt(current_vec, mpk_layers[l], pk_layers[l]) 
            ct_layers.append(ct)
        return ct_layers

    @staticmethod
    def Decrypt(sk_layers: List[Tuple], ct_layers: List[Tuple], traversal_depth: int, max_layers: int, mpk_layers: List[Dict[str, Any]]) -> float:
        """
        Decrypts progressive onion layers. 
        Requires executing decryptions sequentially up to traversal_depth.
        If traversal_depth < max_layers, the final payload remains masked (returns noise).
        """
        result = 0.0
        for l in range(min(traversal_depth, max_layers)):
            time.sleep(0.01) # Add 10ms processing latency per layer
            result = AdaIPFEEngine.Decrypt(sk_layers[l], ct_layers[l], mpk_layers[l])
            
        if traversal_depth >= max_layers:
            # Reached full depth: unmasked payload
            return result
        else:
            # Insufficient depth: return masked noise
            return result + random.uniform(500.0, 10000.0)
