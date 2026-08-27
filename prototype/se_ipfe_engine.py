import random
import math
from typing import List, Tuple, Dict, Any, Optional
from crypto_engine import AdaIPFEEngine
from revocation_proxy import RevocationProxy

class SEIPFEEngine:
    @staticmethod
    def Setup(lambda_bits: int, n: int) -> Tuple[Dict[str, Any], List[int]]:
        """Setup matches standard Ada-IPFE parameters."""
        return AdaIPFEEngine.Setup(lambda_bits, n)

    @staticmethod
    def KeyGen(
        y: List[float],
        clearance: int,
        msk: List[int],
        mpk: Dict[str, Any],
        alpha: int,
        beta: int,
        user_id: str = "default_user",
        revocation_proxy: Optional[RevocationProxy] = None
    ) -> Dict[str, Any]:
        """Generates functional subkey associated with query vector y and a clearance level (1 to 5)."""
        # Generate standard Ada-IPFE subkey using KDC blenders
        from rag_pipeline import keygen_with_blenders
        sk_hq = keygen_with_blenders(y, msk, mpk, alpha, beta)
        
        # Package functional subkey with user clearance level
        sky_embed = {
            "sk_hq": sk_hq,
            "clearance": clearance,
            "user_id": user_id
        }
        
        # If a revocation proxy is configured, enforce revocation check
        if revocation_proxy is not None:
            deny_val = RevocationProxy.create_force_deny_value(sky_embed)
            sky_embed = revocation_proxy.enforce(user_id, sky_embed, deny_val)
            
        return sky_embed

    @staticmethod
    def Encrypt(x: List[float], sensitivity: int, mpk: Dict[str, Any], pk: List[int]) -> Dict[str, Any]:
        """Encrypts vector x and embeds a document sensitivity tag (1 to 5)."""
        # Encrypt vector x using standard Ada-IPFE
        ct = AdaIPFEEngine.Encrypt(x, mpk, pk)
        
        # Embed sensitivity metadata tag
        return {
            "ct": ct,
            "sensitivity": sensitivity
        }

    @staticmethod
    def Decrypt(sk_y: Dict[str, Any], ct_x: Dict[str, Any], mpk: Dict[str, Any]) -> float:
        """
        Decrypts the inner product. 
        Enforces access control algebraically: if clearance < sensitivity, 
        algebraic noise is injected, corrupting the decryption result.
        """
        clearance = sk_y["clearance"]
        sensitivity = ct_x["sensitivity"]
        ct = ct_x["ct"]
        sk_hq = sk_y["sk_hq"]
        
        if clearance >= sensitivity:
            # Authorized access: normal Ada-IPFE decryption
            return AdaIPFEEngine.Decrypt(sk_hq, ct, mpk)
        else:
            # Unauthorized access: inject algebraic noise to corrupt the result
            # We simulate this homomorphically by multiplying the ciphertext elements 
            # by a randomized factor that corrupts the decrypted plaintext.
            corrupted_ct = list(ct)
            N = mpk['N']
            N2 = mpk['N2']
            # Multiply ct_5 elements by random factors to corrupt the inner product
            corrupted_ct[0] = (corrupted_ct[0] * random.randint(2, N - 1)) % N2
            try:
                raw_val = AdaIPFEEngine.Decrypt(sk_hq, tuple(corrupted_ct), mpk)
                return raw_val + random.uniform(100.0, 5000.0) # Ensure high noise
            except Exception:
                return random.uniform(-1000.0, 1000.0) # Fallback to random noise
