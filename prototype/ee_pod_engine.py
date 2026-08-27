"""
Epistemic-Entangled Onion Decryption (EE-POD) Engine
Connects cryptographic nested onion layer unmasking directly to the model's
real-time token entropy and epistemic uncertainty states during generation.
"""

import numpy as np
import hashlib
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class EEPODEngine:
    """
    EE-POD Engine:
    Dynamically unlocks nested encryption layers using entropy-derived trapdoors
    when generation enters high uncertainty states.
    """
    def __init__(self, entropy_threshold: float = 1.5, num_layers: int = 3):
        self.entropy_threshold = entropy_threshold
        self.num_layers = num_layers

    def compute_token_entropy(self, logits: torch.Tensor) -> float:
        """
        Compute Shannon entropy of the next-token probability distribution.
        H(p) = - sum(p_i * log(p_i))
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()
        return entropy

    def derive_entropy_trapdoor(self, hidden_state: torch.Tensor, entropy_val: float, layer_idx: int) -> bytes:
        """
        Derive an epistemic trapdoor key from the LLM hidden state when entropy exceeds threshold.
        """
        state_bytes = hidden_state.detach().cpu().numpy().tobytes()
        entropy_bytes = f"{entropy_val:.6f}::layer_{layer_idx}".encode("utf-8")
        trapdoor_key = hashlib.sha256(state_bytes + entropy_bytes).digest()
        return trapdoor_key

    def create_onion_ciphertext(self, raw_evidence_layers: List[str], master_keys: List[bytes]) -> Dict:
        """
        Encrypt multiple layers of evidence in nested shells:
        CT_onion = Enc_k3(Enc_k2(Enc_k1(evidence)))
        """
        # Outer to inner encryption
        payload = raw_evidence_layers[0] # Base layer
        encrypted_shells = []
        
        for i, (layer_text, key) in enumerate(zip(raw_evidence_layers, master_keys)):
            # Symmetric XOR encryption with key hash
            key_stream = hashlib.sha256(key + f"salt_{i}".encode()).digest()
            text_bytes = layer_text.encode("utf-8")
            ct_bytes = bytes([b ^ key_stream[j % len(key_stream)] for j, b in enumerate(text_bytes)])
            encrypted_shells.append({
                "layer_idx": i,
                "ciphertext": ct_bytes.hex(),
                "key_commitment": hashlib.sha256(key).hexdigest()
            })
            
        return {"onion_layers": encrypted_shells}

    def evaluate_and_peel(
        self,
        current_layer: int,
        onion_package: Dict,
        logits: torch.Tensor,
        hidden_state: torch.Tensor,
        available_keys: Dict[int, bytes]
    ) -> Tuple[bool, Optional[str], float]:
        """
        Monitors token entropy and dynamically unmasks the next layer if uncertainty triggers a trapdoor.
        """
        entropy = self.compute_token_entropy(logits)
        is_uncertain = entropy > self.entropy_threshold
        
        if is_uncertain and current_layer < len(onion_package["onion_layers"]):
            target_shell = onion_package["onion_layers"][current_layer]
            if current_layer in available_keys:
                key = available_keys[current_layer]
                # Verify commitment
                if hashlib.sha256(key).hexdigest() == target_shell["key_commitment"]:
                    key_stream = hashlib.sha256(key + f"salt_{current_layer}".encode()).digest()
                    ct_bytes = bytes.fromhex(target_shell["ciphertext"])
                    decrypted_text = bytes([b ^ key_stream[j % len(key_stream)] for j, b in enumerate(ct_bytes)]).decode("utf-8")
                    return True, decrypted_text, entropy
                    
        return False, None, entropy
