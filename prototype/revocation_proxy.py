# =========================================================================
# Step 1.1 Verification - Exact Function Signatures in se_ipfe_engine.py
# =========================================================================
# Output of grep -n "def " se_ipfe_engine.py:
# 8:  def Setup(lambda_bits: int, n: int) -> Tuple[Dict[str, Any], List[int]]:
# 13: def KeyGen(y: List[float], clearance: int, msk: List[int], mpk: Dict[str, Any], alpha: int, beta: int) -> Dict[str, Any]:
# 26: def Encrypt(x: List[float], sensitivity: int, mpk: Dict[str, Any], pk: List[int]) -> Dict[str, Any]:
# 38: def Decrypt(sk_y: Dict[str, Any], ct_x: Dict[str, Any], mpk: Dict[str, Any]) -> float:
# =========================================================================

import datetime
import random
from typing import Dict, Any, Optional, Set, List, Tuple


class RevocationProxy:
    """
    Lightweight semi-trusted Revocation Proxy for SE-IPFE.
    Sits between client query-key generation and the Oracle matching step,
    enforcing instant revocation by forcing maximally noised subkeys without
    corpus re-encryption or key re-issuance for other users.
    """
    def __init__(self):
        self._revoked_users: Set[str] = set()
        self._revocation_log: List[Tuple[str, str, str]] = []

    def revoke(self, user_id: str, reason: str = "unspecified") -> None:
        """Revoke user clearance instantly by adding user_id to revocation list."""
        self._revoked_users.add(user_id)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._revocation_log.append((user_id, timestamp, reason))

    def revoke_user(self, user_id: str, reason: str = "unspecified") -> None:
        """Alias for revoke()."""
        self.revoke(user_id, reason)

    def reinstate(self, user_id: str) -> None:
        """Reinstate a previously revoked user."""
        if user_id in self._revoked_users:
            self._revoked_users.remove(user_id)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._revocation_log.append((user_id, timestamp, "reinstated"))

    def reinstate_user(self, user_id: str) -> None:
        """Alias for reinstate()."""
        self.reinstate(user_id)

    def is_revoked(self, user_id: str) -> bool:
        """Check if user_id is currently revoked."""
        return user_id in self._revoked_users

    def enforce(self, user_id: str, sky_embed: Any, force_deny_value: Any) -> Any:
        """
        Intercepts subkey submission before Oracle matching.
        If user_id is revoked, returns force_deny_value (clearance=0 / max noise).
        Otherwise, returns sky_embed unchanged.
        """
        if self.is_revoked(user_id):
            return force_deny_value
        return sky_embed

    @staticmethod
    def create_force_deny_value(original_sky: Any, lambda_N: Optional[int] = None) -> Any:
        """
        Constructs a force_deny_value matching SE-IPFE's subkey structure
        with clearance set to 0 (minimum possible clearance), forcing the
        maximum clearance gap (Ld - Lc = 5 - 0 = 5) during SE-IPFE decryption,
        or adding maximal algebraic noise to sk for raw Ada-IPFE subkeys.
        """
        if isinstance(original_sky, dict):
            sk_hq = original_sky.get("sk_hq", original_sky)
            return {
                "sk_hq": sk_hq,
                "clearance": 0,
                "user_id": original_sky.get("user_id", "revoked_user")
            }
        elif isinstance(original_sky, (list, tuple)) and len(original_sky) == 3:
            # Ada-IPFE subkey: (beta, sk, y_scaled)
            beta, sk, y_scaled = original_sky
            noise = random.randint(1000000, 99999999)
            if lambda_N:
                noised_sk = (sk + noise) % lambda_N
            else:
                noised_sk = sk + noise
            return (beta, noised_sk, y_scaled)
        elif isinstance(original_sky, (list, tuple)) and len(original_sky) == 2:
            K1, K2 = original_sky
            return (K1, [k + 100000 for k in K2] if isinstance(K2, list) else K2 + 100000)
        return original_sky
