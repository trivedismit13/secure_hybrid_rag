import hashlib
import json
from typing import Tuple, List, Dict, Any

class IPFSMock:
    def __init__(self):
        """Simulates off-chain IPFS storage for encrypted vector embeddings."""
        self.store: Dict[str, Tuple[int, int, int, int, int, List[int]]] = {}

    def upload(self, encrypted_vector: Tuple[int, int, int, int, int, List[int]]) -> str:
        """
        Encrypts and stores a vector embedding, returning its content-addressable CID.
        encrypted_vector: The ct_x tuple from Ada-IPFE encryption.
        """
        # Serialize the tuple to JSON string to generate a unique hash
        serialized = json.dumps({
            'ct_0': encrypted_vector[0],
            'ct_1': encrypted_vector[1],
            'ct_2': encrypted_vector[2],
            'ct_3': encrypted_vector[3],
            'ct_4': encrypted_vector[4],
            'ct_5': encrypted_vector[5]
        }, sort_keys=True)
        
        # Generate SHA-256 hash as the CID
        cid_hash = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        cid = f"ipfs://Qm{cid_hash[:46]}"
        
        # Store the payload
        self.store[cid] = encrypted_vector
        return cid

    def fetch(self, cid: str) -> Tuple[int, int, int, int, int, List[int]]:
        """
        Retrieves the encrypted embedding ciphertext matching the CID.
        """
        if cid not in self.store:
            raise KeyError(f"CID {cid} not found in IPFS storage.")
        return self.store[cid]

    def size(self) -> int:
        return len(self.store)
