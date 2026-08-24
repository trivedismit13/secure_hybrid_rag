"""
Ada-IPFE (Adaptive Inner-Product Functional Encryption).

Implements the DCR (Decisional Composite Residuosity) based construction
for inner-product functional encryption, matching the CipheRAG specification.

Mathematical Construction:
- Setup: Generate RSA modulus N = p*q. Generator g in Z*_{N^2}.
         Master secret key s = (s_1, ..., s_n) drawn randomly.
         Public parameters {h_i = g^{s_i} mod N^2}.
- KeyGen: Given query vector y, functional key sk_y = <s, y>.
- Encrypt: Given embedding vector x, choose random r.
           ct_0 = g^r mod N^2
           ct_i = h_i^r * (1+N)^{x_i} mod N^2
- Decrypt: Numerator = prod(ct_i^{y_i}) mod N^2
           Denominator = ct_0^{sk_y} mod N^2
           Value = Numerator * Denominator^{-1} mod N^2
           (Value - 1) / N = <x, y> mod N

Float embeddings are quantized to integers via a SCALE factor before encryption.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import gmpy2
from Crypto.Util import number


@dataclass
class MasterPublicKey:
    n: int
    n_sq: int
    g: int
    h: list[int]
    scale: int


@dataclass
class MasterSecretKey:
    s: list[int]


@dataclass
class FunctionalKey:
    sk_y: int
    y: list[int]  # The quantized query vector


@dataclass
class Ciphertext:
    ct_0: int
    ct: list[int]


class AdaIPFE:
    """
    Adaptive Inner-Product Functional Encryption over the Paillier group.
    
    Used for privacy-preserving similarity search in Component B.
    """

    def __init__(self, key_size: int = 1024, scale: int = 1000) -> None:
        """
        Args:
            key_size: Bit length of the modulus N. (Default 1024 for testing speed).
            scale:    Scaling factor to quantize float embeddings to integers.
        """
        self.key_size = key_size
        self.scale = scale

    def setup(self, dimension: int) -> tuple[MasterPublicKey, MasterSecretKey]:
        """Generate master public and secret keys for a given vector dimension."""
        # Generate safe primes for N to ensure large order
        p = number.getPrime(self.key_size // 2)
        q = number.getPrime(self.key_size // 2)
        n = p * q
        n_sq = n * n

        # Generator g of Z*_{N^2}. Random element in [1, n_sq-1] coprime to n_sq.
        while True:
            g = random.randrange(1, n_sq)
            if gmpy2.gcd(g, n_sq) == 1:
                break

        # Master secret key components
        s = [random.randrange(1, n) for _ in range(dimension)]
        
        # Public parameters h_i = g^{s_i} mod N^2
        h = [int(pow(g, si, n_sq)) for si in s]

        mpk = MasterPublicKey(n=n, n_sq=n_sq, g=g, h=h, scale=self.scale)
        msk = MasterSecretKey(s=s)
        return mpk, msk

    def keygen(self, msk: MasterSecretKey, y: list[float]) -> FunctionalKey:
        """Derive a functional decryption key for a query vector y."""
        if len(y) != len(msk.s):
            raise ValueError(f"Query vector dimension {len(y)} does not match MSK {len(msk.s)}")
        
        # Quantize query vector
        y_int = [int(round(v * self.scale)) for v in y]
        
        # Compute inner product <s, y_int> over the integers
        # (Technically can be mod N*phi(N) if we know the group order, but integer math is safer 
        # since it's just used as an exponent in Z*_{N^2})
        sk_y = sum(si * yi for si, yi in zip(msk.s, y_int))
        
        return FunctionalKey(sk_y=sk_y, y=y_int)

    def encrypt(self, mpk: MasterPublicKey, x: list[float]) -> Ciphertext:
        """Encrypt a knowledge embedding vector x."""
        if len(x) != len(mpk.h):
            raise ValueError(f"Vector dimension {len(x)} does not match MPK {len(mpk.h)}")
        
        # Quantize embedding
        x_int = [int(round(v * mpk.scale)) for v in x]
        
        # Random r in [1, N/4]
        r = random.randrange(1, mpk.n // 4)
        
        # ct_0 = g^r mod N^2
        ct_0 = int(pow(mpk.g, r, mpk.n_sq))
        
        # ct_i = h_i^r * (1 + N)^{x_i} mod N^2
        # Note: (1 + N)^{x_i} mod N^2 = (1 + x_i * N) mod N^2
        ct = []
        for i in range(len(x_int)):
            h_r = pow(mpk.h[i], r, mpk.n_sq)
            # Handle negative x_i properly for (1 + x_i*N) mod N^2
            factor = (1 + x_int[i] * mpk.n) % mpk.n_sq
            ct_i = (h_r * factor) % mpk.n_sq
            ct.append(int(ct_i))
            
        return Ciphertext(ct_0=ct_0, ct=ct)

    def decrypt(self, mpk: MasterPublicKey, func_key: FunctionalKey, ciphertext: Ciphertext) -> float:
        """Decrypt to recover the inner product <x, y>."""
        if len(func_key.y) != len(ciphertext.ct):
            raise ValueError("Ciphertext and functional key dimensions mismatch")

        # Numerator: prod_{i=1}^n ct_i^{y_i} mod N^2
        num = 1
        for ct_i, y_i in zip(ciphertext.ct, func_key.y):
            # If y_i is negative, compute modular inverse first
            if y_i < 0:
                ct_i_inv = int(gmpy2.invert(ct_i, mpk.n_sq))
                term = pow(ct_i_inv, abs(y_i), mpk.n_sq)
            else:
                term = pow(ct_i, y_i, mpk.n_sq)
            num = (num * term) % mpk.n_sq

        # Denominator: ct_0^{sk_y} mod N^2
        if func_key.sk_y < 0:
            ct_0_inv = int(gmpy2.invert(ciphertext.ct_0, mpk.n_sq))
            den = pow(ct_0_inv, abs(func_key.sk_y), mpk.n_sq)
        else:
            den = pow(ciphertext.ct_0, func_key.sk_y, mpk.n_sq)

        den_inv = int(gmpy2.invert(den, mpk.n_sq))
        
        # Value = Numerator * Denominator^{-1} mod N^2
        val = (num * den_inv) % mpk.n_sq
        
        # The value should be of the form (1 + <x, y> * N) mod N^2
        # So <x, y> = (val - 1) / N mod N
        inner_prod_int = ((val - 1) // mpk.n) % mpk.n
        
        # Handle negative inner products (values in upper half of Z_N)
        if inner_prod_int > mpk.n // 2:
            inner_prod_int -= mpk.n
            
        # De-quantize: divide by scale^2
        return float(inner_prod_int) / (mpk.scale * mpk.scale)

    def serialize_ciphertext(self, ct: Ciphertext) -> bytes:
        """Serialize a ciphertext into bytes for the wire format."""
        # Format: [4B dim][len_0][ct_0][len_1][ct_1]...
        import struct
        
        def encode_int(val: int) -> bytes:
            b = val.to_bytes((val.bit_length() + 7) // 8 or 1, 'big')
            return struct.pack("!I", len(b)) + b
            
        parts = [struct.pack("!I", len(ct.ct)), encode_int(ct.ct_0)]
        for c in ct.ct:
            parts.append(encode_int(c))
            
        return b"".join(parts)

    def deserialize_ciphertext(self, data: bytes) -> Ciphertext:
        """Deserialize bytes into a Ciphertext."""
        import struct
        
        offset = 0
        (dim,) = struct.unpack_from("!I", data, offset)
        offset += 4
        
        def decode_int() -> int:
            nonlocal offset
            (length,) = struct.unpack_from("!I", data, offset)
            offset += 4
            val = int.from_bytes(data[offset:offset+length], 'big')
            offset += length
            return val
            
        ct_0 = decode_int()
        ct = [decode_int() for _ in range(dim)]
        return Ciphertext(ct_0=ct_0, ct=ct)
