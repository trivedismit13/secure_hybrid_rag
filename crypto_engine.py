import random
import math
from typing import List, Tuple, Dict, Any
from config import L_D, SCALE_FACTOR, USE_2048_BIT_RSA

def is_prime(n: int, k: int = 10) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    # Check small primes first
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generate_safe_prime(bits: int) -> int:
    """Generates a safe prime p = 2p' + 1 where p' is also prime."""
    while True:
        # Generate a candidate prime p'
        p_prime = random.getrandbits(bits - 1)
        p_prime |= (1 << (bits - 2)) | 1  # Ensure correct bit size and odd
        if is_prime(p_prime):
            p = 2 * p_prime + 1
            if is_prime(p):
                return p

class AdaIPFEEngine:
    @staticmethod
    def Setup(lambda_bits: int, n: int) -> Tuple[Dict[str, Any], List[int]]:
        """
        Setup phase: Generates Master Public Key (mpk) and Master Secret Key (msk).
        lambda_bits: bit size for the safe primes p_tilde and q_tilde.
        n: vector dimension.
        """
        # 1. Modulus N = p_tilde * q_tilde
        p_tilde = generate_safe_prime(lambda_bits)
        q_tilde = generate_safe_prime(lambda_bits)
        N = p_tilde * q_tilde
        N2 = N * N
        
        # lambda(N) = lcm(p_tilde-1, q_tilde-1)
        p_sub = (p_tilde - 1) // 2
        q_sub = (q_tilde - 1) // 2
        lambda_N = 2 * p_sub * q_sub
        
        # 2. Generator g = (g')^(2N) mod N^2
        while True:
            g_prime = random.randint(2, N2 - 1)
            if math.gcd(g_prime, N) == 1:
                break
        g = pow(g_prime, 2 * N, N2)
        
        # 3. Master secret key s = (s_1, ..., s_n) in Z_{lambda_N}^n
        s = [random.randint(1, lambda_N - 1) for _ in range(n)]
        
        # 4. Public parameters eta_i = g^{s_i} mod N^2
        eta = [pow(g, s_i, N2) for s_i in s]
        
        mpk = {
            'g': g,
            'eta': eta,
            'N': N,
            'N2': N2,
            'lambda_N': lambda_N,
            'n': n
        }
        msk = s
        
        return mpk, msk

    @staticmethod
    def KeyGen(y: List[float], msk: List[int], mpk: Dict[str, Any]) -> Tuple[Tuple[int, int, List[int]], Tuple[int, int]]:
        """
        KeyGen phase: Generates functional subkeys sk_y and public keys pk_y.
        y: Query vector (scaled to integers).
        msk: Master secret key.
        mpk: Master public key.
        """
        lambda_N = mpk['lambda_N']
        N = mpk['N']
        N2 = mpk['N2']
        g = mpk['g']
        
        # Scale query vector y to integers
        y_scaled = [round(val * SCALE_FACTOR) for val in y]
        
        # Sample blenders
        alpha = random.randint(1, lambda_N - 1)
        beta = random.randint(1, lambda_N - 1)
        
        # Compute sk = (<s, y> + alpha + beta) mod lambda_N
        dot_s_y = sum(s_i * y_i for s_i, y_i in zip(msk, y_scaled))
        sk = (dot_s_y + alpha + beta) % lambda_N
        
        sk_y = (beta, sk, y_scaled)
        
        # Compute pk_1 = g^alpha mod N^2, pk_2 = g^beta mod N^2
        pk_1 = pow(g, alpha, N2)
        pk_2 = pow(g, beta, N2)
        pk_y = (pk_1, pk_2)
        
        return sk_y, pk_y

    @staticmethod
    def Encrypt(x: List[float], mpk: Dict[str, Any], pk_y: Tuple[int, int]) -> Tuple[int, int, int, int, int, List[int]]:
        """
        Encrypt phase: Encrypts knowledge database vector x.
        x: Database vector (floats).
        mpk: Master public key.
        pk_y: Query-associated public keys pk_1, pk_2.
        """
        N = mpk['N']
        N2 = mpk['N2']
        g = mpk['g']
        lambda_N = mpk['lambda_N']
        eta = mpk['eta']
        n = mpk['n']
        pk_1, pk_2 = pk_y
        
        # Scale database vector x to integers
        x_scaled = []
        for val in x:
            scaled = round(val * SCALE_FACTOR)
            # Map negative values to Z_N
            x_scaled.append(scaled % N)
            
        # Sample r, a, b, c
        r = random.randint(1, N // 4)
        while True:
            a = random.randint(2, lambda_N - 1)
            if math.gcd(a, lambda_N) == 1:
                break
        b = random.randint(1, lambda_N - 1)
        c = random.randint(1, lambda_N - 1)
        
        # Compute sigma = pk_2^c mod N
        sigma = pow(pk_2, c, N)
        
        # Compute ciphertexts
        ct_0 = a
        ct_1 = pow(g, r, N2)
        ct_2 = pow(g, b, N2)
        ct_3 = (c * a - b) % lambda_N
        ct_4 = pow(pk_1, r, N2)
        
        ct_5 = []
        for i in range(n):
            # ct_i = eta_i^r * (1 + N * sigma * x_i) mod N^2
            term1 = pow(eta[i], r, N2)
            term2 = (1 + N * sigma * x_scaled[i]) % N2
            ct_i = (term1 * term2) % N2
            ct_5.append(ct_i)
            
        return (ct_0, ct_1, ct_2, ct_3, ct_4, ct_5)

    @staticmethod
    def Decrypt(sk_y: Tuple[int, int, List[int]], ct_x: Tuple[int, int, int, int, int, List[int]], mpk: Dict[str, Any]) -> float:
        """
        Decrypt phase: Decrypts the functional inner product.
        sk_y: Query-associated subkey (beta, sk, y_scaled).
        ct_x: Ciphertext vector (ct_0, ct_1, ct_2, ct_3, ct_4, ct_5).
        mpk: Master public key.
        """
        N = mpk['N']
        N2 = mpk['N2']
        g = mpk['g']
        lambda_N = mpk['lambda_N']
        
        beta, sk, y_scaled = sk_y
        ct_0, ct_1, ct_2, ct_3, ct_4, ct_5 = ct_x
        
        # 1. Compute tau_1 = ((ct_2 * g^ct_3)^(beta * ct_0^-1 mod lambda_N) mod N^2) mod N
        a_inv = pow(ct_0, -1, lambda_N)
        exp = (beta * a_inv) % lambda_N
        
        g_ct3 = pow(g, ct_3, N2)
        base = (ct_2 * g_ct3) % N2
        tau_1 = pow(base, exp, N2) % N
        
        # 2. Compute numerator: ct_1^(beta - sk mod lambda_N) * ct_4 * prod(ct_5_i ^ y_i) mod N^2
        exp_ct1 = (beta - sk) % lambda_N
        t1 = pow(ct_1, exp_ct1, N2)
        t2 = ct_4
        
        t3 = 1
        for i in range(len(y_scaled)):
            t3 = (t3 * pow(ct_5[i], y_scaled[i], N2)) % N2
            
        num_product = (t1 * t2 * t3) % N2
        
        # 3. Paillier extraction: L(V) = (V - 1) // N
        L_val = (num_product - 1) // N
        
        # 4. Recover inner product mod N: L_val * tau_1^-1 mod N
        tau_1_inv = pow(tau_1, -1, N)
        dot_scaled = (L_val * tau_1_inv) % N
        
        # Map back to signed range
        if dot_scaled > N // 2:
            dot_scaled -= N
            
        # Scale back from 10^{2 * l_D}
        return dot_scaled / (SCALE_FACTOR * SCALE_FACTOR)
