import unittest
import random
import time
from crypto_engine import AdaIPFEEngine

class TestAdaIPFE(unittest.TestCase):
    def test_end_to_end_inner_product(self):
        """
        MANDATORY UNIT TEST GATE: Asserts correctness of Ada-IPFE.
        Evaluates Decrypt(KeyGen(y, msk), Encrypt(x, mpk, pk_y)) == <x, y>
        over random 384-dimensional vectors.
        """
        print("\n=== Running Ada-IPFE Correctness Test ===")
        dimension = 384
        
        # 1. Generate random 384-dimensional vectors with float values
        random.seed(42)
        x = [random.uniform(-1.0, 1.0) for _ in range(dimension)]
        y = [random.uniform(-1.0, 1.0) for _ in range(dimension)]
        
        # Calculate expected inner product
        expected_dot = sum(xi * yi for xi, yi in zip(x, y))
        print(f"Calculated expected inner product: {expected_dot:.6f}")
        
        # 2. Run Setup
        # Use 256-bit primes (512-bit modulus) for fast local verification in tests
        print("Running Setup...")
        start_time = time.time()
        mpk, msk = AdaIPFEEngine.Setup(lambda_bits=256, n=dimension)
        setup_time = time.time() - start_time
        print(f"Setup complete in {setup_time:.3f} seconds. Modulus N size: {mpk['N'].bit_length()} bits.")
        
        # 3. Run KeyGen for query vector y
        print("Running KeyGen...")
        sk_y, pk_y = AdaIPFEEngine.KeyGen(y, msk, mpk)
        
        # 4. Run Encrypt for database vector x
        print("Running Encrypt...")
        start_enc = time.time()
        ct_x = AdaIPFEEngine.Encrypt(x, mpk, pk_y)
        enc_time = time.time() - start_enc
        print(f"Encryption complete in {enc_time:.3f} seconds.")
        
        # 5. Run Decrypt
        print("Running Decrypt...")
        start_dec = time.time()
        decrypted_dot = AdaIPFEEngine.Decrypt(sk_y, ct_x, mpk)
        dec_time = time.time() - start_dec
        print(f"Decryption complete in {dec_time:.3f} seconds.")
        print(f"Decrypted inner product: {decrypted_dot:.6f}")
        
        # Assert accuracy within decimal floating-point scale limits
        # Scale factor is 10^4, so inner product scale is 10^8, allowing minor rounding error
        difference = abs(decrypted_dot - expected_dot)
        print(f"Absolute difference: {difference:.8f}")
        self.assertLess(difference, 1e-3, "Decrypted inner product does not match expected value within tolerance!")
        print("=== Test Passed Successfully ===\n")

if __name__ == '__main__':
    unittest.main()
