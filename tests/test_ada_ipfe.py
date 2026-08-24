"""
Tests for Ada-IPFE.
Verifies Setup, KeyGen, Encrypt, and Decrypt properties for DCR-based IPFE.
"""
import pytest
import math
import random
from shield_rag.crypto.ada_ipfe import AdaIPFE, MasterPublicKey, MasterSecretKey, FunctionalKey, Ciphertext

class TestAdaIPFE:
    @pytest.fixture(scope="class")
    def ipfe_small(self):
        # Use small key size for fast tests (512-bit modulus)
        return AdaIPFE(key_size=512, scale=100)

    def test_setup(self, ipfe_small):
        dim = 16
        mpk, msk = ipfe_small.setup(dimension=dim)
        assert len(mpk.h) == dim
        assert len(msk.s) == dim
        assert mpk.n > 0
        assert mpk.n_sq == mpk.n * mpk.n

    def test_keygen(self, ipfe_small):
        dim = 16
        mpk, msk = ipfe_small.setup(dimension=dim)
        y = [random.random() * 2 - 1 for _ in range(dim)]  # random floats in [-1, 1]
        func_key = ipfe_small.keygen(msk, y)
        assert len(func_key.y) == dim
        assert isinstance(func_key.sk_y, int)

    def test_encrypt_decrypt(self, ipfe_small):
        dim = 16
        mpk, msk = ipfe_small.setup(dimension=dim)
        
        # Test 5 random vectors
        for _ in range(5):
            x = [random.random() * 2 - 1 for _ in range(dim)]
            y = [random.random() * 2 - 1 for _ in range(dim)]
            
            ct = ipfe_small.encrypt(mpk, x)
            sk_y = ipfe_small.keygen(msk, y)
            
            result = ipfe_small.decrypt(mpk, sk_y, ct)
            
            # Since we quantized, there's a quantization error
            # x_int = x * 100, y_int = y * 100
            # expected_int = sum(int(xi*100) * int(yi*100))
            expected_int = sum(int(round(xi * ipfe_small.scale)) * int(round(yi * ipfe_small.scale)) for xi, yi in zip(x, y))
            expected = expected_int / (ipfe_small.scale**2)
            
            assert math.isclose(result, expected, rel_tol=1e-5, abs_tol=1e-5)

    def test_serialization(self, ipfe_small):
        dim = 4
        mpk, msk = ipfe_small.setup(dimension=dim)
        x = [0.1, 0.2, 0.3, -0.4]
        ct = ipfe_small.encrypt(mpk, x)
        
        data = ipfe_small.serialize_ciphertext(ct)
        restored_ct = ipfe_small.deserialize_ciphertext(data)
        
        assert ct.ct_0 == restored_ct.ct_0
        assert ct.ct == restored_ct.ct

    def test_large_dimensions(self):
        # Test with the actual target dimension 384 and 1024-bit key, but only 1 iteration
        ipfe = AdaIPFE(key_size=1024, scale=1000)
        dim = 64 # Use 64 for a slightly faster test of larger dimensions. 384 might be slow in Python tests.
        mpk, msk = ipfe.setup(dimension=dim)
        
        x = [random.random() * 2 - 1 for _ in range(dim)]
        y = [random.random() * 2 - 1 for _ in range(dim)]
        
        ct = ipfe.encrypt(mpk, x)
        sk_y = ipfe.keygen(msk, y)
        result = ipfe.decrypt(mpk, sk_y, ct)
        
        expected_int = sum(int(round(xi * ipfe.scale)) * int(round(yi * ipfe.scale)) for xi, yi in zip(x, y))
        expected = expected_int / (ipfe.scale**2)
        
        assert math.isclose(result, expected, rel_tol=1e-5, abs_tol=1e-5)
