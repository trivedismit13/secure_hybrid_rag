import os

# Floating-point decimal scaling factor
# Scale decimal floats to integers via x' = 10^{l_D} x
L_D = 4
SCALE_FACTOR = 10 ** L_D

# Flag to switch between fast test runs (512-bit RSA) and fully secure benchmarks (2048-bit RSA)
# Safe prime generation for 2048-bit RSA is computationally expensive in pure Python.
USE_2048_BIT_RSA = False

# ALSH Configuration
ALSH_K = 128  # Number of random hyperplanes
ALSH_M = 3    # Dimension mapping power exponent (P-transformation uses 2^m components)
ALSH_U = 0.83 # Normalization bound for knowledge vectors (||v_i||_2 <= U)

# Model configuration
# Benchmarks: 'gpt2' (768 dim, 12 layers), 'qwen2' (3584 dim), 'llama3' (4096 dim)
MODEL_NAME = 'gpt2'

# Hardware configuration
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'

# Data and output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
