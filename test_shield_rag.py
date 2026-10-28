"""
Root-level test runner shim for Phase 1 Step 2:
`pytest test_shield_rag.py -v`
"""
import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tests.test_shield_rag import *
