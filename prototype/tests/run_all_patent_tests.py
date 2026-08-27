"""
Master Test Runner for all 5 SHIELD-RAG Patent Upgrades.
"""

import unittest
import sys
import os

test_dir = os.path.dirname(__file__)

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
