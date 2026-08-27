"""
Unit test for Novelty #4: Empirical Indistinguishability of SE-IPFE Noise Gate.
"""

import unittest
import sys
import os

# Add prototype directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from empirical_indistinguishability_test import run_indistinguishability_experiment


class TestSEIPFEIndistinguishability(unittest.TestCase):

    def test_statistical_indistinguishability_bounds(self):
        """
        Step 4.4 Verification: Run 500 trials per strategy and verify advantage <= 0.08.
        """
        try:
            run_indistinguishability_experiment(num_trials=500)
        except AssertionError as e:
            self.fail(f"Statistical indistinguishability test failed: {e}")


if __name__ == "__main__":
    unittest.main()
