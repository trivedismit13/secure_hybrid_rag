"""
SHIELD-RAG Interactive Live Demonstration Walkthrough (Plan B).
Root-level entrypoint for the 10-step demonstration.
"""
import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = str(Path(__file__).resolve().parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from demo.run_demo import main, run_demo

__all__ = ["main", "run_demo"]

if __name__ == "__main__":
    main()
