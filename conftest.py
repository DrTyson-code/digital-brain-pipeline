"""Pytest configuration — adds the repo root to sys.path so that
``import src.llm`` etc. resolve correctly without requiring ``PYTHONPATH=.``.
"""

import sys
from pathlib import Path

# Ensure the repo root is first on sys.path
sys.path.insert(0, str(Path(__file__).parent))
