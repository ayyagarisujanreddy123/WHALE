"""Compatibility shim for the canonical chess-puzzle data preparation script.

New code should use ``scripts/prepare_chess_puzzle_data.py``.  This module stays
importable for existing commands that used the former implementation path.
"""

from __future__ import annotations

import sys
from pathlib import Path


DOMAIN_DIR = Path(__file__).resolve().parents[1]
if str(DOMAIN_DIR) not in sys.path:
    sys.path.insert(0, str(DOMAIN_DIR))

from scripts.prepare_chess_puzzle_data import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    main()
