"""Ensure repo root is on sys.path and cwd for scripts/eval/*.py."""
from __future__ import annotations

import os
import sys


def ensure_repo_root() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    return root
