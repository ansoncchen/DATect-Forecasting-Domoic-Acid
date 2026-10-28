"""Run engine code in a fresh Python process.

config.py and per_site_models.py read DATECT_* environment variables at import,
so experiments that vary those variables must run in a new interpreter. The
child script reports results by printing a single JSON line on stdout.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def run_python(code: str, env_overrides: dict[str, str], timeout: int,
               show_stderr: bool = True) -> subprocess.CompletedProcess:
    """Run ``code`` with ``python -c`` and the current env plus ``env_overrides``."""
    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=None if show_stderr else subprocess.DEVNULL,
        text=True,
        timeout=timeout,
    )


def last_json_line(stdout: str) -> dict | None:
    """Parse the last stdout line that holds a JSON object, or None if there is none."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None
