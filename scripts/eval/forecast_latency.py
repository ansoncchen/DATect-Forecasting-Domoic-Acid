"""
Latency of POST /api/forecast/enhanced, measured against a running API process.

Starts the API from ``--repo`` (any checkout, e.g. a worktree of main for the
"before" number) with uvicorn on a free local port, waits for /health, sends one
untimed warm-up request (the first request builds the feature frame), then times
``--n`` requests one after another. Requests are (site, date) pairs drawn with a
fixed seed from the weekly rows of final_output.parquet, so the same list is sent
before and after a change. Each response body is hashed without the serving fields
(source, model_version); ``--compare-to`` counts responses identical to an earlier run.

Usage:
    python scripts/eval/forecast_latency.py --repo /path/to/main-worktree \
        --out results/serving/latency_before.json
    DATECT_FORECAST_TABLE=$LAKE/gold/forecasts \
        python scripts/eval/forecast_latency.py --repo . --out results/serving/latency_after.json \
        --compare-to results/serving/latency_before.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import requests


def sample_requests(repo: str, n: int, seed: int) -> list[tuple[str, str]]:
    data = pd.read_parquet(os.path.join(repo, "data", "processed", "final_output.parquet"),
                           columns=["date", "site"])
    data["date"] = pd.to_datetime(data["date"])
    pairs = data.sort_values(["site", "date"]).reset_index(drop=True)
    idx = np.random.default_rng(seed).choice(len(pairs), size=n, replace=False)
    return [(pairs.at[i, "site"], pairs.at[i, "date"].strftime("%Y-%m-%d")) for i in idx]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_healthy(url: str, proc: subprocess.Popen, timeout: float) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"API exited with code {proc.returncode}")
        try:
            if requests.get(url + "/health", timeout=2).ok:
                return time.perf_counter() - t0
        except requests.RequestException:
            pass
        time.sleep(0.2)
    raise TimeoutError("API did not become healthy")


def _post(url: str, site: str, date: str, model: str) -> tuple[float, dict]:
    t0 = time.perf_counter()
    r = requests.post(url + "/api/forecast/enhanced",
                      json={"date": date, "site": site, "task": "regression", "model": model},
                      timeout=600)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return elapsed, r.json()


def body_sha256(body: dict) -> str:
    """Hash of a response without the fields that say where it came from."""
    rest = {k: v for k, v in body.items() if k not in ("source", "model_version")}
    return hashlib.sha256(json.dumps(rest, sort_keys=True).encode()).hexdigest()


def _git_sha(repo: str) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--repo", default=".", help="checkout the API is started from")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="ensemble")
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare-to", help="earlier output JSON with the same requests")
    args = parser.parse_args()
    repo = os.path.abspath(args.repo)

    pairs = sample_requests(repo, args.n + 1, args.seed)
    warmup, pairs = pairs[0], pairs[1:]

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ, PYTHONPATH=repo)
    t_launch = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        cwd=repo, env=env)
    try:
        startup = _wait_healthy(url, proc, timeout=300)
        first, _ = _post(url, *warmup, args.model)
        rows = []
        for site, date in pairs:
            elapsed, body = _post(url, site, date, args.model)
            rows.append({"site": site, "date": date, "seconds": elapsed,
                         "success": bool(body.get("success")),
                         "has_forecast": body.get("regression") is not None,
                         "source": body.get("source", "computed"),
                         "body_sha256": body_sha256(body)})
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    total = time.perf_counter() - t_launch

    secs = np.array([r["seconds"] for r in rows])
    result = {
        "repo_sha": _git_sha(repo),
        "host": socket.gethostname(),
        "cpus": os.cpu_count(),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "python": platform.python_version(),
        "forecast_table": os.environ.get("DATECT_FORECAST_TABLE"),
        "model": args.model,
        "seed": args.seed,
        "n": len(rows),
        "startup_seconds": startup,
        "warmup_request": {"site": warmup[0], "date": warmup[1], "seconds": first},
        "p50_seconds": float(np.percentile(secs, 50)),
        "p95_seconds": float(np.percentile(secs, 95)),
        "mean_seconds": float(secs.mean()),
        "min_seconds": float(secs.min()),
        "max_seconds": float(secs.max()),
        "percentile_method": "numpy linear interpolation",
        "sources": pd.Series([r["source"] for r in rows]).value_counts().to_dict(),
        "total_seconds": total,
        "requests": rows,
    }
    if args.compare_to:
        with open(args.compare_to) as f:
            earlier = {(r["site"], r["date"]): r for r in json.load(f)["requests"]}
        same = [earlier.get((r["site"], r["date"]), {}).get("body_sha256") == r["body_sha256"] for r in rows]
        before = np.array([earlier[(r["site"], r["date"])]["seconds"] for r in rows
                           if (r["site"], r["date"]) in earlier])
        result["compared_to"] = {"file": args.compare_to, "identical_responses": int(sum(same)),
                                 "n": len(rows), "p50_before_seconds": float(np.percentile(before, 50)),
                                 "p95_before_seconds": float(np.percentile(before, 95))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    if args.compare_to:
        print(f"responses identical to {args.compare_to}: "
              f"{result['compared_to']['identical_responses']}/{len(rows)}")
    print(f"p50 {result['p50_seconds']:.3f}s  p95 {result['p95_seconds']:.3f}s  "
          f"(n={len(rows)}, sources {result['sources']}, startup {startup:.1f}s, "
          f"first request {first:.1f}s)")


if __name__ == "__main__":
    main()
