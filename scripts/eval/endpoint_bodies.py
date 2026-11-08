"""
Response bodies of the API's precomputed endpoints, hashed for a before/after gate.

Starts the API from ``--repo`` (as forecast_latency.py does), requests the spectral
analysis of each site and of all sites, and each site's correlation heatmap, and
records the sha256 of each body without the fields that only say where it was served from. Two runs, e.g. the old
file cache and the Delta store, pass the gate when every hash matches
(``--compare-to``). Bodies are also written gzipped for inspecting a mismatch.

Usage:
    ENABLE_PRECOMPUTED_CACHE=true python scripts/eval/endpoint_bodies.py \
        --repo /path/to/old-checkout --out results/store/bodies_before.json
    DATECT_PRECOMPUTED_DIR=$LAKE/gold python scripts/eval/endpoint_bodies.py --repo . \
        --out results/store/bodies_after.json --compare-to results/store/bodies_before.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecast_latency import HTTP, _free_port, _git_sha, _wait_healthy  # noqa: E402

# Where a body says it was served from: excluded from the hash
SERVING_FIELDS = {"cache", "cached", "source", "precomputed"}


def body_sha256(body: dict) -> str:
    rest = {k: v for k, v in body.items() if k not in SERVING_FIELDS}
    return hashlib.sha256(json.dumps(rest, sort_keys=True).encode()).hexdigest()


def requests_for(repo: str) -> list[tuple[str, str, str, dict | None]]:
    """(name, method, path, json body) for every precomputed endpoint."""
    sites = sorted(pd.read_parquet(os.path.join(repo, "data", "processed", "final_output.parquet"),
                                   columns=["site"])["site"].unique())
    out = [("spectral/all", "GET", "/api/visualizations/spectral/all", None)]
    out += [(f"spectral/{s}", "GET", f"/api/visualizations/spectral/{s}", None) for s in sites]
    out += [(f"correlation/{s}", "GET", f"/api/visualizations/correlation/{s}", None) for s in sites]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare-to", help="earlier output JSON")
    args = parser.parse_args()
    repo = os.path.abspath(args.repo)

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        cwd=repo, env=dict(os.environ, PYTHONPATH=repo))
    rows, bodies = [], {}
    try:
        _wait_healthy(url, proc, timeout=300)
        for name, method, path, payload in requests_for(repo):
            t0 = time.perf_counter()
            r = HTTP.request(method, url + path, json=payload, timeout=3600)
            seconds = time.perf_counter() - t0
            body = r.json()
            bodies[name] = body
            rows.append({"name": name, "status": r.status_code, "seconds": seconds,
                         "success": body.get("success"),
                         "served_from": {k: body[k] for k in SERVING_FIELDS if k in body},
                         "body_sha256": body_sha256(body)})
            print(f"{name:40s} {r.status_code} {seconds:7.2f}s {rows[-1]['body_sha256'][:12]}", flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=30)

    result = {"repo_sha": _git_sha(repo), "requests": rows}
    if args.compare_to:
        with open(args.compare_to) as f:
            earlier = {r["name"]: r["body_sha256"] for r in json.load(f)["requests"]}
        same = [r["name"] for r in rows if earlier.get(r["name"]) == r["body_sha256"]]
        result["compared_to"] = {"file": args.compare_to, "identical": len(same), "n": len(rows),
                                 "different": [r["name"] for r in rows if r["name"] not in same]}
        print(f"identical to {args.compare_to}: {len(same)}/{len(rows)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    with gzip.open(args.out.replace(".json", ".bodies.json.gz"), "wt") as f:
        json.dump(bodies, f, default=str)


if __name__ == "__main__":
    main()
