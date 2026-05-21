#!/usr/bin/env python3
"""
Run a feature-extension chain: augment parquet, then A/B retrospective.

Each chain (chains/cX_*.py) defines:
  CHAIN_NAME, NEW_FEATURES, TARGET_SITES, add_features(df)

This driver:
  1. Loads data/processed/final_output.parquet
  2. Calls the chain's add_features() to produce an augmented dataframe
  3. Saves it to chains/output/final_output_<chain>.parquet
  4. Runs TWO retrospective evals via the existing engine subprocess pattern:
       (a) baseline: drop NEW_FEATURES via DATECT_EXTRA_DROP_FEATURES
       (b) +chain:   keep them (TARGET_SITES' feature_subset gets the names appended)
  5. Saves both result JSONs + a comparison summary

Outputs:
  chains/output/<chain>_baseline.json
  chains/output/<chain>_with.json
  chains/output/<chain>_summary.txt

Usage:
  python chains/run_chain.py --chain lagged_pn
  python chains/run_chain.py --chain beuti_derivatives
  python chains/run_chain.py --chain nemo_mooring
  python chains/run_chain.py --chain esp_offshore_pda
  python chains/run_chain.py --chain ALL    # runs all 4 sequentially
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

# Ensure repo root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval.paper_ablation_study import SUBPROCESS_SCRIPT
from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS

CHAINS = ["lagged_pn", "beuti_derivatives", "nemo_mooring", "esp_offshore_pda"]


def run_eval(env_overrides: dict, timeout: int = 5400) -> dict:
    """Run one retrospective subprocess and parse {overall, per_site}."""
    env = os.environ.copy()
    env.update(env_overrides)
    env.setdefault("DATECT_ABLATION_SEED", "123")
    env["DATECT_MIN_TRAINING_FOR_TUNING"] = "99999"  # skip inner tuning (same trick as Optuna)

    result = subprocess.run(
        [sys.executable, "-c", SUBPROCESS_SCRIPT],
        env=env, stdout=subprocess.PIPE, stderr=None,
        text=True, timeout=timeout,
    )
    if result.returncode != 0:
        return {"error": f"exit {result.returncode}"}
    for line in result.stdout.strip().split("\n"):
        if line.strip().startswith("{"):
            return json.loads("\n".join(result.stdout.strip().split("\n")[result.stdout.strip().split("\n").index(line):]))
    return {"error": "no_json"}


def patch_per_site_models_for_chain(chain_module, target_sites: list[str] | None):
    """Temporarily extend feature_subset for target sites with the chain's new features.
    Returns a list of (site, original_subset) for restore.
    """
    new_features = chain_module.NEW_FEATURES
    sites = target_sites if target_sites is not None else list(SITE_SPECIFIC_CONFIGS.keys())
    original = []
    for site in sites:
        cfg = SITE_SPECIFIC_CONFIGS.get(site)
        if cfg is None:
            continue
        fs = cfg.get("feature_subset")
        original.append((site, fs))
        if fs is not None:
            cfg["feature_subset"] = list(fs) + new_features
        # if fs is None ("use all features") the new columns are already included
    return original


def restore_per_site_models(original):
    for site, fs in original:
        SITE_SPECIFIC_CONFIGS[site]["feature_subset"] = fs


def run_one_chain(name: str, out_dir: Path):
    print(f"\n{'='*70}\nCHAIN: {name}\n{'='*70}", flush=True)
    chain = importlib.import_module(f"chains.c1_lagged_pn" if name == "lagged_pn"
                                    else f"chains.c2_beuti_derivatives" if name == "beuti_derivatives"
                                    else f"chains.c3_nemo_mooring" if name == "nemo_mooring"
                                    else f"chains.c4_esp_offshore_pda" if name == "esp_offshore_pda"
                                    else None)
    if chain is None:
        print(f"  unknown chain: {name}")
        return

    base_parquet = Path("data/processed/final_output.parquet")
    backup = Path(f"chains/output/final_output_{name}.backup_orig.parquet")
    aug_parquet = Path(f"chains/output/final_output_{name}.parquet")
    aug_parquet.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {base_parquet}", flush=True)
    df = pd.read_parquet(base_parquet)
    print(f"  Original shape: {df.shape}", flush=True)

    print(f"  Calling {name}.add_features()...", flush=True)
    t0 = time.time()
    df_aug = chain.add_features(df)
    print(f"  Augmented shape: {df_aug.shape}  ({time.time()-t0:.1f}s)", flush=True)
    new_cols = [c for c in df_aug.columns if c not in df.columns]
    print(f"  New columns added: {new_cols}", flush=True)

    # Backup original, swap in augmented
    if not backup.exists():
        shutil.copy(base_parquet, backup)
    df_aug.to_parquet(base_parquet, index=False)
    df_aug.to_parquet(aug_parquet, index=False)

    try:
        # Patch per_site_models in-memory
        original = patch_per_site_models_for_chain(chain, chain.TARGET_SITES)

        # Run baseline: drop new features
        drop_csv = ",".join(chain.NEW_FEATURES)
        print(f"\n  --- BASELINE (drop {len(chain.NEW_FEATURES)} new features) ---", flush=True)
        baseline = run_eval({"DATECT_EXTRA_DROP_FEATURES": drop_csv})
        (out_dir / f"{name}_baseline.json").write_text(json.dumps(baseline, indent=2))
        if "overall" in baseline:
            print(f"  baseline R²={baseline['overall']['r2']:.4f} MAE={baseline['overall']['mae']:.2f}", flush=True)

        # Run +chain: keep new features
        print(f"\n  --- WITH +{name} features ---", flush=True)
        with_chain = run_eval({})  # nothing dropped
        (out_dir / f"{name}_with.json").write_text(json.dumps(with_chain, indent=2))
        if "overall" in with_chain:
            print(f"  +chain  R²={with_chain['overall']['r2']:.4f} MAE={with_chain['overall']['mae']:.2f}", flush=True)

        # Summary
        if "overall" in baseline and "overall" in with_chain:
            d_r2 = with_chain["overall"]["r2"] - baseline["overall"]["r2"]
            d_mae = with_chain["overall"]["mae"] - baseline["overall"]["mae"]
            summary = [
                f"Chain: {name}",
                f"New features: {len(chain.NEW_FEATURES)} ({chain.NEW_FEATURES})",
                f"Target sites: {chain.TARGET_SITES}",
                "",
                f"Pooled (N={baseline['overall']['n']}):",
                f"  baseline R² = {baseline['overall']['r2']:+.4f}, MAE = {baseline['overall']['mae']:.3f}",
                f"  +chain   R² = {with_chain['overall']['r2']:+.4f}, MAE = {with_chain['overall']['mae']:.3f}",
                f"  Δ R²        = {d_r2:+.4f}",
                f"  Δ MAE       = {d_mae:+.3f}",
                "",
                "Per-site (sorted by Δ R² desc):",
            ]
            rows = []
            for site in baseline["per_site"]:
                b = baseline["per_site"][site]
                w = with_chain["per_site"].get(site, {})
                if not w: continue
                rows.append((site, b["r2"], w.get("r2", float("nan")), w.get("r2", 0) - b["r2"], b["n"]))
            for site, b_r2, w_r2, dr, n in sorted(rows, key=lambda x: -x[3]):
                in_target = "*" if (chain.TARGET_SITES is None or site in chain.TARGET_SITES) else " "
                summary.append(f"  {in_target} {site:<18} baseline={b_r2:+.4f}  +chain={w_r2:+.4f}  Δ={dr:+.4f}  N={n}")
            (out_dir / f"{name}_summary.txt").write_text("\n".join(summary))
            print("\n" + "\n".join(summary))
    finally:
        # Restore original parquet and per_site_models
        shutil.copy(backup, base_parquet)
        restore_per_site_models(original)
        print(f"\n  Restored original {base_parquet}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chain", required=True, choices=CHAINS + ["ALL"])
    p.add_argument("--out-dir", default="chains/output")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = CHAINS if args.chain == "ALL" else [args.chain]
    for c in targets:
        try:
            run_one_chain(c, out_dir)
        except Exception as e:
            print(f"\n  CHAIN {c} FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
