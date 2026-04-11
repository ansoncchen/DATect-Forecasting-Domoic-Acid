#!/usr/bin/env python3
"""
DATect Stability & Sensitivity Study for Paper

Phase 1 of the systematic tuning validation plan:
  - Phase 1A: Multi-seed noise floor (5 seeds × per-site ON/OFF = 10 runs)
  - Phase 1B: Targeted perturbations (12 runs at seed=42)

Each experiment runs as a subprocess (clean config reload) following the
same isolation pattern as paper_ablation_study.py.

Usage (run on Hyak):
    python3 paper_stability_study.py                    # full run (~3 hrs)
    python3 paper_stability_study.py --quick             # 1% sample (~5 min)
    python3 paper_stability_study.py --phase 1a          # noise floor only
    python3 paper_stability_study.py --phase 1b          # perturbations only

Output: eval_results/stability/stability_results.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join("eval_results", "stability")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "stability_results.json")

SEEDS = [42, 123, 456, 789, 1337]

WA_SITES = {"Kalaloch", "Quinault", "Copalis", "Twin Harbors", "Long Beach"}
OR_SITES = {"Clatsop Beach", "Cannon Beach", "Newport", "Coos Bay", "Gold Beach"}

# Current per-site model assignments (for swap perturbations)
RF_ONLY_SITES = ["Copalis", "Kalaloch", "Twin Harbors", "Quinault", "Coos Bay", "Cannon Beach"]
XGB_ONLY_SITES = ["Long Beach", "Clatsop Beach", "Gold Beach", "Newport"]


# ── Subprocess runner ────────────────────────────────────────────────────────

def _build_subprocess_script(perturbation_code: str = "") -> str:
    """Build the subprocess Python script with optional perturbation code.

    The perturbation_code block runs AFTER config is imported but BEFORE
    the engine is created, allowing monkey-patching of per_site_models
    and config values.
    """
    return f'''
import warnings
warnings.filterwarnings('ignore')

import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import config

# Override seed and sample fraction from env
seed = int(os.environ.get("DATECT_STABILITY_SEED", "42"))
sample_frac = float(os.environ.get("DATECT_STABILITY_FRACTION", "0.20"))
config.RANDOM_SEED = seed
config.TEST_SAMPLE_FRACTION = sample_frac

# Print config for debugging
print(f"Seed={{seed}}, Fraction={{sample_frac}}, PerSite={{config.USE_PER_SITE_MODELS}}", file=sys.stderr)

# ── Perturbation code (injected per experiment) ──
{perturbation_code}
# ── End perturbation code ──

from forecasting.raw_forecast_engine import RawForecastEngine
engine = RawForecastEngine(validate_on_init=False)

results_df = engine.run_retrospective_evaluation(
    task="regression",
    model_type="ensemble",
    min_test_date="2008-01-01",
)

if results_df is None or results_df.empty:
    print(json.dumps({{"error": "no results"}}))
    sys.exit(1)

# Normalise columns
if "actual_da_raw" in results_df.columns and "actual_da" not in results_df.columns:
    results_df = results_df.rename(columns={{"actual_da_raw": "actual_da"}})

y_true = results_df["actual_da"].values
y_pred = results_df["predicted_da"].values

overall = {{
    "r2": float(r2_score(y_true, y_pred)),
    "mae": float(mean_absolute_error(y_true, y_pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    "n": len(y_true),
}}

per_site = {{}}
for site in sorted(results_df["site"].unique()):
    mask = results_df["site"] == site
    yt = results_df.loc[mask, "actual_da"].values
    yp = results_df.loc[mask, "predicted_da"].values
    if len(yt) < 2:
        continue
    per_site[site] = {{
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "n": int(mask.sum()),
    }}

print(json.dumps({{"overall": overall, "per_site": per_site}}))
'''


def run_experiment(name, seed, sample_fraction,
                   env_overrides=None,
                   perturbation_code=""):
    """Run a single experiment in a subprocess for clean config isolation."""
    t0 = time.time()
    print(f"\n{'─'*60}")
    print(f"  {name}  (seed={seed}, frac={sample_fraction})")
    print(f"{'─'*60}")

    env = os.environ.copy()
    env["DATECT_STABILITY_SEED"] = str(seed)
    env["DATECT_STABILITY_FRACTION"] = str(sample_fraction)
    if env_overrides:
        env.update(env_overrides)
        for k, v in env_overrides.items():
            print(f"  ENV: {k}={v}")
    if perturbation_code:
        print(f"  PATCH: {perturbation_code[:80]}...")

    script = _build_subprocess_script(perturbation_code)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=None,  # Show progress live
            text=True,
            timeout=7200,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 2 hours")
        return None

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}, {elapsed:.0f}s)")
        return None

    # Parse JSON from subprocess stdout
    try:
        lines = result.stdout.strip().split("\n")
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = i
                break
        if json_start is None:
            print(f"  ERROR: No JSON in output")
            if result.stdout:
                print(result.stdout[-500:])
            return None

        json_str = "\n".join(lines[json_start:])
        data = json.loads(json_str)
        overall = data["overall"]
        print(f"  R²={overall['r2']:.4f}, MAE={overall['mae']:.2f}, "
              f"N={overall['n']}, time={elapsed:.0f}s")
        return data

    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ERROR parsing: {e}")
        if result.stdout:
            print(result.stdout[-500:])
        return None


# ── Phase 1A: Multi-Seed Noise Floor ─────────────────────────────────────────

def run_phase_1a(sample_fraction: float) -> dict:
    """5 seeds × {per-site ON, per-site OFF} = 10 runs."""
    print("\n" + "=" * 70)
    print("PHASE 1A: Multi-Seed Noise Floor")
    print("=" * 70)

    results = {}
    for seed in SEEDS:
        # Per-site ON (current config)
        data = run_experiment(
            f"per_site_ON seed={seed}", seed, sample_fraction,
        )
        results[f"persite_on_seed{seed}"] = data

        # Per-site OFF (global defaults)
        data = run_experiment(
            f"per_site_OFF seed={seed}", seed, sample_fraction,
            env_overrides={"DATECT_USE_PER_SITE_MODELS": "false"},
        )
        results[f"persite_off_seed{seed}"] = data

    return results


# ── Phase 1B: Perturbation Tests ─────────────────────────────────────────────

# Perturbation code snippets (injected into subprocess after config import)

PATCH_SWAP_RF_TO_XGB = """
from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS
for site, cfg in SITE_SPECIFIC_CONFIGS.items():
    w = cfg.get('ensemble_weights')
    if w and w[1] > w[0]:  # RF-dominant
        cfg['ensemble_weights'] = (1.00, 0.00, 0.00)
"""

PATCH_SWAP_XGB_TO_RF = """
from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS
for site, cfg in SITE_SPECIFIC_CONFIGS.items():
    w = cfg.get('ensemble_weights')
    if w and w[0] > w[1]:  # XGB-dominant
        cfg['ensemble_weights'] = (0.00, 1.00, 0.00)
"""

PATCH_GLOBAL_DEFAULTS = """
# Disable per-site overrides entirely
config.USE_PER_SITE_MODELS = False
"""

# RF param perturbations use DATECT_RF_PARAMS_JSON env var — loky workers
# inherit env and re-import config.py which reads the override at module load.
ENV_RF_SHALLOW = {"DATECT_RF_PARAMS_JSON":
    '{"n_estimators": 200, "max_depth": 6, "min_samples_split": 5, "min_samples_leaf": 3, "max_features": 0.5}'}
ENV_RF_DEEP = {"DATECT_RF_PARAMS_JSON":
    '{"n_estimators": 600, "max_depth": 16, "min_samples_split": 5, "min_samples_leaf": 3, "max_features": 0.9}'}
ENV_RF_MORE_TREES = {"DATECT_RF_PARAMS_JSON":
    '{"n_estimators": 800, "max_depth": 12, "min_samples_split": 5, "min_samples_leaf": 3, "max_features": 0.85}'}
ENV_RF_LESS_REG = {"DATECT_RF_PARAMS_JSON":
    '{"n_estimators": 400, "max_depth": 12, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 0.85}'}

# Feature subset and clipping use env vars read by per_site_models.py at import time.
ENV_ALL_FEATURES = {"DATECT_FEATURE_SUBSET_MODE": "all"}
ENV_MINIMAL_FEATURES = {"DATECT_FEATURE_SUBSET_MODE": "minimal"}
ENV_RELAX_CLIPPING = {"DATECT_CLIP_Q_OVERRIDE": "0.99"}
# no_clipping sets both global (config.py) and per-site (per_site_models.py) to None
ENV_NO_CLIPPING = {"DATECT_CLIP_Q_OVERRIDE": "none"}

# Targeted blend tests: 50/50 XGB+RF for sites whose winner-take-all choice
# is consistently worse than the global 50/50 default (Table C "always −").
PATCH_BLEND_COPALIS_LONGBEACH = """
from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS
for site in ('Copalis', 'Long Beach'):
    if site in SITE_SPECIFIC_CONFIGS:
        SITE_SPECIFIC_CONFIGS[site]['ensemble_weights'] = (0.50, 0.50, 0.00)
"""


PERTURBATIONS = [
    # Model selection (3 runs) — blending applied in parent process, patch works fine
    ("swap_rf_to_xgb", "Swap all RF→XGB", {}, PATCH_SWAP_RF_TO_XGB),
    ("swap_xgb_to_rf", "Swap all XGB→RF", {}, PATCH_SWAP_XGB_TO_RF),
    ("global_defaults", "No per-site config", {"DATECT_USE_PER_SITE_MODELS": "false"}, ""),

    # RF hyperparameters (4 runs) — env vars read by config.py at loky worker import
    ("rf_shallow", "RF shallow (depth=6, trees=200)", ENV_RF_SHALLOW, ""),
    ("rf_deep", "RF deep (depth=16, trees=600)", ENV_RF_DEEP, ""),
    ("rf_more_trees", "RF more trees (n=800)", ENV_RF_MORE_TREES, ""),
    ("rf_less_reg", "RF less regularized (leaf=1)", ENV_RF_LESS_REG, ""),

    # Features + clipping (5 runs) — env vars read by per_site_models.py at loky worker import
    ("all_features", "All features everywhere", ENV_ALL_FEATURES, ""),
    ("minimal_features", "Minimal features (4 only)", ENV_MINIMAL_FEATURES, ""),
    ("relax_clipping", "Relax clipping (0.99 all)", ENV_RELAX_CLIPPING, ""),
    ("no_clipping", "No prediction clipping", ENV_NO_CLIPPING, ""),
    ("monotonic_off", "Monotonic constraints off",
     {"DATECT_USE_MONOTONIC_CONSTRAINTS": "false"}, ""),

    # Targeted blend: Copalis + Long Beach 50/50 (Table C showed winner-take-all
    # is consistently worse than 50/50 for these two sites)
    ("blend_copalis_longbeach", "50/50 blend: Copalis + Long Beach",
     {}, PATCH_BLEND_COPALIS_LONGBEACH),
]


def run_phase_1b(sample_fraction: float, seed: int = 42) -> dict:
    """12 targeted perturbation runs at a single seed."""
    print("\n" + "=" * 70)
    print("PHASE 1B: Targeted Perturbations")
    print("=" * 70)

    # First run baseline at this seed
    baseline = run_experiment(
        "Baseline (current config)", seed, sample_fraction,
    )

    results = {"baseline": baseline}
    for key, name, env_overrides, patch_code in PERTURBATIONS:
        data = run_experiment(
            name, seed, sample_fraction,
            env_overrides=env_overrides if env_overrides else None,
            perturbation_code=patch_code,
        )
        results[key] = data

    return results


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_results(phase_1a: dict, phase_1b: dict) -> dict:
    """Compute noise floor, per-site deltas, perturbation significance."""
    analysis = {}

    # ── 1A: Noise floor ──
    if phase_1a:
        per_site_r2 = {}  # site -> [r2 across seeds] for per-site ON
        no_persite_r2 = {}  # site -> [r2 across seeds] for per-site OFF
        pooled_on = []
        pooled_off = []

        for seed in SEEDS:
            on_data = phase_1a.get(f"persite_on_seed{seed}")
            off_data = phase_1a.get(f"persite_off_seed{seed}")

            if on_data and "overall" in on_data:
                pooled_on.append(on_data["overall"]["r2"])
                for site, m in on_data.get("per_site", {}).items():
                    per_site_r2.setdefault(site, []).append(m["r2"])

            if off_data and "overall" in off_data:
                pooled_off.append(off_data["overall"]["r2"])
                for site, m in off_data.get("per_site", {}).items():
                    no_persite_r2.setdefault(site, []).append(m["r2"])

        # Per-site stability
        site_stability = {}
        for site in sorted(set(per_site_r2) | set(no_persite_r2)):
            on_vals = per_site_r2.get(site, [])
            off_vals = no_persite_r2.get(site, [])
            entry = {}
            if on_vals:
                entry["on_mean"] = round(float(np.mean(on_vals)), 4)
                entry["on_std"] = round(float(np.std(on_vals)), 4)
                entry["on_values"] = [round(v, 4) for v in on_vals]
            if off_vals:
                entry["off_mean"] = round(float(np.mean(off_vals)), 4)
                entry["off_std"] = round(float(np.std(off_vals)), 4)
                entry["off_values"] = [round(v, 4) for v in off_vals]
            if on_vals and off_vals:
                deltas = [a - b for a, b in zip(on_vals, off_vals)]
                entry["delta_mean"] = round(float(np.mean(deltas)), 4)
                entry["delta_std"] = round(float(np.std(deltas)), 4)
                entry["delta_values"] = [round(d, 4) for d in deltas]
                entry["delta_all_positive"] = all(d > 0 for d in deltas)
                entry["delta_all_negative"] = all(d < 0 for d in deltas)
            site_stability[site] = entry

        noise_floor = {
            "pooled_on_mean": round(float(np.mean(pooled_on)), 4) if pooled_on else None,
            "pooled_on_std": round(float(np.std(pooled_on)), 4) if pooled_on else None,
            "pooled_off_mean": round(float(np.mean(pooled_off)), 4) if pooled_off else None,
            "pooled_off_std": round(float(np.std(pooled_off)), 4) if pooled_off else None,
            "per_site": site_stability,
        }
        analysis["noise_floor"] = noise_floor

    # ── 1B: Perturbation significance ──
    if phase_1b:
        baseline = phase_1b.get("baseline")
        if baseline and "overall" in baseline:
            baseline_r2 = baseline["overall"]["r2"]
            noise_std = analysis.get("noise_floor", {}).get("pooled_on_std", 0.05)
            threshold = 2.0 * noise_std if noise_std else 0.02

            perturbation_results = {}
            for key, name, _, _ in PERTURBATIONS:
                data = phase_1b.get(key)
                if data and "overall" in data:
                    delta = data["overall"]["r2"] - baseline_r2
                    significant = abs(delta) > threshold
                    entry = {
                        "name": name,
                        "r2": round(data["overall"]["r2"], 4),
                        "mae": round(data["overall"]["mae"], 2),
                        "delta_r2": round(delta, 4),
                        "significant": significant,
                        "threshold": round(threshold, 4),
                    }
                    # Per-site deltas
                    site_deltas = {}
                    bl_sites = baseline.get("per_site", {})
                    for site, m in data.get("per_site", {}).items():
                        if site in bl_sites:
                            sd = m["r2"] - bl_sites[site]["r2"]
                            site_deltas[site] = round(sd, 4)
                    entry["site_deltas"] = site_deltas
                    perturbation_results[key] = entry
                else:
                    perturbation_results[key] = {"name": name, "status": "failed"}

            analysis["perturbations"] = {
                "baseline_r2": round(baseline_r2, 4),
                "significance_threshold": round(threshold, 4),
                "results": perturbation_results,
            }

    # ── Go/no-go recommendation ──
    recommendations = []
    if "perturbations" in analysis:
        pr = analysis["perturbations"]["results"]

        # Check RF sensitivity
        rf_keys = ["rf_shallow", "rf_deep", "rf_more_trees", "rf_less_reg"]
        rf_significant = [k for k in rf_keys if pr.get(k, {}).get("significant")]
        if rf_significant:
            recommendations.append(
                f"RF params sensitive ({', '.join(rf_significant)}) → Phase 2A (RF tuning)"
            )

        # Check model selection
        model_keys = ["swap_rf_to_xgb", "swap_xgb_to_rf"]
        model_significant = [k for k in model_keys if pr.get(k, {}).get("significant")]
        if model_significant:
            recommendations.append(
                f"Model selection sensitive ({', '.join(model_significant)}) → Phase 2B"
            )

        # Check feature/clipping
        fc_keys = ["all_features", "minimal_features", "relax_clipping", "no_clipping",
                    "monotonic_off"]
        fc_significant = [k for k in fc_keys if pr.get(k, {}).get("significant")]
        if fc_significant:
            recommendations.append(
                f"Feature/clipping sensitive ({', '.join(fc_significant)}) → Phase 2B"
            )

        if not recommendations:
            recommendations.append(
                "No significant perturbations found → skip Phase 2, proceed to Phase 3 (stability table)"
            )

    # Check model selection stability across seeds
    if "noise_floor" in analysis:
        unstable_sites = []
        for site, entry in analysis["noise_floor"]["per_site"].items():
            if "delta_values" in entry:
                # If the per-site tuning delta flips sign across seeds
                if not entry["delta_all_positive"] and not entry["delta_all_negative"]:
                    unstable_sites.append(site)
        if unstable_sites:
            recommendations.append(
                f"Per-site tuning unstable across seeds at: {', '.join(unstable_sites)}"
            )

    analysis["recommendations"] = recommendations
    return analysis


def print_summary(analysis: dict):
    """Print human-readable summary of results."""

    # ── Noise floor table ──
    nf = analysis.get("noise_floor")
    if nf:
        print(f"\n{'='*90}")
        print("TABLE A: Multi-Seed Stability (Phase 1A)")
        print(f"{'='*90}")
        print(f"  {'Site':<18} {'ON mean':>8} {'ON std':>8} {'OFF mean':>8} "
              f"{'Δ mean':>8} {'Δ std':>8} {'Stable?':>8}")
        print("  " + "─" * 80)

        for site, entry in sorted(nf["per_site"].items()):
            on_m = f"{entry['on_mean']:.4f}" if "on_mean" in entry else "n/a"
            on_s = f"{entry['on_std']:.4f}" if "on_std" in entry else "n/a"
            off_m = f"{entry['off_mean']:.4f}" if "off_mean" in entry else "n/a"
            d_m = f"{entry['delta_mean']:+.4f}" if "delta_mean" in entry else "n/a"
            d_s = f"{entry['delta_std']:.4f}" if "delta_std" in entry else "n/a"
            stable = "Yes" if entry.get("delta_all_positive") or entry.get("delta_all_negative") else "No"
            print(f"  {site:<18} {on_m:>8} {on_s:>8} {off_m:>8} {d_m:>8} {d_s:>8} {stable:>8}")

        print(f"\n  Pooled ON:  R² = {nf['pooled_on_mean']:.4f} ± {nf['pooled_on_std']:.4f}")
        print(f"  Pooled OFF: R² = {nf['pooled_off_mean']:.4f} ± {nf['pooled_off_std']:.4f}")

    # ── Perturbation table ──
    pt = analysis.get("perturbations")
    if pt:
        print(f"\n{'='*90}")
        print("TABLE B: Perturbation Sensitivity (Phase 1B)")
        print(f"{'='*90}")
        print(f"  {'Perturbation':<35} {'R²':>8} {'ΔR²':>8} {'MAE':>8} {'>2σ?':>6}")
        print("  " + "─" * 70)
        print(f"  {'Baseline':35} {pt['baseline_r2']:>8.4f} {'---':>8} {'':>8}")

        for key, entry in pt["results"].items():
            if "r2" not in entry:
                print(f"  {entry.get('name', key):<35} {'FAILED':>8}")
                continue
            sig = "YES" if entry["significant"] else "no"
            print(f"  {entry['name']:<35} {entry['r2']:>8.4f} "
                  f"{entry['delta_r2']:>+8.4f} {entry['mae']:>8.2f} {sig:>6}")

        print(f"\n  Significance threshold: |ΔR²| > {pt['significance_threshold']:.4f} (2× noise floor std)")

    # ── Recommendations ──
    recs = analysis.get("recommendations", [])
    if recs:
        print(f"\n{'='*90}")
        print("GO/NO-GO RECOMMENDATIONS")
        print(f"{'='*90}")
        for rec in recs:
            print(f"  → {rec}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DATect stability & sensitivity study (Phase 1)"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Use 1%% sample for fast smoke test (~5 min)")
    parser.add_argument("--sample-fraction", type=float, default=0.20,
                        help="Sample fraction for evaluation (default: 0.20)")
    parser.add_argument("--phase", choices=["1a", "1b", "all"], default="all",
                        help="Which phase to run (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for Phase 1B perturbations (default: 42)")
    args = parser.parse_args()

    sample_fraction = 0.01 if args.quick else args.sample_fraction

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    mode = "quick smoke test" if args.quick else f"full ({sample_fraction*100:.0f}% sample)"
    print(f"DATect Stability Study — {mode}")
    print(f"Phase: {args.phase}, Output: {OUTPUT_FILE}")
    print(f"Seeds: {SEEDS}")

    # Load existing results if running only one phase
    existing = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    phase_1a = existing.get("phase_1a", {})
    phase_1b = existing.get("phase_1b", {})

    if args.phase in ("1a", "all"):
        phase_1a = run_phase_1a(sample_fraction)

    if args.phase in ("1b", "all"):
        phase_1b = run_phase_1b(sample_fraction, seed=args.seed)

    # Analyze
    analysis = analyze_results(phase_1a, phase_1b)

    # Save everything
    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_fraction": sample_fraction,
            "seeds": SEEDS,
            "perturbation_seed": args.seed,
            "phase": args.phase,
            "elapsed_s": round(time.time() - t0, 1),
            "quick": args.quick,
        },
        "phase_1a": phase_1a,
        "phase_1b": phase_1b,
        "analysis": analysis,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Total elapsed: {time.time() - t0:.0f}s")

    # Print summary tables
    print_summary(analysis)


if __name__ == "__main__":
    main()
