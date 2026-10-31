#!/usr/bin/env python3
"""
Sensitivity studies for the paper appendix (seed 42, 20% development sample).

  --study features       drop one feature group at a time (feature-group ablation)
  --study perturbations  change one design choice at a time (perturbation table)

Each experiment runs in a fresh subprocess through paper_ablation_study.run_ablation,
so every change travels as a DATECT_* environment variable that joblib workers
also read. The results file is rewritten after each experiment and finished
experiments are skipped on restart, so a preempted ckpt job resumes where it
stopped.

Usage (from repository root; run on Hyak):
    python3 scripts/eval/paper_sensitivity_study.py --study features
    python3 scripts/eval/paper_sensitivity_study.py --study perturbations

Output: results/ablation/paper_{features,perturbations}_results.json
"""

import argparse
import json
import os
import tempfile

from paper_ablation_study import run_ablation

# ── Feature groups (features the models still use; the other MODIS columns are
# already dropped as zero-importance in config/tuned_hyperparameters.json) ──
SATELLITE = ["modis-sst", "sst-anom", "modis-flr"]
CLIMATE = ["pdo", "oni"]
PSEUDO_NITZSCHIA = ["pn", "pn_log"]
ROLLING = [
    "raw_obs_roll_mean_4", "raw_obs_roll_std_4", "raw_obs_roll_max_4",
    "raw_obs_roll_std_8", "raw_obs_roll_max_8",
    "raw_obs_roll_std_12", "raw_obs_roll_max_12",
]
CALENDAR = ["sin_day_of_year", "cos_day_of_year", "month"]

FEATURE_GROUPS = [
    ("satellite", "Satellite (SST, SST anomaly, FLH)", SATELLITE),
    ("climate_indices", "Climate indices (PDO, ONI)", CLIMATE),
    ("upwelling", "Upwelling (BEUTI)", ["beuti"]),
    ("discharge", "River discharge", ["discharge"]),
    ("pseudo_nitzschia", "Pseudo-nitzschia (count, log count)", PSEUDO_NITZSCHIA),
    ("rolling_stats", "Rolling DA statistics (7)", ROLLING),
    ("calendar", "Calendar (month, day-of-year sin/cos)", CALENDAR),
    ("all_environmental", "All environmental (satellite, climate, BEUTI, discharge, PN)",
     SATELLITE + CLIMATE + ["beuti", "discharge"] + PSEUDO_NITZSCHIA),
]

# ── Design perturbations ──
# RF overlays apply to the global RF parameters; the four sites with their own
# rf_params (Coos Bay, Cannon Beach, Gold Beach, Newport) keep those keys.
RF_VARIANTS = [
    ("rf_shallow", "RF shallow (depth 6, 200 trees)",
     {"n_estimators": 200, "max_depth": 6, "min_samples_split": 5, "min_samples_leaf": 3, "max_features": 0.5}),
    ("rf_deep", "RF deep (depth 16, 600 trees)",
     {"n_estimators": 600, "max_depth": 16, "min_samples_split": 5, "min_samples_leaf": 3, "max_features": 0.9}),
    ("rf_more_trees", "RF more trees (800)",
     {"n_estimators": 800, "max_depth": 12, "min_samples_split": 5, "min_samples_leaf": 3, "max_features": 0.85}),
    ("rf_less_reg", "RF less regularized (leaf 1)",
     {"n_estimators": 400, "max_depth": 12, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 0.85}),
]


def _weights_override(tmpdir: str, name: str, weights: tuple) -> str:
    """Write a DATECT_HPARAM_OVERRIDE_JSON file that sets every site's ensemble weights."""
    from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS

    path = os.path.join(tmpdir, f"{name}.json")
    with open(path, "w") as f:
        json.dump({site: {"ensemble_weights": list(weights)} for site in SITE_SPECIFIC_CONFIGS}, f)
    return path


def experiments(study: str, tmpdir: str) -> list:
    """(key, label, env_overrides) for every run in the study, baseline first."""
    runs = [("baseline", "Baseline (full DATect)", {})]
    if study == "features":
        runs += [
            (key, label, {"DATECT_EXTRA_DROP_FEATURES": ",".join(cols)})
            for key, label, cols in FEATURE_GROUPS
        ]
        return runs
    runs += [
        ("xgb_everywhere", "XGBoost at every site",
         {"DATECT_HPARAM_OVERRIDE_JSON": _weights_override(tmpdir, "xgb", (1.0, 0.0, 0.0))}),
        ("rf_everywhere", "Random Forest at every site",
         {"DATECT_HPARAM_OVERRIDE_JSON": _weights_override(tmpdir, "rf", (0.0, 1.0, 0.0))}),
        ("no_per_site", "No per-site config", {"DATECT_USE_PER_SITE_MODELS": "false"}),
    ]
    runs += [(key, label, {"DATECT_RF_PARAMS_JSON": json.dumps(params)}) for key, label, params in RF_VARIANTS]
    runs += [
        ("all_features", "All features at every site", {"DATECT_FEATURE_SUBSET_MODE": "all"}),
        ("minimal_features", "Minimal features (4)", {"DATECT_FEATURE_SUBSET_MODE": "minimal"}),
        ("clip_q95", "Clip at the 0.95 training quantile everywhere", {"DATECT_CLIP_Q_OVERRIDE": "0.95"}),
        ("clip_q99", "Clip at the 0.99 training quantile everywhere", {"DATECT_CLIP_Q_OVERRIDE": "0.99"}),
        ("monotonic_off", "No monotonic constraints", {"DATECT_USE_MONOTONIC_CONSTRAINTS": "false"}),
    ]
    return runs


def main():
    from _repo import ensure_repo_root

    ensure_repo_root()
    parser = argparse.ArgumentParser(description="DATect sensitivity studies for the paper appendix")
    parser.add_argument("--study", choices=["features", "perturbations"], required=True)
    parser.add_argument("--seed", type=int, default=42, help="Test-point sampling seed (default: 42 = dev set)")
    args = parser.parse_args()

    out_path = os.path.join("results", "ablation", f"paper_{args.study}_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    saved = {"study": args.study, "seed": args.seed, "runs": {}}
    if os.path.exists(out_path):
        with open(out_path) as f:
            previous = json.load(f)
        if previous.get("seed") == args.seed:
            saved = previous

    with tempfile.TemporaryDirectory() as tmpdir:
        for key, label, env in experiments(args.study, tmpdir):
            if saved["runs"].get(key, {}).get("overall"):
                print(f"skip {key}: already in {out_path}")
                continue
            data = run_ablation(label, args.seed, env_overrides=env)
            # The override file lives in tmpdir; record what it set, not its path
            env_record = {k: v for k, v in env.items() if k != "DATECT_HPARAM_OVERRIDE_JSON"}
            if "DATECT_HPARAM_OVERRIDE_JSON" in env:
                env_record["ensemble_weights"] = key.split("_")[0]
            saved["runs"][key] = {"label": label, "env": env_record, **(data or {})}
            with open(out_path, "w") as f:
                json.dump(saved, f, indent=2)

    base = saved["runs"].get("baseline", {}).get("overall")
    print(f"\n{'=' * 72}\n{args.study.upper()} (seed {args.seed})\n{'=' * 72}")
    print(f"{'Run':<58} {'R2':>6} {'dR2':>7} {'MAE':>6}")
    for key, run in saved["runs"].items():
        ov = run.get("overall")
        if not ov:
            print(f"{run['label']:<58} FAILED")
            continue
        delta = "---" if key == "baseline" or not base else f"{ov['r2'] - base['r2']:+.3f}"
        print(f"{run['label']:<58} {ov['r2']:>6.3f} {delta:>7} {ov['mae']:>6.2f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
