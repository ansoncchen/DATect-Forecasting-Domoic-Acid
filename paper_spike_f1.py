#!/usr/bin/env python3
"""
Compute spike detection metrics (DA > 20 µg/g) per site and overall.

Loads ensemble regression predictions and evaluates binary classification
of "spike" events (DA exceeding the FDA regulatory limit of 20 µg/g).

Usage:
    python3 paper_spike_f1.py

Output:
    paper_spike_f1_latex.txt    (LaTeX table for paper)
    paper_spike_f1_results.json (full metrics)

By default uses the seed=42 development set (cache/retrospective/).
Set CACHE_DIR to cache_seed123/retrospective/ once that data is available.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ── Configuration ────────────────────────────────────────────────────────────
CACHE_DIR = Path("./cache/retrospective")  # Switch to cache_seed123/retrospective/ when available
SPIKE_THRESHOLD = 20.0  # FDA regulatory limit (µg/g)
OUTPUT_JSON = "paper_spike_f1_results.json"
OUTPUT_LATEX = "paper_spike_f1_latex.txt"

SITE_ORDER = [
    "Copalis", "Twin Harbors", "Quinault", "Long Beach", "Kalaloch",
    "Clatsop Beach", "Cannon Beach", "Gold Beach", "Coos Bay", "Newport",
]


def load_predictions(cache_dir: Path) -> list[dict]:
    """Load ensemble regression predictions from cache."""
    json_path = cache_dir / "regression_ensemble.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Cache file not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)


def compute_spike_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute binary spike detection metrics."""
    actual_spike = (y_true > SPIKE_THRESHOLD).astype(int)
    pred_spike = (y_pred > SPIKE_THRESHOLD).astype(int)

    n_total = len(y_true)
    n_spikes = actual_spike.sum()

    if n_spikes == 0:
        return {
            "n": n_total, "n_spikes": 0,
            "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
            "tp": 0, "fp": 0, "fn": 0, "tn": n_total,
        }

    prec = precision_score(actual_spike, pred_spike, zero_division=0)
    rec = recall_score(actual_spike, pred_spike, zero_division=0)
    f1 = f1_score(actual_spike, pred_spike, zero_division=0)

    cm = confusion_matrix(actual_spike, pred_spike, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "n": n_total, "n_spikes": int(n_spikes),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap 95% CI for spike F1."""
    rng = np.random.RandomState(seed)
    actual_spike = (y_true > SPIKE_THRESHOLD).astype(int)
    pred_spike = (y_pred > SPIKE_THRESHOLD).astype(int)

    if actual_spike.sum() == 0:
        return {"f1_ci_lo": float("nan"), "f1_ci_hi": float("nan")}

    f1s = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if actual_spike[idx].sum() == 0:
            continue
        f1s.append(f1_score(actual_spike[idx], pred_spike[idx], zero_division=0))

    if len(f1s) < 100:
        return {"f1_ci_lo": float("nan"), "f1_ci_hi": float("nan")}

    return {
        "f1_ci_lo": float(np.percentile(f1s, 2.5)),
        "f1_ci_hi": float(np.percentile(f1s, 97.5)),
    }


def main():
    print("=" * 70)
    print("Spike Detection F1 Analysis (DA > 20 µg/g)")
    print(f"Cache: {CACHE_DIR}")
    print("=" * 70)

    data = load_predictions(CACHE_DIR)
    print(f"Loaded {len(data)} predictions")

    # Organize by site
    site_data = {}
    for row in data:
        site = row["site"]
        if site not in site_data:
            site_data[site] = {"actual": [], "predicted": []}
        if row["actual_da"] is not None and row["predicted_da"] is not None:
            site_data[site]["actual"].append(row["actual_da"])
            site_data[site]["predicted"].append(row["predicted_da"])

    # Compute per-site metrics
    results = {}
    all_actual, all_predicted = [], []

    print(f"\n{'Site':<18} {'N':>5} {'Spikes':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'F1 95% CI':>18}")
    print("-" * 75)

    for site in SITE_ORDER:
        if site not in site_data:
            continue
        y_true = np.array(site_data[site]["actual"])
        y_pred = np.array(site_data[site]["predicted"])
        all_actual.extend(site_data[site]["actual"])
        all_predicted.extend(site_data[site]["predicted"])

        metrics = compute_spike_metrics(y_true, y_pred)
        ci = bootstrap_f1(y_true, y_pred)
        metrics.update(ci)
        results[site] = metrics

        f1_str = f"{metrics['f1']:.3f}" if not np.isnan(metrics['f1']) else "---"
        ci_str = f"[{ci['f1_ci_lo']:.3f}, {ci['f1_ci_hi']:.3f}]" if not np.isnan(ci['f1_ci_lo']) else "---"
        prec_str = f"{metrics['precision']:.3f}" if not np.isnan(metrics['precision']) else "---"
        rec_str = f"{metrics['recall']:.3f}" if not np.isnan(metrics['recall']) else "---"
        print(f"{site:<18} {metrics['n']:>5} {metrics['n_spikes']:>7} {prec_str:>7} {rec_str:>7} {f1_str:>7} {ci_str:>18}")

    # Overall
    y_true_all = np.array(all_actual)
    y_pred_all = np.array(all_predicted)
    overall = compute_spike_metrics(y_true_all, y_pred_all)
    overall_ci = bootstrap_f1(y_true_all, y_pred_all)
    overall.update(overall_ci)
    results["Overall"] = overall

    print("-" * 75)
    print(f"{'Overall':<18} {overall['n']:>5} {overall['n_spikes']:>7} {overall['precision']:>7.3f} "
          f"{overall['recall']:>7.3f} {overall['f1']:>7.3f} [{overall_ci['f1_ci_lo']:.3f}, {overall_ci['f1_ci_hi']:.3f}]")

    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"threshold": SPIKE_THRESHOLD, "cache_dir": str(CACHE_DIR), "results": results}, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    # Generate LaTeX table
    with open(OUTPUT_LATEX, "w") as f:
        f.write("% Spike detection metrics (DA > 20 µg/g)\n")
        f.write(f"% Generated from {CACHE_DIR}\n\n")
        f.write("\\begin{table}[H]\n")
        f.write("\\caption{Spike detection performance (DA $> \\SI{20}{\\microgrampeg}$, the FDA regulatory action level) "
                "per site. Precision, recall, and F1 evaluate binary classification of spike events. "
                "Bootstrap 95\\% confidence intervals ($B = 2{,}000$) are shown for F1.}\n")
        f.write("\\label{tab:spike-f1}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Site} & $\\boldsymbol{N}$ & \\textbf{Spikes} & \\textbf{Precision} & "
                "\\textbf{Recall} & \\textbf{F1} & \\textbf{F1 [95\\% CI]} \\\\\n")
        f.write("\\midrule\n")

        for i, site in enumerate(SITE_ORDER):
            if site not in results:
                continue
            m = results[site]

            # Add spacing between tiers
            if i == 5:  # After Kalaloch (5 WA sites)
                f.write("\\addlinespace\n")
            if i == 8:  # After Gold Beach
                f.write("\\addlinespace\n")

            if np.isnan(m["f1"]):
                f.write(f"{site} & {m['n']} & {m['n_spikes']} & --- & --- & --- & --- \\\\\n")
            else:
                ci_str = f"[{m['f1_ci_lo']:.2f}, {m['f1_ci_hi']:.2f}]"
                f.write(f"{site:<18} & {m['n']} & {m['n_spikes']} & {m['precision']:.2f} & "
                        f"{m['recall']:.2f} & {m['f1']:.2f} & {ci_str} \\\\\n")

        f.write("\\midrule\n")
        m = results["Overall"]
        ci_str = f"[{m['f1_ci_lo']:.2f}, {m['f1_ci_hi']:.2f}]"
        f.write(f"\\textbf{{Overall}} & \\textbf{{{m['n']:,}}} & \\textbf{{{m['n_spikes']}}} & "
                f"\\textbf{{{m['precision']:.2f}}} & \\textbf{{{m['recall']:.2f}}} & "
                f"\\textbf{{{m['f1']:.2f}}} & {ci_str} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved: {OUTPUT_LATEX}")


if __name__ == "__main__":
    main()
