#!/usr/bin/env python3
"""
Per-site model comparison: R² for each model type at each site.

Shows that no single model dominates everywhere, justifying the per-site
weighted ensemble approach.

Usage:
    python3 paper_per_site_model_comparison.py

Output:
    paper_model_comparison_latex.txt    (LaTeX table for paper)
    paper_model_comparison_results.json (full metrics)

By default uses the seed=42 development set (cache/retrospective/).
Set CACHE_DIR to cache_seed123/retrospective/ once that data is available.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

# ── Configuration ────────────────────────────────────────────────────────────
CACHE_DIR = Path("./cache_seed123/retrospective")
OUTPUT_JSON = "paper_model_comparison_results.json"
OUTPUT_LATEX = "paper_model_comparison_latex.txt"

MODEL_TYPES = ["ensemble", "xgboost", "rf", "naive", "linear"]
MODEL_DISPLAY = {
    "ensemble": "Ensemble",
    "xgboost": "XGBoost",
    "rf": "RF",
    "naive": "Na\\\"ive",
    "linear": "Ridge",
}

SITE_ORDER = [
    "Copalis", "Twin Harbors", "Quinault", "Long Beach", "Kalaloch",
    "Clatsop Beach", "Cannon Beach", "Gold Beach", "Coos Bay", "Newport",
]


def load_model_predictions(cache_dir: Path, model_type: str) -> list[dict]:
    """Load regression predictions for a model type."""
    json_path = cache_dir / f"regression_{model_type}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Cache file not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)


def compute_site_r2(data: list[dict]) -> dict[str, dict]:
    """Compute R² per site from prediction records."""
    site_data = {}
    for row in data:
        site = row["site"]
        if site not in site_data:
            site_data[site] = {"actual": [], "predicted": []}
        if row["actual_da"] is not None and row["predicted_da"] is not None:
            site_data[site]["actual"].append(row["actual_da"])
            site_data[site]["predicted"].append(row["predicted_da"])

    results = {}
    for site, vals in site_data.items():
        y_true = np.array(vals["actual"])
        y_pred = np.array(vals["predicted"])
        if len(y_true) < 2 or np.std(y_true) == 0:
            results[site] = {"r2": float("nan"), "mae": float("nan"), "n": len(y_true)}
        else:
            results[site] = {
                "r2": float(r2_score(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "n": len(y_true),
            }
    return results


def main():
    print("=" * 70)
    print("Per-Site Model Comparison (R² by model type)")
    print(f"Cache: {CACHE_DIR}")
    print("=" * 70)

    # Load all models
    all_results = {}
    for model_type in MODEL_TYPES:
        try:
            data = load_model_predictions(CACHE_DIR, model_type)
            all_results[model_type] = compute_site_r2(data)
            print(f"  Loaded {model_type}: {len(data)} predictions")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    # Print comparison table
    header = f"{'Site':<18}"
    for mt in MODEL_TYPES:
        header += f" {MODEL_DISPLAY[mt].replace(chr(92), ''):>10}"
    header += "  Best"
    print(f"\n{header}")
    print("-" * (18 + 10 * len(MODEL_TYPES) + 8))

    site_winners = {}
    for site in SITE_ORDER:
        row = f"{site:<18}"
        best_r2 = -999
        best_model = ""

        for mt in MODEL_TYPES:
            if mt in all_results and site in all_results[mt]:
                r2 = all_results[mt][site]["r2"]
                if not np.isnan(r2):
                    row += f" {r2:>10.3f}"
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = mt
                else:
                    row += f" {'---':>10}"
            else:
                row += f" {'---':>10}"

        row += f"  {best_model}"
        site_winners[site] = best_model
        print(row)

    # Count wins per model
    print(f"\nModel wins: ", end="")
    win_counts = {}
    for model in site_winners.values():
        win_counts[model] = win_counts.get(model, 0) + 1
    for mt in MODEL_TYPES:
        if mt in win_counts:
            print(f"{mt}={win_counts[mt]}", end="  ")
    print()

    # Save JSON
    output = {
        "cache_dir": str(CACHE_DIR),
        "model_types": MODEL_TYPES,
        "per_site": {},
    }
    for site in SITE_ORDER:
        output["per_site"][site] = {}
        for mt in MODEL_TYPES:
            if mt in all_results and site in all_results[mt]:
                output["per_site"][site][mt] = all_results[mt][site]

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    # Generate LaTeX table
    with open(OUTPUT_LATEX, "w") as f:
        f.write("% Per-site model comparison table\n")
        f.write(f"% Generated from {CACHE_DIR}\n\n")
        f.write("\\begin{table}[H]\n")
        f.write("\\caption{Per-site $R^2$ for each model variant. Bold values indicate the best-performing "
                "model at each site. No single model dominates across all sites, justifying the per-site "
                "weighted ensemble approach.}\n")
        f.write("\\label{tab:model-per-site}\n")
        f.write("\\small\n")

        col_spec = "l" + "r" * len(MODEL_TYPES)
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")

        # Header row
        header_cells = ["\\textbf{Site}"]
        for mt in MODEL_TYPES:
            header_cells.append(f"\\textbf{{{MODEL_DISPLAY[mt]}}}")
        f.write(" & ".join(header_cells) + " \\\\\n")
        f.write("\\midrule\n")

        for i, site in enumerate(SITE_ORDER):
            # Add spacing between tiers
            if i == 5:
                f.write("\\addlinespace\n")
            if i == 8:
                f.write("\\addlinespace\n")

            # Find best model for this site
            best_r2 = -999
            best_mt = ""
            for mt in MODEL_TYPES:
                if mt in all_results and site in all_results[mt]:
                    r2 = all_results[mt][site]["r2"]
                    if not np.isnan(r2) and r2 > best_r2:
                        best_r2 = r2
                        best_mt = mt

            cells = [site]
            for mt in MODEL_TYPES:
                if mt in all_results and site in all_results[mt]:
                    r2 = all_results[mt][site]["r2"]
                    if np.isnan(r2):
                        cells.append("---")
                    elif mt == best_mt:
                        if r2 < 0:
                            cells.append(f"\\textbf{{$-${abs(r2):.3f}}}")
                        else:
                            cells.append(f"\\textbf{{{r2:.3f}}}")
                    else:
                        if r2 < 0:
                            cells.append(f"$-${abs(r2):.3f}")
                        else:
                            cells.append(f"{r2:.3f}")
                else:
                    cells.append("---")
            f.write(" & ".join(cells) + " \\\\\n")

        # Overall row
        f.write("\\midrule\n")
        cells = ["\\textbf{Overall}"]
        for mt in MODEL_TYPES:
            if mt in all_results:
                # Compute overall R² from all predictions
                data = load_model_predictions(CACHE_DIR, mt)
                y_true = np.array([r["actual_da"] for r in data if r["actual_da"] is not None and r["predicted_da"] is not None])
                y_pred = np.array([r["predicted_da"] for r in data if r["actual_da"] is not None and r["predicted_da"] is not None])
                r2 = r2_score(y_true, y_pred)
                if r2 < 0:
                    cells.append(f"$-${abs(r2):.3f}")
                else:
                    cells.append(f"{r2:.3f}")
            else:
                cells.append("---")
        f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved: {OUTPUT_LATEX}")


if __name__ == "__main__":
    main()
