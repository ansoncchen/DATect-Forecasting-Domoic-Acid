#!/usr/bin/env python3
"""
DATect Stability Table Generator (Phase 3)

Reads eval_results/stability/stability_results.json (produced by
paper_stability_study.py) and generates paper-ready tables in both
plain text and LaTeX formats.

Usage:
    python3 paper_stability_table.py                    # plain text
    python3 paper_stability_table.py --latex             # LaTeX output
    python3 paper_stability_table.py --output tables.tex # save to file
"""

import argparse
import json
import os
import sys

INPUT_FILE = os.path.join("eval_results", "stability", "stability_results.json")

WA_SITES = ["Kalaloch", "Quinault", "Copalis", "Twin Harbors", "Long Beach"]
OR_SITES = ["Clatsop Beach", "Cannon Beach", "Newport", "Coos Bay", "Gold Beach"]
ALL_SITES = WA_SITES + OR_SITES


def load_results() -> dict:
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run paper_stability_study.py first.")
        sys.exit(1)
    with open(INPUT_FILE) as f:
        return json.load(f)


# ── Plain-text tables ─────────────────────────────────────────────────────────

def print_table_a(data: dict):
    """Table A: Multi-Seed Stability."""
    nf = data.get("analysis", {}).get("noise_floor")
    if not nf:
        print("No Phase 1A data available.")
        return

    seeds = data.get("metadata", {}).get("seeds", [42, 123, 456, 789, 1337])
    per_site = nf["per_site"]

    print(f"\n{'='*100}")
    print("TABLE A — Multi-Seed Stability (per-site ON)")
    print(f"{'='*100}")

    # Header
    seed_cols = "".join(f"  R²({s})" for s in seeds)
    print(f"  {'Site':<18}{seed_cols}  {'Mean':>8}  {'Std':>7}")
    print("  " + "─" * 90)

    for site in ALL_SITES:
        entry = per_site.get(site, {})
        vals = entry.get("on_values", [])
        if not vals:
            print(f"  {site:<18}  {'n/a':>8}" * len(seeds))
            continue
        val_str = "".join(f"  {v:>7.4f}" for v in vals)
        mean = entry.get("on_mean", 0)
        std = entry.get("on_std", 0)
        print(f"  {site:<18}{val_str}  {mean:>8.4f}  {std:>6.4f}")

    # Pooled
    print("  " + "─" * 90)
    p_on_m = nf.get("pooled_on_mean")
    p_on_s = nf.get("pooled_on_std")
    if p_on_m is not None:
        print(f"  {'Pooled':<18}{'':>{8*len(seeds)}}  {p_on_m:>8.4f}  {p_on_s:>6.4f}")


def print_table_b(data: dict):
    """Table B: Perturbation Sensitivity."""
    pt = data.get("analysis", {}).get("perturbations")
    if not pt:
        print("No Phase 1B data available.")
        return

    print(f"\n{'='*90}")
    print("TABLE B — Perturbation Sensitivity")
    print(f"{'='*90}")
    print(f"  {'Perturbation':<35} {'R²':>8} {'ΔR²':>8} {'MAE':>8} {'Sig?':>6}")
    print("  " + "─" * 70)
    print(f"  {'Baseline':<35} {pt['baseline_r2']:>8.4f} {'---':>8}")

    for key, entry in pt["results"].items():
        if "r2" not in entry:
            print(f"  {entry.get('name', key):<35} {'FAILED':>8}")
            continue
        sig = "YES*" if entry["significant"] else "no"
        print(f"  {entry['name']:<35} {entry['r2']:>8.4f} "
              f"{entry['delta_r2']:>+8.4f} {entry['mae']:>8.2f} {sig:>6}")

    print(f"\n  * Significant: |ΔR²| > {pt['significance_threshold']:.4f} (2× seed std)")


def print_table_c(data: dict):
    """Table C: Model Selection Robustness."""
    nf = data.get("analysis", {}).get("noise_floor")
    if not nf:
        return

    per_site = nf["per_site"]

    print(f"\n{'='*90}")
    print("TABLE C — Per-Site Tuning Robustness")
    print(f"{'='*90}")
    print(f"  {'Site':<18} {'Current':>10} {'Δ(ON-OFF)':>10} {'Δ std':>8} "
          f"{'Consistent':>11} {'Verdict':>10}")
    print("  " + "─" * 75)

    for site in ALL_SITES:
        entry = per_site.get(site, {})
        on_m = entry.get("on_mean")
        d_m = entry.get("delta_mean")
        d_s = entry.get("delta_std")
        if on_m is None or d_m is None:
            print(f"  {site:<18}  n/a")
            continue

        consistent = entry.get("delta_all_positive") or entry.get("delta_all_negative")
        direction = "always +" if entry.get("delta_all_positive") else (
            "always −" if entry.get("delta_all_negative") else "flips sign"
        )

        if consistent and d_m > 0:
            verdict = "keep"
        elif consistent and d_m < 0:
            verdict = "remove"
        else:
            verdict = "unstable"

        print(f"  {site:<18} {on_m:>10.4f} {d_m:>+10.4f} {d_s:>8.4f} "
              f"{direction:>11} {verdict:>10}")


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def latex_table_a(data: dict) -> str:
    """Generate LaTeX for Table A."""
    nf = data.get("analysis", {}).get("noise_floor")
    if not nf:
        return "% No Phase 1A data\n"

    seeds = data.get("metadata", {}).get("seeds", [42, 123, 456, 789, 1337])
    per_site = nf["per_site"]
    n_seeds = len(seeds)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Multi-seed stability of per-site R\textsuperscript{2} (20\% sample).}",
        r"\label{tab:stability}",
        r"\begin{tabular}{l" + "r" * n_seeds + "rr}",
        r"\toprule",
        "Site & " + " & ".join(f"Seed {s}" for s in seeds) + r" & Mean & Std \\",
        r"\midrule",
    ]

    for site in ALL_SITES:
        entry = per_site.get(site, {})
        vals = entry.get("on_values", [])
        if not vals:
            continue
        mean = entry.get("on_mean", 0)
        std = entry.get("on_std", 0)
        val_cells = " & ".join(f"{v:.3f}" for v in vals)
        lines.append(f"{site} & {val_cells} & {mean:.3f} & {std:.3f} \\\\")

    lines.extend([
        r"\midrule",
        f"Pooled & & & & & {nf.get('pooled_on_mean', 0):.3f} & {nf.get('pooled_on_std', 0):.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def latex_table_b(data: dict) -> str:
    """Generate LaTeX for Table B."""
    pt = data.get("analysis", {}).get("perturbations")
    if not pt:
        return "% No Phase 1B data\n"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Perturbation sensitivity analysis ($\Delta$R\textsuperscript{2} vs.\ baseline).}",
        r"\label{tab:perturbation}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Perturbation & R\textsuperscript{2} & $\Delta$R\textsuperscript{2} & MAE & Sig.\ \\",
        r"\midrule",
        f"Baseline & {pt['baseline_r2']:.3f} & --- & & \\\\",
    ]

    for key, entry in pt["results"].items():
        if "r2" not in entry:
            continue
        sig = r"\checkmark" if entry["significant"] else ""
        name = entry["name"].replace("→", r"$\rightarrow$")
        lines.append(
            f"{name} & {entry['r2']:.3f} & {entry['delta_r2']:+.3f} "
            f"& {entry['mae']:.2f} & {sig} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\\\[2pt] \\footnotesize Significance: $|\\Delta R^2| > {pt['significance_threshold']:.3f}$ (2$\\times$ seed std).",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate stability tables")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX format")
    parser.add_argument("--output", type=str, help="Save output to file")
    args = parser.parse_args()

    data = load_results()

    if args.latex:
        output = []
        output.append("% DATect Stability Study — LaTeX Tables")
        output.append(f"% Generated from {INPUT_FILE}\n")
        output.append(latex_table_a(data))
        output.append("")
        output.append(latex_table_b(data))

        text = "\n".join(output)
        if args.output:
            with open(args.output, "w") as f:
                f.write(text)
            print(f"LaTeX saved to {args.output}")
        else:
            print(text)
    else:
        print_table_a(data)
        print_table_b(data)
        print_table_c(data)

        recs = data.get("analysis", {}).get("recommendations", [])
        if recs:
            print(f"\n{'='*90}")
            print("RECOMMENDATIONS")
            print(f"{'='*90}")
            for rec in recs:
                print(f"  → {rec}")
        print()


if __name__ == "__main__":
    main()
