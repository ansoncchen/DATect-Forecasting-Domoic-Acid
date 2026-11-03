"""
Does picking weights and clipping per site beat one shared setting?

Reads the re-tune's Delta prediction tables (``retune predict``), scores every
weight x clip setting applied to all sites at once with the same yearly folds,
objective and tie-break as ``retune score``, and compares on the holdout:
the current and proposed per-site settings, the shared fold winner and simple
shared rules. Paired bootstrap CIs are against the equal blend without clipping.

Usage (Hyak, from the repository root, same environment as retune.sbatch):
    python -m datect_spark.shared_vs_per_site
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from datect_spark import retune
from datect_spark.session import get_spark, lake_path


def _setting(w_xgb: float, clip_q: float, clip_max: float | None) -> dict:
    return {"ensemble_weights": (w_xgb, 1.0 - w_xgb, 0.0),
            "prediction_clip_q": clip_q, "prediction_clip_max": clip_max}


def main() -> None:
    spark = get_spark("datect-shared-vs-per-site")
    tables = {w: spark.read.format("delta").load(lake_path("gold", f"retune_predictions_{w}"))
              .where("ok").toPandas() for w in ("tuning", "holdout")}
    spark.stop()
    tune, hold = tables["tuning"], tables["holdout"].reset_index(drop=True)
    tune["fold"] = pd.to_datetime(tune["date"]).dt.year

    rows = []
    for w in retune.W_XGB_GRID:
        for q in retune.CLIP_Q_GRID:
            for m in retune.CLIP_MAX_GRID:
                pred = retune._apply(tune, _setting(w, q, m))
                r2 = [retune._r2(g["actual_da"].to_numpy(), pred[g.index]) for _, g in tune.groupby("fold")]
                rows.append({"w_xgb": w, "clip_q": q, "clip_max": m, "mean_r2": np.mean(r2),
                             "std_r2": np.std(r2), "score": np.mean(r2) - 0.5 * np.std(r2)})
    grid = pd.DataFrame(rows)
    # Same tie-break as retune.score: no hard ceiling, looser quantile, more XGBoost
    grid["has_max"] = grid["clip_max"].notna()
    grid["score_key"] = grid["score"].round(10)
    grid = grid.sort_values(["score_key", "has_max", "clip_q", "w_xgb"],
                            ascending=[False, True, False, False]).drop(columns=["has_max", "score_key"])
    best = grid.iloc[0]

    with open(os.path.join(retune.OUT_DIR, "proposed_overrides.json")) as f:
        proposed = json.load(f)
    y = hold["actual_da"].to_numpy()
    candidates = {
        "current_per_site": retune._ensemble(hold, retune._current_configs()).to_numpy(),
        "proposed_per_site": retune._ensemble(hold, proposed).to_numpy(),
        "shared_fold_winner": retune._apply(hold, _setting(best.w_xgb, best.clip_q,
                                                           None if pd.isna(best.clip_max) else best.clip_max)),
        "shared_equal_no_clip": retune._apply(hold, _setting(0.5, 1.0, None)),
        "shared_rf_no_clip": retune._apply(hold, _setting(0.0, 1.0, None)),
        "shared_xgb_no_clip": retune._apply(hold, _setting(1.0, 1.0, None)),
    }
    ref = candidates["shared_equal_no_clip"]
    boot = np.random.default_rng(0).integers(0, len(y), size=(2000, len(y)))
    out = {"shared_fold_winner_setting": {"w_xgb": float(best.w_xgb), "clip_q": float(best.clip_q),
                                          "clip_max": None if pd.isna(best.clip_max) else float(best.clip_max)},
           "holdout": {}}
    for name, pred in candidates.items():
        delta = [retune._r2(y[i], pred[i]) - retune._r2(y[i], ref[i]) for i in boot]
        out["holdout"][name] = {"r2": retune._r2(y, pred), "mae": float(np.abs(y - pred).mean()),
                                "delta_r2_vs_shared_equal_no_clip": retune._r2(y, pred) - retune._r2(y, ref),
                                "ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))]}

    grid.to_csv(os.path.join(retune.OUT_DIR, "shared_setting_scores.csv"), index=False)
    with open(os.path.join(retune.OUT_DIR, "shared_vs_per_site_holdout.json"), "w") as f:
        json.dump(out, f, indent=2)
    for name, r in out["holdout"].items():
        print(f"{name:22s} R2 {r['r2']:.3f}  MAE {r['mae']:.2f}  "
              f"dR2 {r['delta_r2_vs_shared_equal_no_clip']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
