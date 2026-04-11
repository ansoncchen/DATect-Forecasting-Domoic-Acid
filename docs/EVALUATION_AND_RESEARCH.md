# Evaluation and research scripts

Paper and Hyak evaluation entry points live under **`scripts/eval/`** (run commands from the **repository root** so `config`, `data/`, and `eval_results/` resolve correctly). `scripts/eval/_repo.py` sets `sys.path` and `os.chdir` to the repo root for each script.

They are **not** imported by `run_datect.py` or the FastAPI app.

## Why not one “mega” Hyak script?

- **`paper_stability_study.py`** and **`paper_ablation_study.py`** rely on **subprocesses** so `config` reloads cleanly between experiments (same pattern as joblib workers).
- Runtimes differ widely (seconds vs hours); a single driver would still shell out or duplicate that logic.
- **`run_full_validation.sh`** already orchestrates parallel jobs for a full paper refresh.

## Script index (`scripts/eval/`)

| Script | Role | Typical run location | Main outputs |
|--------|------|----------------------|--------------|
| `eval_paper_metrics.py` | Tables 1–3, bootstrap CIs, weight robustness | Hyak or strong local | `eval_results/paper_metrics/` |
| `paper_ablation_study.py` | Component ablation (subprocess per variant) | Hyak | `paper_ablation_results.json` (repo root) |
| `paper_stability_study.py` | Multi-seed + perturbation sensitivity (Phase 1A/1B) | Hyak (`--quick` locally) | `eval_results/stability/stability_results.json` |
| `paper_stability_table.py` | Formats stability JSON → Markdown/LaTeX | Any | stdout / file per `--output` |
| `spike_detection_eval.py` | Spike / transition metrics vs naive (retrospective-backed) | Hyak or local with cache | `eval_results/spike_metrics/` |

**Example (from repo root):**

```bash
python3 scripts/eval/eval_paper_metrics.py --seed 123 --force-rerun
python3 scripts/eval/paper_stability_study.py --quick
python3 scripts/eval/paper_stability_table.py --latex
```

## Automated tests

| Path | Role |
|------|------|
| `tests/test_leakage_audit.py` | Pytest leakage / integrity checks |

## Production vs research code

- **`forecasting/`** — training and prediction paths used by the API and eval scripts.
- **`forecasting/validation.py`** — lightweight checks at **API startup**, not full paper metrics.
- **`backend/`**, **`frontend/`** — dashboard and cache; no dependency on `scripts/eval/`.

## Documentation map (overlap on purpose)

| Topic | Primary doc |
|-------|-------------|
| End-to-end **forecast** flow, leakage rules | `PIPELINE_DEEP_DIVE.md` |
| **`dataset-creation.py`** | `DATA_PIPELINE_DETAILED.md` |
| Hyak + cache | `HYAK_SETUP.md` |
| Cache → GCP | `DEPLOYMENT_GUIDE.md` |
