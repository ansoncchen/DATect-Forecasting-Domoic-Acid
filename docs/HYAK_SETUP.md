# Hyak (klone) Setup and Workflow

All heavy compute (evaluations, tuning sweeps, `precompute_cache.py`, OAD training) runs on Hyak. The laptop is for editing, review and running the dashboard against a pre-computed cache.

## SSH

`~/.ssh/config` on your machine:

```
Host klone-login
    User YOUR_UWNETID
    Hostname klone.hyak.uw.edu
    ServerAliveInterval 30
    ServerAliveCountMax 1200
    ControlMaster auto
    ControlPersist 3600
    ControlPath ~/.ssh/%r@klone-login:%p

Host klone-node
    User YOUR_UWNETID
    Hostname n3000          # replace with the node salloc gives you
    ProxyJump klone-login
```

The ControlMaster connection expires periodically (2FA); re-authenticate from a terminal with `ssh klone-login 'whoami'`.

**Do not poll Hyak in SSH loops.** Many failed or rapid connections get your client IP blocked (symptom: `Connection refused` on `ssh klone-login` from every terminal while jobs keep running). If that happens, stop all polling and wait 15–30 minutes. For long waits, check by hand or with a single poll using long, increasing intervals.

`klone-node` is only needed for interactive work on a compute node: `salloc --partition=ckpt --account=stf-ckpt --cpus-per-task=4 --mem=32G --time=4:00:00`, then put the granted node name in the `Hostname` line.

## Environment (one recipe)

The project uses a single conda environment with Python 3.10 on `/gscratch`. The existing one is `/gscratch/stf/ac283/envs/datect_scratch`; it is the default interpreter of every sbatch file. To create your own:

```bash
module load foster/python/miniconda/3.8
conda create -p /gscratch/stf/$USER/envs/datect_scratch python=3.10 -y
PY=/gscratch/stf/$USER/envs/datect_scratch/bin/python

cd /gscratch/stf/$USER/DATect-Forecasting-Domoic-Acid     # your clone
$PY -m pip install -r requirements.txt -r requirements-dev.txt -r oad/requirements.txt
```

`requirements.txt` covers the engine, API and `dataset-creation.py`; `requirements-dev.txt` adds pytest, ruff and matplotlib; `oad/requirements.txt` adds torch, zarr and dask for the OAD subproject. Call the interpreter by path (no activation needed). DATect sbatch files take it from `DATECT_PY`:

```bash
export DATECT_PY=/gscratch/stf/$USER/envs/datect_scratch/bin/python
```

## Submitting DATect jobs

The sbatch files in `scripts/eval/` and `scripts/tune/` run the checkout they were **submitted from** (`$SLURM_SUBMIT_DIR`), so submit from the repo root:

```bash
cd /gscratch/stf/$USER/DATect-Forecasting-Domoic-Acid
mkdir -p logs                     # Slurm opens logs/... before the script runs
sbatch scripts/eval/multi_seed_baseline.sbatch
```

- **Partition:** `--partition=ckpt --account=stf-ckpt --requeue` (already set in the files). `ckpt` is preemptible but free; the `compute` partition often fails with `QOSGrpMemLimit` even for modest memory requests. A requeued job restarts from the beginning, so long jobs should write resumable outputs.
- **Per-anchor tuning** is off by default (`DATECT_PER_ANCHOR_TUNING`), so outer loops no longer need `DATECT_MIN_TRAINING_FOR_TUNING=99999`.
- **Parallel tasks must not write shared files** (e.g. `data/processed/final_output.parquet`); give each task its own output or serialize with `--array=0-N%1`.
- **Progress:** `print()` output is buffered into `.out`; tqdm writes to `.err`. Check with `tail -1 logs/*.err | tr "\r" "\n" | tail -3`.

Script-level documentation is in the script index of [`QUICKSTART_FOR_REVIEWERS.md`](QUICKSTART_FOR_REVIEWERS.md).

## Golden-baseline workflow

Every evaluation that backs a reported number can be snapshotted for a git ref and compared file by file. Behavior-neutral commits must reproduce the previous snapshot exactly.

```bash
# from any Hyak checkout of the repo
bash scripts/eval/golden_baseline.sh <git-ref> [label]
```

This creates a detached worktree at `${GOLDEN_ROOT:-/gscratch/scrubbed/$USER/golden}/<label>` (label defaults to the short SHA) and submits `scripts/eval/golden_baseline.sbatch` inside it (ckpt, 32 cores, 96 GB, 6 h). The job records `golden_sha.txt` and runs, with the default (headline) config:

1. `chronological_eval.py` on the validation window `2019-01-01:2022-01-01` and the holdout `2022-01-01:2024-01-01`;
2. `multi_seed_baseline.py` for seeds 42–46 (into `golden_multi_seed/`);
3. `eval_paper_metrics.py --seed 123 --sample-fraction 0.40 --force-rerun`;
4. `spike_detection_eval.py --seed 123 --sample-fraction 0.40`.

Compare two snapshots:

```bash
$DATECT_PY scripts/eval/compare_golden.py <golden_root>/<label_a> <golden_root>/<label_b>
```

Prediction parquets must match exactly; metric tables may differ only within a relative 1e-12 (reported as rounding). Exit code 0 means identical. For differing files it prints the headline-metric deltas to quote in the commit message. Remove a finished worktree with `git worktree remove <path>`.

## Pre-computed dashboard cache

```bash
$DATECT_PY precompute_cache.py      # on a compute node or in a batch job, from the repo root
```

Then copy `cache/` to your machine (see [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md)).

## OAD subproject

- OAD data and outputs live under `/gscratch/stf/ac283/oad_current`, a symlink to the OAD tree on scratch (raw NetCDFs under `data/raw/`, the cube at `data/cube.zarr`, and `outputs/`).
- The OAD sbatch files in `oad/hyak/` `cd` into `OAD_ROOT` (default: that symlink) and run the OAD code found there (`config.py`, `scripts/`, `src/`), not the checkout you submit from. Point them elsewhere with `sbatch --export=ALL,OAD_ROOT=/path/to/oad ...`.
- Their interpreter is fixed to `/gscratch/stf/ac283/envs/datect_scratch/bin/python` (they do not read `DATECT_PY`). Most use `--account=stf` with GPU or ckpt partitions as set in each file.
- After new scores are produced, bring a checkpoint into DATect with `scripts/data/export_oad_features.py <checkpoint> --scores-dir <OAD_ROOT>/outputs/scores` (see [`DATA_PIPELINE_DETAILED.md`](DATA_PIPELINE_DETAILED.md)).

Workflow details: `oad/README.md`, `oad/AGENTS.md`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Access denied by pam_slurm_adopt: you have no active jobs on this node` | The interactive job ended; `salloc` again and update `klone-node`. |
| `Connection refused` from `ssh klone-login` everywhere | IP block from too many connections; stop polling and wait 15–30 min. |
| Job fails instantly with no log | `logs/` did not exist in the submit directory. |
| `SLURM_SUBMIT_DIR: submit from the repo root` | The file was run with `bash` instead of submitted; use `sbatch` from the repo root. |
| `QOSGrpMemLimit` pending reason | Use the ckpt partition. |
