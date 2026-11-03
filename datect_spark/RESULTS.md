# DATect on Spark + Delta Lake + MLflow: results

Open-source Spark, Delta Lake and MLflow on UW's Hyak cluster. This is not Databricks,
and every job ran in Spark local mode on a single node. Each section reports what was
measured, the parity check that ties the Spark output to the existing pipeline, and
the caveats.

DRAFT: sections 2 and 3 are filled in from the Hyak runs.

## Hardware and versions

| | |
|---|---|
| Nodes | Hyak `ckpt` partition, 1 node per job, 16 cores and 32 GB requested (Slurm) |
| Spark | PySpark 4.0.1, local mode `local[16]`, Java 17 (`coenv/jdk/17.0.8`) |
| Delta Lake | delta-spark 4.0.0 (writes), deltalake 1.6.6 / delta-rs (API reads, no JVM) |
| MLflow | 3.17.2, sqlite tracking and model registry on `/gscratch/stf` |
| Python stack | Python 3.10, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, xgboost 3.1.3, pyarrow 23.0.0 |
| Storage | Lake and Spark scratch on `/gscratch/scrubbed` (purged after 21 idle days, rebuildable) |

## 1. Distributed re-tune with MLflow tracking (commit e76647bd)

Each of the ensemble's per-site weight and clip settings is post-processing, so one
XGBoost + Random Forest pair per test point serves every configuration. `retune.py
predict` trains those pairs on Spark (one task per real DA measurement, each on its own
pre-anchor data) and `score` applies the 100 weight × clip settings in Spark SQL.

| | |
|---|---|
| Forecasts trained | 1,990 (1,581 tuning points 2015–2021 + 409 holdout points 2022–2023), each an XGBoost + RF pair |
| Wall time, 16 cores | 883 s (675 s + 208 s) |
| Task time | 7,710 s (5,676 s + 2,034 s): 8.7× the wall time |
| MLflow runs | 2,990: 1,990 forecast runs + 1,000 (site, setting) scoring runs |
| Parity | The current configuration rebuilt from the Spark predictions equals the committed chronological predictions exactly (max abs diff 0.0 on 635 + 409 rows) |

Hyak job 42629914. The per-site picks it produced tie a single shared rule on the
holdout, so they were not adopted (`docs/CORRECTED_NUMBERS.md`).
