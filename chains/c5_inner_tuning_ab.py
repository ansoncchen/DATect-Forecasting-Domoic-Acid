"""
Chain 5: Inner per-anchor tuning A/B — does disabling it actually help?

Observation from holdout validation: baseline R² jumped from 0.17 (Task 10,
default DATect) → 0.42 (holdout baseline, with DATECT_MIN_TRAINING_FOR_TUNING=99999
which disables inner tuning). That's a 0.25 R² gap — way bigger than any tuning
or feature change we've tested.

This chain tests cleanly: same sampling, same per_site_models.py, only the
inner-tuning flag differs.

Task 0: inner tuning ENABLED (default)
Task 1: inner tuning DISABLED (set DATECT_MIN_TRAINING_FOR_TUNING=99999)

Both use min_test_date=2008-01-01 and the same N seed. Outputs:
  chains/output/inner_tuning_enabled.json
  chains/output/inner_tuning_disabled.json

If disabling reproducibly improves R² by >0.05, that's a clean DATect improvement
that doesn't require any new features.
"""
# This chain is special — it doesn't call add_features(). It's run by a
# dedicated sbatch (run_inner_tuning_ab.sbatch) that directly invokes the engine
# subprocess pattern with different env vars.

CHAIN_NAME = "inner_tuning_ab"
