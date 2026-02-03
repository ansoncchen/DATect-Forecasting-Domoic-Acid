#!/usr/bin/env python3
"""
Measure regression R² with the current config (same protocol as retrospective).
Run from repo root: python scripts/measure_regression_r2.py

Use this to compare before/after changes (e.g. toggle features in config).
"""
import os
import sys

# Run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    import config
    from forecasting.forecast_engine import ForecastEngine

    n_anchors = int(os.getenv("N_ANCHORS", "100"))  # 100 for quick; 500 for full
    model_type = getattr(config, "FORECAST_MODEL", "xgboost")

    print(f"Config: FORECAST_MODEL={model_type}, N_ANCHORS={n_anchors}")
    print(f"Features: USE_LAG={config.USE_LAG_FEATURES}, USE_ROLLING={config.USE_ROLLING_FEATURES}, USE_LOG_TARGET={config.USE_LOG_TARGET_TRANSFORM}")
    print("Running retrospective evaluation...")

    engine = ForecastEngine(validate_on_init=False)
    df = engine.run_retrospective_evaluation(
        task="regression",
        model_type=model_type,
        n_anchors=n_anchors,
        min_test_date="2008-01-01",
    )

    if df is not None and not df.empty:
        from sklearn.metrics import r2_score, mean_absolute_error
        valid = df.dropna(subset=["actual_da", "predicted_da"])
        r2 = r2_score(valid["actual_da"], valid["predicted_da"])
        mae = mean_absolute_error(valid["actual_da"], valid["predicted_da"])
        print(f"\nRegression R² = {r2:.4f}, MAE = {mae:.4f} (n={len(valid)})")
        return r2
    print("No results.")
    return None

if __name__ == "__main__":
    main()
