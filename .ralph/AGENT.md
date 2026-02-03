# DATect - Agent Build Instructions

## Project Type
Python (ML/Data Science)

## Build Commands
```bash
# Install dependencies
python3 -m pip install -r requirements.txt
```

## Run Commands
```bash
# Run full system (backend + frontend)
python3 run_datect.py

# Run model evaluation (~5 min) - MANDATORY after each model change
python3 precompute_cache.py
```

## Test Commands
```bash
# Temporal integrity is automatically validated during precompute_cache.py
# No separate test script needed
```

## Git Commands (MANDATORY - END OF EVERY ITERATION)
```bash
# Commit and push to ralph-improvement branch
git add -A
git commit -m "Ralph: [description] - R² = X.XXX"
git push origin ralph-improvement
```

## Other Commands
```bash
# Regenerate dataset (30-60 min - only when data changes)
python3 dataset-creation.py

# Frontend development
cd frontend && npm run dev
cd frontend && npm run build
```
