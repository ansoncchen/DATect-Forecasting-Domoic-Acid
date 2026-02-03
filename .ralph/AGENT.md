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

# Run model evaluation (~5 min)
python3 precompute_cache.py
```

## Test Commands
```bash
# Temporal integrity is automatically validated during precompute_cache.py
# No separate test script needed
```

## Other Commands
```bash
# Regenerate dataset (30-60 min - only when data changes)
python3 dataset-creation.py

# Frontend development
cd frontend && npm run dev
cd frontend && npm run build
```
