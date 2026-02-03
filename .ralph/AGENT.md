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

# Run model evaluation (~8 min)
python3 precompute_cache.py
```

## Test Commands
```bash
# Verify temporal integrity (REQUIRED after changes)
python3 verify_temporal_integrity.py
```

## Other Commands
```bash
# Regenerate dataset (30-60 min - only when data changes)
python3 dataset-creation.py

# Frontend development
cd frontend && npm run dev
cd frontend && npm run build
```
