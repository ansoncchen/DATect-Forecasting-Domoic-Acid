# Fixing Disk Quota Issues on HYAK

## Problem
Conda environments are stored in `~/.conda/envs/` by default, which counts against your home directory quota (~10GB). Large packages like PyTorch can easily exceed this.

## Solution: Move Conda Environment to Scratch Space

### Step 1: Check Current Usage
```bash
# Check your quota
hyakstorage

# See what's taking space
du -sh ~/.conda ~/.cache ~/.local 2>/dev/null | sort -hr
```

### Step 2: Create Conda Environment in Scratch Space

**Option A: Clone existing environment to scratch (if you want to keep it)**
```bash
# Load conda
module load foster/python/miniconda/3.8

# Create environment directory in scratch
mkdir -p /gscratch/stf/YOUR_UWNETID/conda_envs

# Clone existing environment to scratch
conda create --prefix /gscratch/stf/YOUR_UWNETID/conda_envs/datect --clone datect -y

# Remove old environment from home (frees ~8-10GB)
conda env remove -n datect
```

**Option B: Create fresh environment in scratch (recommended if installation failed)**
```bash
# Load conda
module load foster/python/miniconda/3.8

# Create environment directory in scratch
mkdir -p /gscratch/stf/YOUR_UWNETID/conda_envs

# Create new environment in scratch space
conda create --prefix /gscratch/stf/YOUR_UWNETID/conda_envs/datect python=3.10 -y

# Activate using full path
conda activate /gscratch/stf/YOUR_UWNETID/conda_envs/datect

# Upgrade pip
pip install --upgrade pip

# Install dependencies
cd /gscratch/stf/YOUR_UWNETID/DATect-Forecasting-Domoic-Acid
pip install -r requirements.txt
```

### Step 3: Set Up Convenient Activation

Create an alias or script to easily activate the environment:

```bash
# Add to ~/.bashrc
echo 'alias datect-env="conda activate /gscratch/stf/YOUR_UWNETID/conda_envs/datect"' >> ~/.bashrc
source ~/.bashrc

# Or create a simple activation script
cat > ~/activate_datect.sh << 'EOF'
#!/bin/bash
module load foster/python/miniconda/3.8
conda activate /gscratch/stf/YOUR_UWNETID/conda_envs/datect
cd /gscratch/stf/YOUR_UWNETID/DATect-Forecasting-Domoic-Acid
EOF

chmod +x ~/activate_datect.sh
```

### Step 4: Clean Up Home Directory

After moving the environment, clean up:

```bash
# Clean conda cache (if any remains)
conda clean --all -y

# Clean pip cache
pip cache purge

# Remove old conda environments (if any)
conda env list  # Check what's left
conda env remove -n old_env_name  # Remove unused ones

# Check freed space
du -sh ~/.conda ~/.cache 2>/dev/null
hyakstorage
```

## Alternative: Use Conda's CONDA_ENVS_PATH

You can also configure conda to use scratch space by default:

```bash
# Add to ~/.bashrc
export CONDA_ENVS_PATH=/gscratch/stf/YOUR_UWNETID/conda_envs
```

Then all new environments will be created in scratch space.

## Verify It Works

```bash
# Activate environment
conda activate /gscratch/stf/YOUR_UWNETID/conda_envs/datect

# Check Python location (should be in scratch)
which python

# Test imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
```

## Notes

- Scratch space (`/gscratch`) has much more capacity than home directory
- Environments in scratch persist across sessions (same as home)
- Always use full path when activating: `conda activate /gscratch/stf/YOUR_UWNETID/conda_envs/datect`
- Replace `YOUR_UWNETID` with your actual UW NetID
