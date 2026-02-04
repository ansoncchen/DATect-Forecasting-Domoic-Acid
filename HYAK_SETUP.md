# Hyak (klone) Setup Guide for DATect

## One-Time SSH Setup

### On klone (remote cluster)

```bash
ssh YOUR_UWNETID@klone.hyak.uw.edu

# Generate intracluster SSH key
ssh-keygen -C klone -t rsa -b 2048 -f ~/.ssh/id_rsa -q -N ""
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### On your local Mac

**Create `~/.ssh/config`:**
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
    Include klone-node-config
```

**Create `~/.ssh/klone-node-config`:**
```
Host klone-node
  User YOUR_UWNETID
  Hostname n3000
  ProxyJump klone-login
```

**Set permissions and copy key:**
```bash
chmod 600 ~/.ssh/config ~/.ssh/klone-node-config
ssh-copy-id klone-login
```

---

## Each Session Workflow

### 1. Request a compute node

**CPU-only (default):**
```bash
ssh klone-login
salloc --partition=ckpt --cpus-per-task=4 --mem=32G --time=4:00:00 --job-name=vsc-proxy-jump
```

**GPU (for XGBoost acceleration, TFT, etc.):**
```bash
ssh klone-login
salloc --partition=ckpt-g2 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=4:00:00 --job-name=vsc-proxy-jump
```

Partitions: `ckpt` (CPU), `ckpt-g2` (L40/L40S GPUs), `ckpt-all` (either). Note the node number (e.g., `n3285`).

### 2. Update local config with node number

On your **local Mac**, edit `~/.ssh/klone-node-config`:
```
Host klone-node
  User YOUR_UWNETID
  Hostname n3285
  ProxyJump klone-login
```

### 3. Connect directly to the compute node

```bash
ssh klone-node
```

Or use VS Code Remote-SSH extension → Connect to Host → `klone-node`

---

## First-Time Project Setup (on compute node)

```bash
# Create project directory
mkdir -p /gscratch/stf/YOUR_UWNETID
cd /gscratch/stf/YOUR_UWNETID

# Clone repository
git clone https://github.com/ansoncchen/DATect-Forecasting-Domoic-Acid.git
cd DATect-Forecasting-Domoic-Acid

# Checkout your branch (if not main)
git checkout ralph-improvement
```

---

## Python Environment Setup (first time only)

Using Miniconda (Hyak's preferred method):

```bash
# Load miniconda module
module load foster/python/miniconda/3.8

# Create conda environment with Python 3.10
conda create -n datect python=3.10 -y

# Activate the environment
conda activate datect

# Upgrade pip
pip install --upgrade pip

# Install project dependencies
cd /gscratch/stf/YOUR_UWNETID/DATect-Forecasting-Domoic-Acid
pip install -r requirements.txt
```

**GPU support (XGBoost):** XGBoost auto-detects GPUs via `nvidia-smi`. On GPU nodes, set `config.USE_GPU = True` or leave as `None` for auto-detect. Ensure `xgboost` is built with CUDA (default pip wheel on Linux usually includes GPU support).

**Note:** The conda environment is stored in `~/.conda/envs/datect` and persists across sessions.

---

## Python Environment Setup (subsequent sessions)

```bash
# Load miniconda and activate existing environment
module load foster/python/miniconda/3.8
conda activate datect

# Navigate to project
cd /gscratch/stf/YOUR_UWNETID/DATect-Forecasting-Domoic-Acid
```

---

## Running the Project

```bash
# Precompute cache (uses CPU parallelization - fast!)
python precompute_cache.py

# Generate dataset (30-60 min)
python dataset-creation.py

# Run the full system
python run_datect.py
```

---

## GPU vs CPU Usage

**Important:** The choice between GPU and CPU depends on your workload type.

### Use CPU (`USE_GPU = False` in config.py) for:
- `precompute_cache.py` - runs 5000 small parallel training jobs
- Any joblib-parallelized workload with many small models
- **Expected performance:** ~5 minutes for full cache generation

### Use GPU (`USE_GPU = True` in config.py) for:
- Single large model training with 100K+ rows
- Deep learning models (TFT, TCN) with large batch sizes
- Hyperparameter search on a single large dataset

### Why CPU is faster for parallel small jobs:

The retrospective evaluation trains thousands of small models (one per anchor point). With GPU:
- Each of 32 parallel workers initializes CUDA context
- Each transfers tiny datasets to GPU
- GPU memory contention causes crashes or slowdowns
- Overhead dominates actual compute time

With CPU:
- 32 workers run independently on CPU cores
- No data transfer overhead
- No memory contention
- ~5 minutes total vs hours with GPU

### Commands:

```bash
# Force CPU (recommended for precompute_cache.py)
# Option 1: Set in config.py (USE_GPU = False) - already the default
python precompute_cache.py

# Option 2: Hide GPUs via environment variable
CUDA_VISIBLE_DEVICES="" python precompute_cache.py

# Force GPU (for single large training jobs)
# Set USE_GPU = True in config.py, then:
python your_single_model_training.py
```

---

## End Session

```bash
exit  # from compute node
scancel --name vsc-proxy-jump  # cancel the job
```

---

## Troubleshooting

**"Access denied by pam_slurm_adopt: you have no active jobs on this node"**
- Your job ended or timed out. Request a new job with `salloc` and update `~/.ssh/klone-node-config` with the new node.

**Conda create gets stuck downloading packages**
- Network can be slow. Wait a few minutes. If stuck for 5+ minutes, `Ctrl+C` and try again.
- Alternative: try `conda create -n datect python=3.10 -c conda-forge -y`

**"ModuleNotFoundError: No module named '_ctypes'"**
- This happens with the `coenv/python` modules. Use Miniconda instead (as shown above).

**conda activate doesn't work**
- Make sure you ran `module load foster/python/miniconda/3.8` first.

**XGBoost GPU or CUDA errors**
- Request a GPU node: `salloc --partition=ckpt-g2 --gres=gpu:1 ...`
- Force CPU: set `USE_GPU = False` in `config.py` (uses tree_method='hist', device='cpu')
- Ensure `nvidia-smi` works on the node before running

**"cudaErrorMemoryAllocation: out of memory" during precompute_cache.py**
- This happens when parallel workers all try to use GPU simultaneously
- **Solution:** Set `USE_GPU = False` in `config.py` (already the default)
- Or run: `CUDA_VISIBLE_DEVICES="" python precompute_cache.py`
- CPU mode is actually faster for this workload (5 min vs hours)

**Environment already exists error**
- Use: `conda activate datect` (it's already created)
- Or remove and recreate: `conda env remove -n datect` then create again
