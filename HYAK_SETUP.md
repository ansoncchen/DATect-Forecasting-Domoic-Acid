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

```bash
ssh klone-login
salloc --partition=ckpt --cpus-per-task=4 --mem=32G --time=4:00:00 --job-name=vsc-proxy-jump
```

Note the node number (e.g., `n3285`).

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

### Workflow A: validate_on_raw_data.py (analysis + validation)

Use this when you want full validation plots, metrics analysis, and the detailed CSV.

```bash
# On Hyak compute node:
python validate_on_raw_data.py
```

Then on your **local Mac**:
```bash
# Download results
scp -r klone-node:/gscratch/stf/YOUR_UWNETID/DATect-Forecasting-Domoic-Acid/raw_validation_plots/ ./raw_validation_plots/

# Convert CSV to API cache format
python convert_results_to_cache.py

# Start local dashboard
python run_datect.py
```

### Workflow B: precompute_cache.py (deployment cache)

Use this when you just need the API cache for the local dashboard or production deployment.

```bash
# On Hyak compute node:
python precompute_cache.py
```

Then on your **local Mac**:
```bash
# Download cache
scp -r klone-node:/gscratch/stf/YOUR_UWNETID/DATect-Forecasting-Domoic-Acid/cache/ ./cache/

# Start local dashboard
python run_datect.py
```

### Other Commands

```bash
# Generate dataset (30-60 min, only when data sources change)
python dataset-creation.py
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

**Environment already exists error**
- Use: `conda activate datect` (it's already created)
- Or remove and recreate: `conda env remove -n datect` then create again
