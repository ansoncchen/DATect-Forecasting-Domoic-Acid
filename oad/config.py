"""Central configuration for the ocean anomaly detection subproject."""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
CUBE_PATH = ROOT / "data" / "cube.zarr"
MODELS_DIR = ROOT / "models"
SCORES_DIR = ROOT / "outputs" / "scores"
FIGURES_DIR = ROOT / "outputs" / "figures"

# ---------------------------------------------------------------------------
# Spatial domain
# 41–49°N, 229–239.2°E  ≡  -131 to -120.8°W (ERDDAP _LonPM180 convention)
# ---------------------------------------------------------------------------
LAT_MIN = 41.0
LAT_MAX = 49.0
LON_MIN = -131.0   # 229°E
LON_MAX = -120.8   # 239.2°E

# ---------------------------------------------------------------------------
# Channels: (friendly_name, erddap_dataset_id, erddap_variable_name, log_transform)
# ---------------------------------------------------------------------------
CHANNELS = [
    ("chla",  "erdMWchla8day_LonPM180",   "chlorophyll",  True),
    ("k490",  "erdMWk4908day_LonPM180",   "k490",         True),
    ("nflh",  "erdMWcflh8day_LonPM180",   "fluorescence", True),
    ("sst",   "erdMWsstd8day_LonPM180",   "sst",          False),
]
CHANNEL_NAMES = [c[0] for c in CHANNELS]

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"

# ---------------------------------------------------------------------------
# Download settings
# ---------------------------------------------------------------------------
DEFAULT_STRIDE = 2
DEFAULT_WORKERS = 8
DOWNLOAD_START = "2002-07-04"
DOWNLOAD_END = "2025-01-01"

# ---------------------------------------------------------------------------
# Cube / preprocessing
# ---------------------------------------------------------------------------
LOG_CLIP_MIN = 1e-4

# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------
PATCH_SIZE = 64
PATCH_STRIDE = 32
MIN_VALID_FRACTION = 0.5

# ---------------------------------------------------------------------------
# Training (shared 2D + 3D)
# ---------------------------------------------------------------------------
SEED = 42
LATENT_DIM = 32
LATENT_SWEEP = [4, 16, 32, 64, 128]
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
BATCH_SIZE = 64
PATCHES_PER_EPOCH = 10_000
LR = 1e-3

# Coastal patch overlap: fraction of patch cells inside the Overall coastal bbox
# Set to 0.0 to sample anywhere in the cube domain (ablation flag).
TRAIN_COASTAL_PATCH_MIN_OVERLAP = 0.5

# ---------------------------------------------------------------------------
# Temporal (Phase A / B)
# ---------------------------------------------------------------------------
TEMPORAL_WINDOW = 4          # T frames stacked per training sample (anchor + T-1 lookback)
TEMPORAL_STRIDE = 1          # 1 = use every anchor frame; >1 to subsample anchors
# 3D ConvAE convolves time + space; with TEMPORAL_WINDOW=4 and 2 stride-2 time downs,
# the temporal axis collapses to 1 in the bottleneck, matching 2D flat dim.

# ---------------------------------------------------------------------------
# PCA baseline sweep
# ---------------------------------------------------------------------------
PCA_K_SWEEP = [4, 16, 32, 64, 128]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
BOOTSTRAP_ITERS = 1000
TRAIN_FRACTION = 0.80
LASSO_LAGS = [1, 2, 3, 4]

# Event dates for E2 annotation (proposal §7 E2)
EVENTS = {
    "MHW 2014–2016": ("2014-09-01", "2016-04-01"),
    "PN bloom 2015": ("2015-05-01", "2015-09-01"),
    "MHW 2019":       ("2019-05-01", "2019-12-01"),
}
