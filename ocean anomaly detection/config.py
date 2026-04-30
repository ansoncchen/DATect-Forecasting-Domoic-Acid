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
    ("chla",  "erdMWchla8day_LonPM180",  "chlorophyll", True),
    ("k490",  "erdMWk4908day_LonPM180",  "k490",        True),
    ("nflh",  "erdMWcflh8day_LonPM180",  "fluorescence", True),
    ("sst",   "erdMWsstd8day_LonPM180",  "sst",         False),
]
CHANNEL_NAMES = [c[0] for c in CHANNELS]  # ["chla","k490","nflh","sst"]

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"

# ---------------------------------------------------------------------------
# Download settings
# ---------------------------------------------------------------------------
DEFAULT_STRIDE = 2          # 0.025° resolution (proposal §3 fallback)
DEFAULT_WORKERS = 8
DOWNLOAD_START = "2002-07-04"
DOWNLOAD_END = "2025-01-01"

# ---------------------------------------------------------------------------
# Cube / preprocessing
# ---------------------------------------------------------------------------
LOG_CLIP_MIN = 1e-4         # clip before log to avoid log(0)

# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------
PATCH_SIZE = 64
PATCH_STRIDE = 32           # 50% overlap
MIN_VALID_FRACTION = 0.5    # reject training patches below this

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
SEED = 42
LATENT_DIM = 32             # default; sweep overrides this
LATENT_SWEEP = [4, 16, 32, 64, 128]
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
BATCH_SIZE = 64
PATCHES_PER_EPOCH = 10_000
LR = 1e-3

# ---------------------------------------------------------------------------
# PCA baseline sweep
# ---------------------------------------------------------------------------
PCA_K_SWEEP = [4, 16, 32, 64, 128]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
BOOTSTRAP_ITERS = 1000
TRAIN_FRACTION = 0.80       # E4 forecastability split
LASSO_LAGS = [1, 2, 3, 4]

# Event dates for E2 annotation
EVENTS = {
    "MHW 2014–2016": ("2014-09-01", "2016-04-01"),
    "PN bloom 2015": ("2015-05-01", "2015-09-01"),
    "MHW 2019": ("2019-05-01", "2019-12-01"),
}
