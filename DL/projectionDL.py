"""
Morocco Climate Prediction - LSTM multi-horizon forecast 2025-2050
===================================================================

Deep learning counterpart to projectionML.py (XGBoost).
Architecture: 2-layer LSTM + static feature branch + multi-output head.

WHY THIS FILE EXISTS:
    To produce a fair DL comparison against XGBoost on the exact same data.
    Same train/val/test split logic, same output format (GeoTIFFs + summary CSV
    with baselines), so you can diff the two pipelines and compare honestly.

KEY DIFFERENCES FROM XGBOOST VERSION:
    1. Uses 1958-2024 training range (DL benefits more from extra data).
    2. Each sample is (sequence_of_last_15_years, static_summary) instead of
       one flat feature vector.
    3. Single multi-output model predicts all 6 variables jointly, sharing
       representations. This is one of the few places DL has an architectural
       edge over XGBoost on this problem.
    4. All I/O of GeoTIFFs + summary CSV + baselines is identical, so you can
       compare predictions_maroc_ML/ vs predictions_maroc_DL/ one-to-one.

HARDWARE ASSUMED:
    - GTX 1060 3GB or better (CPU fallback automatic)
    - 16 GB RAM
    - Any modern CPU

DO NOT expect this to beat XGBoost on all metrics. It probably won't on short
horizons. Where it should do better: spatial coherence and physically consistent
multi-variable predictions (because all 6 outputs come from one model).
"""

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm import tqdm
import pickle
import warnings
from scipy.stats import linregress
from scipy.ndimage import zoom
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PyTorch - the deep learning framework
# If this import fails: py -m pip install torch --index-url https://download.pytorch.org/whl/cu118
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


# ==================== CENTRALIZED PARAMETERS ====================

class CONFIG:
    """Centralized configuration - modify these to tune model behavior."""

    # ===== DIRECTORIES =====
    # Script lives in <project_root>/DL/, data lives in <project_root>/data/
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "predictions_maroc_DL"
    ELEVATION_FILE = DATA_DIR / "wc2.1_30s_elev.tif"

    # ===== TIME PERIODS =====
    # DL gets more data than XGBoost because it benefits more from it.
    YEARS_ALL = list(range(1958, 2025))       # full data range 1958-2024
    YEARS_TRAIN = list(range(1958, 2015))     # training period (1958-2014)
    YEARS_VAL = list(range(2015, 2018))       # validation period
    YEARS_TEST = list(range(2018, 2025))      # test period
    YEARS_PREDICT = list(range(2025, 2051))   # prediction period

    # ===== VARIABLES =====
    VARIABLES_RAW = ['tmax', 'tmin', 'prec', 'vap', 'ws', 'soil', 'srad', 'aet', 'def']
    VARIABLES = ['tmax', 'tmin', 'prec', 'vap', 'ws', 'def']

    AGGREGATION = {
        'tmax': 'mean', 'tmin': 'mean', 'prec': 'sum', 'vap': 'mean',
        'ws': 'mean', 'soil': 'mean', 'srad': 'mean', 'aet': 'sum', 'def': 'sum'
    }

    # ===== SPATIAL FILTERING =====
    MIN_ANNUAL_PREC = 50
    MAX_ELEVATION = 4000

    # ===== SAMPLING STRATEGY =====
    STRATIFIED_SAMPLING = True
    N_PIXELS_PER_STRATA = 150

    ELEVATION_THRESHOLDS = [500, 1500]
    PRECIPITATION_THRESHOLDS = [200, 400]

    # ===== LSTM ARCHITECTURE =====
    SEQUENCE_LENGTH = 15              # past years fed into the LSTM at each step
    LSTM_HIDDEN_SIZE = 128            # dimensionality of the hidden state
    LSTM_NUM_LAYERS = 2               # stacked LSTM layers
    LSTM_DROPOUT = 0.2                # dropout between LSTM layers
    HEAD_HIDDEN_SIZE = 64             # width of the dense layers after LSTM
    HEAD_DROPOUT = 0.2                # dropout in the dense head

    # ===== TRAINING =====
    BATCH_SIZE = 512                  # samples per gradient step
    LEARNING_RATE = 1e-3              # Adam starting LR (classic default)
    WEIGHT_DECAY = 1e-5               # tiny L2 regularization
    MAX_EPOCHS = 100                  # hard cap - early stopping usually kicks in earlier
    EARLY_STOPPING_PATIENCE = 15      # epochs with no val improvement before stopping
    LR_SCHEDULER_PATIENCE = 5         # epochs with no val improvement before halving LR
    GRAD_CLIP = 1.0                   # clip gradient norm (LSTMs can explode without this)

    # ===== PREDICTION =====
    LOOKBACK_WINDOW = 15              # must equal SEQUENCE_LENGTH for consistency
    MAX_HORIZON = 6                   # model trained on 1-6 year horizons

    # ===== PREDICTION CONSTRAINTS (same as ML) =====
    TMAX_RANGE = (-10, 50)
    TMIN_RANGE = (-20, 40)
    PREC_MIN = 0
    VAP_MIN = 0
    WS_RANGE = (0, 20)
    DEF_MIN = 0

    # ===== MISC =====
    RANDOM_SEED = 42
    VALIDATION_CSV_NAME = "predictions_summary.csv"


# Select device (GPU if available, CPU fallback)
# NOTE: on GTX 1060 3GB, the LSTM+head I've sized will use ~1GB VRAM. Safe.
# If CUDA init fails you'll see a clear error; the fallback to CPU is automatic.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directory
CONFIG.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Set seeds for reproducibility
np.random.seed(CONFIG.RANDOM_SEED)
torch.manual_seed(CONFIG.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG.RANDOM_SEED)

print("=" * 80)
print("MOROCCO CLIMATE PREDICTION - DEEP LEARNING (LSTM)")
print("=" * 80)
print(f"Device             : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print(f"VRAM available     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Data directory     : {CONFIG.DATA_DIR}")
print(f"Output directory   : {CONFIG.OUTPUT_DIR}")
print(f"Training period    : {CONFIG.YEARS_TRAIN[0]}-{CONFIG.YEARS_TRAIN[-1]} ({len(CONFIG.YEARS_TRAIN)} years)")
print(f"Validation period  : {CONFIG.YEARS_VAL[0]}-{CONFIG.YEARS_VAL[-1]} ({len(CONFIG.YEARS_VAL)} years)")
print(f"Test period        : {CONFIG.YEARS_TEST[0]}-{CONFIG.YEARS_TEST[-1]} ({len(CONFIG.YEARS_TEST)} years)")
print(f"Full data range    : {CONFIG.YEARS_ALL[0]}-{CONFIG.YEARS_ALL[-1]} ({len(CONFIG.YEARS_ALL)} years)")
print(f"Sequence length    : {CONFIG.SEQUENCE_LENGTH} years")
print(f"LSTM hidden size   : {CONFIG.LSTM_HIDDEN_SIZE}")
print("=" * 80)


# ==================== DATA LOADING (mirror of ML script) ====================
def load_geotiff(filepath):
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1)
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            return {'data': data, 'bounds': src.bounds,
                    'transform': src.transform, 'crs': src.crs}
    except Exception:
        return None


def get_variable_path(year, variable, month):
    year_dir = CONFIG.DATA_DIR / str(year) / variable
    if not year_dir.exists():
        return None

    pattern = f"morocco_{variable}_{year}_{month:02d}*"
    files = list(year_dir.glob(pattern))
    if files:
        return files[0]

    pattern_fallback = f"*_{month:02d}*"
    files = list(year_dir.glob(pattern_fallback))
    return files[0] if files else None


def load_monthly_data_to_annual(years):
    print(f"\nLoading {years[0]}-{years[-1]}...")

    sample_path = get_variable_path(years[0], 'tmax', 1)
    sample_data = load_geotiff(sample_path)
    shape = sample_data['data'].shape

    n_years = len(years)

    annual_data_raw = np.full((n_years, shape[0], shape[1], len(CONFIG.VARIABLES_RAW)), np.nan)
    monthly_data_full = np.full((n_years, 12, shape[0], shape[1], len(CONFIG.VARIABLES_RAW)), np.nan)

    for year_idx, year in enumerate(tqdm(years, desc="Loading")):
        for month in range(1, 13):
            for var_idx, var in enumerate(CONFIG.VARIABLES_RAW):
                path = get_variable_path(year, var, month)
                if path:
                    var_data = load_geotiff(path)
                    if var_data:
                        monthly_data_full[year_idx, month - 1, :, :, var_idx] = var_data['data']

    for var_idx, var in enumerate(CONFIG.VARIABLES_RAW):
        if CONFIG.AGGREGATION[var] == 'sum':
            annual_data_raw[:, :, :, var_idx] = np.nansum(monthly_data_full[:, :, :, :, var_idx], axis=1)
        else:
            annual_data_raw[:, :, :, var_idx] = np.nanmean(monthly_data_full[:, :, :, :, var_idx], axis=1)

    annual_data = np.full((n_years, shape[0], shape[1], len(CONFIG.VARIABLES)), np.nan)
    for var_idx, var in enumerate(CONFIG.VARIABLES):
        raw_idx = CONFIG.VARIABLES_RAW.index(var)
        annual_data[:, :, :, var_idx] = annual_data_raw[:, :, :, raw_idx]

    prec_mean = np.nanmean(annual_data[:, :, :, CONFIG.VARIABLES.index('prec')], axis=0)
    land_mask_basic = ~np.isnan(prec_mean)

    print(f"Shape: {annual_data.shape}, Variables: {CONFIG.VARIABLES}")

    return {
        'data': annual_data,
        'land_mask_basic': land_mask_basic,
        'bounds': sample_data['bounds'],
        'transform': sample_data['transform'],
        'crs': sample_data['crs'],
        'shape': shape,
        'monthly_data': monthly_data_full
    }


print("\nLoading elevation...")
elevation_data = load_geotiff(CONFIG.ELEVATION_FILE)
elevation_original = elevation_data['data']

all_data = load_monthly_data_to_annual(CONFIG.YEARS_ALL)

print("\nResizing elevation...")
zoom_factors = (all_data['shape'][0] / elevation_original.shape[0],
                all_data['shape'][1] / elevation_original.shape[1])
elevation = zoom(elevation_original, zoom_factors, order=1)

print("\nCreating strict land mask...")
prec_mean_historical = np.nanmean(all_data['data'][:, :, :, CONFIG.VARIABLES.index('prec')], axis=0)

land_mask = (
    all_data['land_mask_basic'] &
    (prec_mean_historical >= CONFIG.MIN_ANNUAL_PREC) &
    (elevation < CONFIG.MAX_ELEVATION) &
    (elevation >= -10)
)

print(f"   Pixels: {np.sum(land_mask):,}")
all_data['land_mask'] = land_mask


# ==================== STRATIFIED SAMPLING (mirror of ML script) ====================
if CONFIG.STRATIFIED_SAMPLING:
    print("\nStratified sampling by climate zone...")

    land_pixels = np.argwhere(all_data['land_mask'])
    strates = []
    for i, j in land_pixels:
        elev_val = elevation[i, j]
        prec_val = prec_mean_historical[i, j]

        if elev_val < CONFIG.ELEVATION_THRESHOLDS[0]:
            elev_zone = 'plain'
        elif elev_val < CONFIG.ELEVATION_THRESHOLDS[1]:
            elev_zone = 'hill'
        else:
            elev_zone = 'mountain'

        if prec_val < CONFIG.PRECIPITATION_THRESHOLDS[0]:
            prec_zone = 'arid'
        elif prec_val < CONFIG.PRECIPITATION_THRESHOLDS[1]:
            prec_zone = 'semi_arid'
        else:
            prec_zone = 'humid'

        strates.append(f"{elev_zone}_{prec_zone}")

    strates = np.array(strates)
    unique_strates = np.unique(strates)

    print(f"   Found {len(unique_strates)} climate strata:")
    for strate in unique_strates:
        count = np.sum(strates == strate)
        print(f"      {strate}: {count:,} pixels")

    sampled_indices = []
    for strate in unique_strates:
        strate_indices = np.where(strates == strate)[0]
        n_sample = min(CONFIG.N_PIXELS_PER_STRATA, len(strate_indices))
        sampled = np.random.choice(strate_indices, n_sample, replace=False)
        sampled_indices.extend(sampled)

    land_pixels_sampled = land_pixels[sampled_indices]
    print(f"\nSampled {len(land_pixels_sampled):,} pixels (stratified)")
else:
    land_pixels = np.argwhere(all_data['land_mask'])
    n_sample = min(3000, len(land_pixels))
    sampled_indices = np.random.choice(len(land_pixels), n_sample, replace=False)
    land_pixels_sampled = land_pixels[sampled_indices]
    print(f"\nSampled {len(land_pixels_sampled):,} pixels (random)")


# ==================== FEATURE COMPUTATION FOR STATIC BRANCH ====================
# For the LSTM, the sequence input is raw values (year-by-year, 6 variables).
# The STATIC input is a small set of engineered summary features that help with
# long-horizon extrapolation - specifically the trend and climatology features.
# We keep the static branch intentionally small so the LSTM is forced to actually
# learn temporal dynamics from the raw sequence.
def calculate_trend(series, window=10):
    if len(series) < window or np.all(np.isnan(series[-window:])):
        return 0.0
    recent = series[-window:]
    valid_mask = ~np.isnan(recent)
    if np.sum(valid_mask) < 3:
        return 0.0
    x = np.arange(len(recent))[valid_mask]
    y = recent[valid_mask]
    if len(x) < 2:
        return 0.0
    slope, _, _, _, _ = linregress(x, y)
    return slope


def compute_static_features(pixel_data, year_idx, elevation_val, lat, lon, horizon):
    """
    Compute the static summary features for a single (pixel, year_idx, horizon) sample.

    Returns a fixed-length vector:
        [elevation, lat, lon, horizon,
         trend_10y for each of 6 variables,
         trend_accel for each of 6 variables,
         mean_10y for each of 6 variables]
    Total length: 4 + 6 + 6 + 6 = 22

    NOTE: lag features and full seasonal stats are NOT included here on purpose -
    the LSTM should learn those from the raw sequence. We only inject features
    that are hard to learn from a 15-year window: long-term trends and their
    acceleration. Without these, the LSTM would struggle to extrapolate past 2024.
    """
    feats = [elevation_val, lat, lon, float(horizon)]

    # Per-variable trend and climatology
    for var_idx in range(len(CONFIG.VARIABLES)):
        history = pixel_data[:year_idx, var_idx]

        # Trend over last 10 years
        trend_recent = calculate_trend(history[-10:], 10) if year_idx >= 10 else 0.0

        # Trend acceleration (recent 10y minus previous 10y)
        if year_idx >= 20:
            trend_old = calculate_trend(history[-20:-10], 10)
            trend_accel = trend_recent - trend_old
        else:
            trend_accel = 0.0

        # 10-year climatology mean
        if year_idx >= 10:
            mean_10y = float(np.nanmean(history[-10:]))
        else:
            mean_10y = 0.0

        feats.extend([trend_recent, trend_accel, mean_10y])

    return np.array(feats, dtype=np.float32)


# Number of static features (useful later for defining model input size)
N_STATIC_FEATURES = 4 + 3 * len(CONFIG.VARIABLES)
print(f"\nStatic feature vector length: {N_STATIC_FEATURES}")


# ==================== BUILD SEQUENCE DATASET ====================
# Each sample is a tuple: (sequence, static, target, year, horizon).
#   sequence: shape (SEQUENCE_LENGTH, n_vars) = (15, 6) raw annual values
#   static:   shape (N_STATIC_FEATURES,) = (22,)
#   target:   shape (n_vars,) = (6,)
print("\nBuilding sequence dataset...")

lat_grid = np.linspace(all_data['bounds'].top, all_data['bounds'].bottom, all_data['shape'][0])
lon_grid = np.linspace(all_data['bounds'].left, all_data['bounds'].right, all_data['shape'][1])

sequences_list = []
statics_list = []
targets_list = []
years_list = []
horizons_list = []

seq_len = CONFIG.SEQUENCE_LENGTH
n_years = len(CONFIG.YEARS_ALL)
n_vars = len(CONFIG.VARIABLES)

for pixel_idx in tqdm(range(len(land_pixels_sampled)), desc="Building"):
    i, j = land_pixels_sampled[pixel_idx]
    pixel_data = all_data['data'][:, i, j, :]  # shape (n_years, n_vars)

    elev_val = float(elevation[i, j])
    lat_val = float(lat_grid[i])
    lon_val = float(lon_grid[j])

    # Skip if pixel has too many NaNs
    if np.sum(np.isnan(pixel_data[:, 0])) > n_years * 0.1:
        continue

    # Iterate over all valid (year_idx, horizon) combinations for this pixel.
    # NOTE: mirrors XGBoost's indexing convention. year_idx is "current", the
    # sequence uses [year_idx - seq_len, year_idx - 1], target is year_idx + horizon.
    for year_idx in range(seq_len, n_years - CONFIG.MAX_HORIZON):
        sequence = pixel_data[year_idx - seq_len:year_idx, :]  # (15, 6)

        # Skip samples with NaN in sequence (rare but possible at data edges)
        if np.any(np.isnan(sequence)):
            continue

        for horizon in range(1, CONFIG.MAX_HORIZON + 1):
            target_idx = year_idx + horizon
            if target_idx >= n_years:
                continue
            target = pixel_data[target_idx, :]  # (6,)

            if np.any(np.isnan(target)):
                continue

            static = compute_static_features(
                pixel_data, year_idx, elev_val, lat_val, lon_val, horizon
            )

            sequences_list.append(sequence.astype(np.float32))
            statics_list.append(static)
            targets_list.append(target.astype(np.float32))
            years_list.append(CONFIG.YEARS_ALL[year_idx])
            horizons_list.append(horizon)

# Stack into tensors
sequences_arr = np.stack(sequences_list)      # (N, 15, 6)
statics_arr = np.stack(statics_list)          # (N, 22)
targets_arr = np.stack(targets_list)          # (N, 6)
years_arr = np.array(years_list)
horizons_arr = np.array(horizons_list)

print(f"Dataset: {len(sequences_arr):,} samples")
print(f"   Sequence shape: {sequences_arr.shape}")
print(f"   Static shape:   {statics_arr.shape}")
print(f"   Target shape:   {targets_arr.shape}")


# ==================== TRAIN/VAL/TEST SPLIT ====================
# Split by year to avoid temporal leakage.
train_mask = np.isin(years_arr, CONFIG.YEARS_TRAIN)
val_mask = np.isin(years_arr, CONFIG.YEARS_VAL)
test_mask = np.isin(years_arr, CONFIG.YEARS_TEST)

print(f"\n   Train: {np.sum(train_mask):,}, Val: {np.sum(val_mask):,}, Test: {np.sum(test_mask):,}")


# ==================== NORMALIZATION ====================
# LSTMs train vastly better when inputs are roughly standardized (mean=0, std=1).
# We fit scalers ONLY on training data to avoid leaking test statistics.
# Scalers are saved with the model and used at inference time to transform
# new inputs and un-transform predictions back to physical units.
print("\nComputing normalization statistics (training data only)...")

# Sequence: per-variable stats (6 vars)
seq_mean = sequences_arr[train_mask].reshape(-1, n_vars).mean(axis=0)  # (6,)
seq_std = sequences_arr[train_mask].reshape(-1, n_vars).std(axis=0) + 1e-6  # (6,)

# Static: per-feature stats (22 features)
stat_mean = statics_arr[train_mask].mean(axis=0)  # (22,)
stat_std = statics_arr[train_mask].std(axis=0) + 1e-6  # (22,)

# Target: per-variable stats (6 vars)
tgt_mean = targets_arr[train_mask].mean(axis=0)  # (6,)
tgt_std = targets_arr[train_mask].std(axis=0) + 1e-6  # (6,)


def normalize_seq(x):
    return (x - seq_mean) / seq_std


def normalize_stat(x):
    return (x - stat_mean) / stat_std


def normalize_tgt(x):
    return (x - tgt_mean) / tgt_std


def denormalize_tgt(x):
    return x * tgt_std + tgt_mean


print(f"   Target means  : {tgt_mean}")
print(f"   Target stds   : {tgt_std}")


# ==================== PYTORCH DATASET ====================
# PyTorch wants training data wrapped in a Dataset class that implements
# __len__ and __getitem__. The DataLoader then batches + shuffles + moves
# things to GPU in parallel.
class ClimateSequenceDataset(Dataset):
    """Wraps (sequence, static, target) arrays for one split (train/val/test)."""

    def __init__(self, sequences, statics, targets):
        # Normalize here, convert to torch tensors once.
        self.sequences = torch.tensor(normalize_seq(sequences), dtype=torch.float32)
        self.statics = torch.tensor(normalize_stat(statics), dtype=torch.float32)
        self.targets = torch.tensor(normalize_tgt(targets), dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.statics[idx], self.targets[idx]


train_dataset = ClimateSequenceDataset(
    sequences_arr[train_mask], statics_arr[train_mask], targets_arr[train_mask]
)
val_dataset = ClimateSequenceDataset(
    sequences_arr[val_mask], statics_arr[val_mask], targets_arr[val_mask]
)
test_dataset = ClimateSequenceDataset(
    sequences_arr[test_mask], statics_arr[test_mask], targets_arr[test_mask]
)

# DataLoader: batches and shuffles. num_workers=0 avoids Windows multiprocessing quirks.
train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=0)


# ==================== LSTM MODEL ====================
# Architecture overview:
#
#   sequence (batch, 15, 6)  -->  LSTM (2 layers, 128 hidden)  -->  last_hidden (batch, 128)
#                                                                        |
#                                                                        v
#                                                            concat with static (batch, 22)
#                                                                        |
#                                                                        v
#                                                       Dense(128+22 -> 64) + ReLU + Dropout
#                                                                        |
#                                                                        v
#                                                       Dense(64 -> 64) + ReLU + Dropout
#                                                                        |
#                                                                        v
#                                                       Dense(64 -> 6)  <-- predicts 6 vars
#
# WHY THIS SHAPE:
# - LSTM extracts temporal features from the raw 15-year sequence
# - Static branch injects hard-to-learn long-term signals (trend, climatology)
# - Shared dense head produces all 6 outputs jointly (enforces soft consistency)
# - ~200k parameters total, comfortably under GPU limits
class LSTMClimateModel(nn.Module):
    def __init__(self, n_vars, n_static, hidden_size=128, num_layers=2,
                 head_hidden=64, lstm_dropout=0.2, head_dropout=0.2):
        super().__init__()

        # The LSTM layer. batch_first=True means input shape is (batch, seq, features)
        # which is more intuitive than the PyTorch default of (seq, batch, features).
        self.lstm = nn.LSTM(
            input_size=n_vars,        # 6 (tmax, tmin, prec, vap, ws, def)
            hidden_size=hidden_size,  # 128
            num_layers=num_layers,    # 2
            dropout=lstm_dropout,     # dropout between layers
            batch_first=True,
        )

        # After LSTM we have (batch, 128). We concat with static (batch, 22).
        # The dense head maps (128 + 22) -> 6 outputs.
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_static, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, n_vars),
        )

    def forward(self, seq, static):
        # seq shape: (batch, 15, 6). static shape: (batch, 22).
        lstm_out, (h_n, c_n) = self.lstm(seq)
        # lstm_out shape: (batch, 15, 128) - output at every timestep
        # h_n shape:      (num_layers, batch, 128) - last hidden state per layer

        # We use the last hidden state of the top layer as the sequence summary.
        # Equivalent to lstm_out[:, -1, :].
        seq_summary = h_n[-1]  # (batch, 128)

        # Fuse sequence summary with static features
        fused = torch.cat([seq_summary, static], dim=1)  # (batch, 128 + 22)

        return self.head(fused)  # (batch, 6)


model = LSTMClimateModel(
    n_vars=n_vars,
    n_static=N_STATIC_FEATURES,
    hidden_size=CONFIG.LSTM_HIDDEN_SIZE,
    num_layers=CONFIG.LSTM_NUM_LAYERS,
    head_hidden=CONFIG.HEAD_HIDDEN_SIZE,
    lstm_dropout=CONFIG.LSTM_DROPOUT,
    head_dropout=CONFIG.HEAD_DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")


# ==================== TRAINING LOOP ====================
# The canonical PyTorch training loop:
#   1. Set model to train mode
#   2. For each batch:
#      a. Move tensors to GPU
#      b. Forward pass (model predicts)
#      c. Compute loss (MSE)
#      d. Zero gradients, backprop, clip gradients, optimizer step
#   3. Set model to eval mode, compute validation loss
#   4. Check early stopping and LR scheduler
criterion = nn.MSELoss()  # plain MSE on normalized targets
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=CONFIG.LEARNING_RATE,
    weight_decay=CONFIG.WEIGHT_DECAY,
)
# Reduce LR by half when val loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5,
    patience=CONFIG.LR_SCHEDULER_PATIENCE,
)


def run_epoch(loader, training=True):
    """Run one epoch. If training=True, also does backprop."""
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    # Disable gradient computation during eval to save memory and time
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for seq, static, target in loader:
            seq = seq.to(DEVICE, non_blocking=True)
            static = static.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            pred = model(seq, static)
            loss = criterion(pred, target)

            if training:
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding gradients (important for LSTMs)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.GRAD_CLIP)
                optimizer.step()

            batch_size = seq.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_val_loss = float('inf')
best_epoch = 0
epochs_without_improvement = 0
training_log = []

for epoch in range(1, CONFIG.MAX_EPOCHS + 1):
    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    training_log.append({
        'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': current_lr
    })

    improved = val_loss < best_val_loss
    marker = " *" if improved else ""
    print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {current_lr:.1e}{marker}")

    if improved:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0
        # Save best weights
        torch.save(model.state_dict(), CONFIG.OUTPUT_DIR / 'best_model.pt')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= CONFIG.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}. Best was epoch {best_epoch}.")
            break

# Reload best weights for evaluation and prediction
print(f"\nReloading best weights from epoch {best_epoch}")
model.load_state_dict(torch.load(CONFIG.OUTPUT_DIR / 'best_model.pt'))
model.eval()


# ==================== TEST SET EVALUATION + BASELINES ====================
# Same metric framework as the ML script: XGBoost MAE/R2 + persistence +
# climatology for each variable. Here instead of XGBoost we evaluate the LSTM.
print("\n" + "=" * 80)
print("TEST SET EVALUATION")
print("=" * 80)

# Gather all test predictions in one pass (in normalized space)
model.eval()
test_preds_norm = []
test_targets_norm = []
with torch.no_grad():
    for seq, static, target in test_loader:
        seq = seq.to(DEVICE)
        static = static.to(DEVICE)
        pred = model(seq, static)
        test_preds_norm.append(pred.cpu().numpy())
        test_targets_norm.append(target.numpy())

test_preds_norm = np.concatenate(test_preds_norm, axis=0)
test_targets_norm = np.concatenate(test_targets_norm, axis=0)

# De-normalize to physical units for reporting
test_preds = denormalize_tgt(test_preds_norm)
test_targets = denormalize_tgt(test_targets_norm)

# Baselines computed from the test subset of sequences/statics directly
# Persistence: target = last value in sequence = sequences[:, -1, :]
# Climatology: target = mean_10y feature = statics[:, 4 + 2 + var*3 : ...]
# Actually mean_10y is stored as the 3rd feature per variable in the static vector:
#   static = [elev, lat, lon, horizon, trend_0, accel_0, mean10y_0, trend_1, accel_1, mean10y_1, ...]
# Index of mean_10y for variable v: 4 + v*3 + 2 = 6 + 3*v
test_sequences = sequences_arr[test_mask]
test_statics = statics_arr[test_mask]

persistence_preds = test_sequences[:, -1, :]  # (N_test, 6)
climatology_preds = np.stack([test_statics[:, 6 + 3 * v] for v in range(n_vars)], axis=1)  # (N_test, 6)

# Per-variable metrics
test_metrics = {}
for var_idx, var in enumerate(CONFIG.VARIABLES):
    y_true = test_targets[:, var_idx]
    y_lstm = test_preds[:, var_idx]
    y_pers = persistence_preds[:, var_idx]
    y_clim = climatology_preds[:, var_idx]

    mae_lstm = mean_absolute_error(y_true, y_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_true, y_lstm))
    r2_lstm = r2_score(y_true, y_lstm)

    mae_pers = mean_absolute_error(y_true, y_pers)
    r2_pers = r2_score(y_true, y_pers)

    mae_clim = mean_absolute_error(y_true, y_clim)
    r2_clim = r2_score(y_true, y_clim)

    skill_pers = 1 - mae_lstm / mae_pers if mae_pers > 0 else np.nan
    skill_clim = 1 - mae_lstm / mae_clim if mae_clim > 0 else np.nan

    test_metrics[var] = {
        'mae_lstm': mae_lstm, 'rmse_lstm': rmse_lstm, 'r2_lstm': r2_lstm,
        'mae_persistence': mae_pers, 'r2_persistence': r2_pers,
        'mae_climatology': mae_clim, 'r2_climatology': r2_clim,
        'skill_vs_persistence': skill_pers, 'skill_vs_climatology': skill_clim,
    }

    print(f"\n{var}:")
    print(f"   LSTM        -> MAE: {mae_lstm:.3f}  RMSE: {rmse_lstm:.3f}  R2: {r2_lstm:.3f}")
    print(f"   Persistence -> MAE: {mae_pers:.3f}  R2: {r2_pers:.3f}  (skill: {skill_pers:+.3f})")
    print(f"   Climatology -> MAE: {mae_clim:.3f}  R2: {r2_clim:.3f}  (skill: {skill_clim:+.3f})")

# Per-horizon breakdown
print("\n" + "-" * 80)
print("Per-horizon test MAE (LSTM only)")
print("-" * 80)
test_horizons = horizons_arr[test_mask]
for horizon in [1, 3, 6]:
    mask = test_horizons == horizon
    if not np.any(mask):
        continue
    print(f"\n   Horizon {horizon} year(s):")
    for var_idx, var in enumerate(CONFIG.VARIABLES):
        mae_h = mean_absolute_error(test_targets[mask, var_idx], test_preds[mask, var_idx])
        r2_h = r2_score(test_targets[mask, var_idx], test_preds[mask, var_idx])
        print(f"      {var}: MAE={mae_h:.2f}, R2={r2_h:.3f}")


# ==================== RECURSIVE PREDICTION 2025-2050 ====================
print("\n" + "=" * 80)
print("RECURSIVE PREDICTION 2025-2050")
print("=" * 80)


def predict_multiyear_recursive_lstm(model, pixel_data, elevation_val, lat, lon,
                                     n_years_ahead, seq_len):
    """
    Roll the model forward year by year. Each year's prediction becomes part of
    the history for the next year's input sequence.

    Mirrors the structure of the XGBoost recursive predictor.
    """
    predictions = {var: [] for var in CONFIG.VARIABLES}
    extended_data = np.copy(pixel_data)  # will grow with each predicted year

    model.eval()
    with torch.no_grad():
        for year_offset in range(n_years_ahead):
            current_year_idx = len(extended_data)  # position where prediction will land

            # Horizon is 1 for all recursive steps (always predicting "next year")
            horizon = 1

            # Build sequence: last seq_len years
            sequence = extended_data[-seq_len:, :]  # (15, 6)
            if np.any(np.isnan(sequence)):
                # If there are NaNs (shouldn't happen for land pixels), output NaN
                for var in CONFIG.VARIABLES:
                    predictions[var].append(np.nan)
                continue

            # Build static features at current_year_idx using extended_data
            static = compute_static_features(
                extended_data, current_year_idx, elevation_val, lat, lon, horizon
            )

            # Normalize and to tensor
            seq_norm = torch.tensor(normalize_seq(sequence), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            stat_norm = torch.tensor(normalize_stat(static), dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Forward pass
            pred_norm = model(seq_norm, stat_norm).cpu().numpy()[0]  # (6,)
            pred_phys = denormalize_tgt(pred_norm)                    # physical units

            # Apply physical constraints (same as ML script)
            year_predictions = {}
            for var_idx, var in enumerate(CONFIG.VARIABLES):
                p = float(pred_phys[var_idx])
                if var == 'tmax':
                    p = np.clip(p, *CONFIG.TMAX_RANGE)
                elif var == 'tmin':
                    p = np.clip(p, *CONFIG.TMIN_RANGE)
                elif var == 'prec':
                    p = max(CONFIG.PREC_MIN, p)
                elif var == 'vap':
                    p = max(CONFIG.VAP_MIN, p)
                elif var == 'ws':
                    p = np.clip(p, *CONFIG.WS_RANGE)
                elif var == 'def':
                    p = max(CONFIG.DEF_MIN, p)
                year_predictions[var] = p
                predictions[var].append(p)

            # Append predicted year to extended data for next iteration
            new_row = np.array([year_predictions[var] for var in CONFIG.VARIABLES],
                               dtype=np.float32)
            extended_data = np.vstack([extended_data, new_row[None, :]])

    return predictions


# Predict for all land pixels
n_future_years = len(CONFIG.YEARS_PREDICT)
shape = all_data['shape']

predictions_all = {
    var: np.full((n_future_years, shape[0], shape[1]), np.nan, dtype=np.float32)
    for var in CONFIG.VARIABLES
}

land_pixels_all = np.argwhere(all_data['land_mask'])

for pixel_idx in tqdm(range(len(land_pixels_all)), desc="Predicting"):
    i, j = land_pixels_all[pixel_idx]

    pixel_data = all_data['data'][:, i, j, :]  # (n_years, 6)

    # Skip pixels with too many NaNs in history
    if np.sum(np.isnan(pixel_data[:, 0])) > n_years * 0.1:
        continue

    elev_val = float(elevation[i, j])
    lat_val = float(lat_grid[i])
    lon_val = float(lon_grid[j])

    preds = predict_multiyear_recursive_lstm(
        model, pixel_data, elev_val, lat_val, lon_val,
        n_years_ahead=n_future_years, seq_len=seq_len
    )

    for var in CONFIG.VARIABLES:
        for future_idx in range(n_future_years):
            predictions_all[var][future_idx, i, j] = preds[var][future_idx]

print("Predictions complete.")


# ==================== MONTHLY DISAGGREGATION (same as ML) ====================
print("\nMonthly disaggregation...")

clim_start_idx = max(0, 1991 - CONFIG.YEARS_ALL[0])
clim_end_idx = min(len(CONFIG.YEARS_ALL), 2020 - CONFIG.YEARS_ALL[0] + 1)

monthly_patterns = {}
for var in ['tmax', 'tmin', 'prec']:
    var_idx = CONFIG.VARIABLES_RAW.index(var)
    monthly_clim = np.nanmean(
        all_data['monthly_data'][clim_start_idx:clim_end_idx, :, :, :, var_idx],
        axis=0
    )
    annual_clim = np.nansum(monthly_clim, axis=0) if var == 'prec' else np.nanmean(monthly_clim, axis=0)
    pattern = np.zeros_like(monthly_clim)
    for month in range(12):
        with np.errstate(divide='ignore', invalid='ignore'):
            if var == 'prec':
                pattern[month] = monthly_clim[month] / annual_clim
            else:
                pattern[month] = monthly_clim[month] - annual_clim
    monthly_patterns[var] = pattern

predictions_monthly = {}
for var in ['tmax', 'tmin', 'prec']:
    predictions_monthly[var] = np.full(
        (n_future_years, 12, shape[0], shape[1]), np.nan, dtype=np.float32
    )
    for future_idx in range(n_future_years):
        annual_pred = predictions_all[var][future_idx]
        for month in range(12):
            if var == 'prec':
                predictions_monthly[var][future_idx, month] = (
                    annual_pred * monthly_patterns[var][month]
                )
            else:
                predictions_monthly[var][future_idx, month] = (
                    annual_pred + monthly_patterns[var][month]
                )

print("Monthly disaggregation complete.")


# ==================== EXPORT GEOTIFFS (same as ML) ====================
print("\nExporting GeoTIFFs...")


def save_geotiff(data, year, variable, metadata):
    year_dir = CONFIG.OUTPUT_DIR / str(year) / variable
    year_dir.mkdir(parents=True, exist_ok=True)
    output_path = year_dir / f"morocco_{variable}_{year}.tif"
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=metadata['shape'][0], width=metadata['shape'][1], count=1,
        dtype=data.dtype, crs=metadata['crs'], transform=metadata['transform'],
        nodata=np.nan
    ) as dst:
        dst.write(data, 1)


def save_monthly_geotiff(data, year, month, variable, metadata):
    year_dir = CONFIG.OUTPUT_DIR / str(year) / variable
    year_dir.mkdir(parents=True, exist_ok=True)
    output_path = year_dir / f"morocco_{variable}_{year}_{month:02d}.tif"
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=metadata['shape'][0], width=metadata['shape'][1], count=1,
        dtype=data.dtype, crs=metadata['crs'], transform=metadata['transform'],
        nodata=np.nan
    ) as dst:
        dst.write(data, 1)


for future_idx, year in enumerate(tqdm(CONFIG.YEARS_PREDICT, desc="Exporting")):
    for var in CONFIG.VARIABLES:
        save_geotiff(predictions_all[var][future_idx], year, var, all_data)
    tavg = (predictions_all['tmax'][future_idx] + predictions_all['tmin'][future_idx]) / 2
    save_geotiff(tavg, year, 'tavg', all_data)
    for month in range(1, 13):
        for var in ['tmax', 'tmin', 'prec']:
            save_monthly_geotiff(
                predictions_monthly[var][future_idx, month - 1],
                year, month, var, all_data
            )


# ==================== PREDICTIONS SUMMARY CSV (same structure as ML) ====================
print("\nBuilding predictions summary CSV...")

lat_grid_2d = np.broadcast_to(lat_grid[:, None], shape)
lon_grid_2d = np.broadcast_to(lon_grid[None, :], shape)

regions = {
    'national':       land_mask,
    'north':          land_mask & (lat_grid_2d > 33),
    'center':         land_mask & (lat_grid_2d >= 31) & (lat_grid_2d <= 33),
    'south':          land_mask & (lat_grid_2d < 31),
    'atlantic_coast': land_mask & (lon_grid_2d < -8) & (lat_grid_2d > 28),
    'mountains':      land_mask & (elevation > 1500),
    'plains':         land_mask & (elevation < 500),
    'sahara':         land_mask & (elevation < 500) & (lat_grid_2d < 30),
}

print("   Region pixel counts:")
for r, m in regions.items():
    print(f"      {r}: {int(np.sum(m)):,}")

# Climatology from training years only (1958-2014)
clim_years_end = len(CONFIG.YEARS_TRAIN)  # exclusive
climatology_fields = {}
last_observed_fields = {}
for var in CONFIG.VARIABLES:
    var_idx = CONFIG.VARIABLES.index(var)
    climatology_fields[var] = np.nanmean(
        all_data['data'][:clim_years_end, :, :, var_idx], axis=0
    )
    last_observed_fields[var] = all_data['data'][-1, :, :, var_idx]

summary_rows = []
for future_idx, year in enumerate(CONFIG.YEARS_PREDICT):
    for var in CONFIG.VARIABLES:
        lstm_field = predictions_all[var][future_idx]
        clim_field = climatology_fields[var]
        pers_field = last_observed_fields[var]

        for region_name, mask in regions.items():
            if not np.any(mask):
                continue
            summary_rows.append({
                'year': year,
                'variable': var,
                'region': region_name,
                'n_pixels': int(np.sum(mask)),
                'lstm_pred': float(np.nanmean(lstm_field[mask])),
                'persistence_2024': float(np.nanmean(pers_field[mask])),
                'climatology_1958_2014': float(np.nanmean(clim_field[mask])),
                'observed': np.nan,
            })

summary_df = pd.DataFrame(summary_rows)
summary_path = CONFIG.OUTPUT_DIR / CONFIG.VALIDATION_CSV_NAME
summary_df.to_csv(summary_path, index=False)
print(f"   Summary CSV written: {summary_path}")
print(f"   Rows: {len(summary_df):,}")

# Test metrics CSV
metrics_rows = []
for var, m in test_metrics.items():
    metrics_rows.append({'variable': var, **m})
metrics_df = pd.DataFrame(metrics_rows)
metrics_path = CONFIG.OUTPUT_DIR / "test_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"   Test metrics CSV written: {metrics_path}")

# Training log CSV (loss curves)
training_log_df = pd.DataFrame(training_log)
training_log_df.to_csv(CONFIG.OUTPUT_DIR / "training_log.csv", index=False)

# Save scalers + model together for later reuse
with open(CONFIG.OUTPUT_DIR / 'model_bundle.pkl', 'wb') as f:
    pickle.dump({
        'seq_mean': seq_mean, 'seq_std': seq_std,
        'stat_mean': stat_mean, 'stat_std': stat_std,
        'tgt_mean': tgt_mean, 'tgt_std': tgt_std,
        'config': {k: v for k, v in CONFIG.__dict__.items() if not k.startswith('_')},
        'n_static_features': N_STATIC_FEATURES,
        'test_metrics': test_metrics,
        'best_epoch': best_epoch,
    }, f)

print("\n" + "=" * 80)
print("DEEP LEARNING PIPELINE COMPLETE")
print("=" * 80)
print(f"\nConfiguration used:")
print(f"   Sequence length  : {CONFIG.SEQUENCE_LENGTH}")
print(f"   LSTM layers      : {CONFIG.LSTM_NUM_LAYERS}")
print(f"   LSTM hidden size : {CONFIG.LSTM_HIDDEN_SIZE}")
print(f"   Total parameters : {n_params:,}")
print(f"   Best val epoch   : {best_epoch}")
print(f"   Training range   : {CONFIG.YEARS_TRAIN[0]}-{CONFIG.YEARS_TRAIN[-1]}")
print(f"\nOutput directory: {CONFIG.OUTPUT_DIR}")
print("=" * 80)
