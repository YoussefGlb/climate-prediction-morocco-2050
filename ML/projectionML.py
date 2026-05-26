"""
Morocco Climate Prediction - XGBoost multi-horizon forecast 2025-2050
Trained on TerraClimate 1981-2024 + elevation.
Outputs: annual + monthly GeoTIFFs per variable, plus a validation CSV with baselines.
"""

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import pickle
import warnings
from scipy.stats import linregress
from scipy.ndimage import zoom
warnings.filterwarnings('ignore')


# ==================== CENTRALIZED PARAMETERS ====================

class CONFIG:
    """Centralized configuration - modify these to tune model performance."""

    # ===== DIRECTORIES =====
    # Script lives in <project_root>/ML/, data lives in <project_root>/data/
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "predictions_maroc_ML_improvedV3entire"
    ELEVATION_FILE = DATA_DIR / "wc2.1_30s_elev.tif"

    # ===== PREDICTION SETTINGS =====
    LOOKBACK_WINDOW = 15  # years of history used for each recursive prediction step

    # ===== TIME PERIODS =====
    YEARS_TRAIN = list(range(1981, 2025))   # training period 
    YEARS_VAL = list(range(2015, 2018))     # validation period
    YEARS_TEST = list(range(2018, 2025))    # test period
    YEARS_ALL = list(range(1981, 2025))
    YEARS_PREDICT = list(range(2025, 2051)) # 2025-2050

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

    # Stratification thresholds
    ELEVATION_THRESHOLDS = [500, 1500]
    PRECIPITATION_THRESHOLDS = [200, 400]

    # ===== XGBOOST HYPERPARAMETERS (OPTIMIZED FOR ACCURACY) =====
    XGBOOST_PARAMS = {
        'prec': {
            'n_estimators': 400,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_alpha': 0.8,
            'reg_lambda': 1.5,
            'random_state': 42,
            'n_jobs': -1
        },
        'tmax': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 2,
            'gamma': 0.05,
            'reg_alpha': 0.2,
            'reg_lambda': 0.8,
            'random_state': 42,
            'n_jobs': -1
        },
        'tmin': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 2,
            'gamma': 0.05,
            'reg_alpha': 0.2,
            'reg_lambda': 0.8,
            'random_state': 42,
            'n_jobs': -1
        },
        'default': {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 3,
            'gamma': 0.05,
            'reg_alpha': 0.3,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
    }

    # ===== TRAINING SETTINGS =====
    EARLY_STOPPING_ROUNDS = 80
    VERBOSE_TRAINING = False

    # ===== FEATURE ENGINEERING =====
    N_LAG_FEATURES = 5
    WINDOW_5Y_STATS = True
    WINDOW_10Y_STATS = True
    SEASONAL_FEATURES = True
    TREND_FEATURES = True

    # ===== PREDICTION CONSTRAINTS =====
    TMAX_RANGE = (-10, 50)
    TMIN_RANGE = (-20, 40)
    PREC_MIN = 0
    VAP_MIN = 0
    WS_RANGE = (0, 20)
    DEF_MIN = 0

    # ===== VALIDATION CSV =====
    # Regions used for the structured prediction dump
    # (definitions applied later once lat/lon grids exist)
    VALIDATION_CSV_NAME = "predictions_summary.csv"


# Create output directory
CONFIG.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("MOROCCO CLIMATE PREDICTION - Optimized Parameters")
print("=" * 80)
print(f"Data directory     : {CONFIG.DATA_DIR}")
print(f"Output directory   : {CONFIG.OUTPUT_DIR}")
print(f"Training period    : {CONFIG.YEARS_TRAIN[0]}-{CONFIG.YEARS_TRAIN[-1]} ({len(CONFIG.YEARS_TRAIN)} years)")
print(f"Validation period  : {CONFIG.YEARS_VAL[0]}-{CONFIG.YEARS_VAL[-1]} ({len(CONFIG.YEARS_VAL)} years)")
print(f"Test period        : {CONFIG.YEARS_TEST[0]}-{CONFIG.YEARS_TEST[-1]} ({len(CONFIG.YEARS_TEST)} years)")
print(f"Prediction period  : {CONFIG.YEARS_PREDICT[0]}-{CONFIG.YEARS_PREDICT[-1]}")
print(f"Pixels per strata  : {CONFIG.N_PIXELS_PER_STRATA}")
print("=" * 80)


# ==================== LOAD DATA ====================
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

    n_dry_months = np.full((n_years, shape[0], shape[1]), np.nan)
    for year_idx in range(n_years):
        monthly_prec = monthly_data_full[year_idx, :, :, :, CONFIG.VARIABLES_RAW.index('prec')]
        n_dry_months[year_idx] = np.sum(monthly_prec < 10, axis=0)

    prec_mean = np.nanmean(annual_data[:, :, :, CONFIG.VARIABLES.index('prec')], axis=0)
    land_mask_basic = ~np.isnan(prec_mean)

    print(f"Shape: {annual_data.shape}, Variables: {CONFIG.VARIABLES}")

    return {
        'data': annual_data, 'n_dry_months': n_dry_months,
        'land_mask_basic': land_mask_basic, 'bounds': sample_data['bounds'],
        'transform': sample_data['transform'], 'crs': sample_data['crs'],
        'shape': shape, 'monthly_data': monthly_data_full
    }


print("\nLoading elevation...")
elevation_data = load_geotiff(CONFIG.ELEVATION_FILE)
elevation_original = elevation_data['data']

all_data = load_monthly_data_to_annual(CONFIG.YEARS_ALL)

# Resize elevation to match climate grid
print("\nResizing elevation...")
zoom_factors = (all_data['shape'][0] / elevation_original.shape[0],
                all_data['shape'][1] / elevation_original.shape[1])
elevation = zoom(elevation_original, zoom_factors, order=1)

# Strict land mask
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


# ==================== STRATIFIED SAMPLING ====================
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

    np.random.seed(42)
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
    np.random.seed(42)
    n_sample = min(3000, len(land_pixels))
    sampled_indices = np.random.choice(len(land_pixels), n_sample, replace=False)
    land_pixels_sampled = land_pixels[sampled_indices]
    print(f"\nSampled {len(land_pixels_sampled):,} pixels (random)")


# ==================== FEATURE ENGINEERING ====================
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


def create_improved_features(pixel_data_full, pixel_monthly_data, year_idx, horizon=1):
    """Features with configurable parameters."""
    if year_idx < 10:
        return None

    features = {'year_idx': year_idx, 'horizon': horizon}

    for var_idx, var_name in enumerate(CONFIG.VARIABLES):
        pixel_history = pixel_data_full[:year_idx, var_idx]

        if np.sum(~np.isnan(pixel_history)) < 5:
            continue

        # Lag features
        for lag in range(1, CONFIG.N_LAG_FEATURES + 1):
            if year_idx - lag >= 0:
                features[f'{var_name}_lag{lag}'] = pixel_data_full[year_idx - lag, var_idx]

        # 5-year window
        if CONFIG.WINDOW_5Y_STATS and year_idx >= 5:
            window_5y = pixel_data_full[year_idx - 5:year_idx, var_idx]
            features[f'{var_name}_mean_5y'] = np.nanmean(window_5y)
            features[f'{var_name}_std_5y'] = np.nanstd(window_5y)

        # 10-year window
        if CONFIG.WINDOW_10Y_STATS and year_idx >= 10:
            window_10y = pixel_data_full[year_idx - 10:year_idx, var_idx]
            features[f'{var_name}_mean_10y'] = np.nanmean(window_10y)
            features[f'{var_name}_std_10y'] = np.nanstd(window_10y)
            features[f'{var_name}_min_10y'] = np.nanmin(window_10y)
            features[f'{var_name}_max_10y'] = np.nanmax(window_10y)

        # Trend features
        if CONFIG.TREND_FEATURES:
            if year_idx >= 10:
                features[f'{var_name}_trend_10y'] = calculate_trend(pixel_history[-10:], 10)
            if year_idx >= 20:
                trend_recent = calculate_trend(pixel_history[-10:], 10)
                trend_old = calculate_trend(pixel_history[-20:-10], 10)
                features[f'{var_name}_trend_accel'] = trend_recent - trend_old

        # Seasonal features
        if CONFIG.SEASONAL_FEATURES and var_name in ['prec', 'tmax', 'tmin']:
            if year_idx > 0:
                monthly_last_year = pixel_monthly_data[year_idx - 1, :, var_idx]

                if not np.all(np.isnan(monthly_last_year)):
                    winter_vals = [monthly_last_year[11], monthly_last_year[0], monthly_last_year[1]]
                    summer_vals = [monthly_last_year[5], monthly_last_year[6], monthly_last_year[7]]
                    spring_vals = [monthly_last_year[2], monthly_last_year[3], monthly_last_year[4]]
                    autumn_vals = [monthly_last_year[8], monthly_last_year[9], monthly_last_year[10]]

                    features[f'{var_name}_winter_mean'] = np.nanmean(winter_vals)
                    features[f'{var_name}_summer_mean'] = np.nanmean(summer_vals)
                    features[f'{var_name}_spring_mean'] = np.nanmean(spring_vals)
                    features[f'{var_name}_autumn_mean'] = np.nanmean(autumn_vals)

                    if features[f'{var_name}_winter_mean'] > 0:
                        features[f'{var_name}_ratio_summer_winter'] = (
                            features[f'{var_name}_summer_mean'] / features[f'{var_name}_winter_mean']
                        )

                    features[f'{var_name}_seasonal_amplitude'] = (
                        np.nanmax(monthly_last_year) - np.nanmin(monthly_last_year)
                    )

                    if var_name == 'prec':
                        features['prec_driest_month'] = np.nanmin(monthly_last_year)
                        features['prec_wettest_month'] = np.nanmax(monthly_last_year)

    return features


def build_dataset_multistep(pixel_data, pixel_monthly_data, elevation_val, lat, lon):
    """Build dataset for all horizons (1-6 years)."""
    rows = []
    n_years = pixel_data.shape[0]

    for year_idx in range(10, n_years - 6):
        for horizon in range(1, 7):
            row = {
                'year_idx': year_idx,
                'year': CONFIG.YEARS_ALL[year_idx],
                'elevation': elevation_val,
                'lat': lat,
                'lon': lon
            }

            var_features = create_improved_features(
                pixel_data[:, :],
                pixel_monthly_data[:, :, :],
                year_idx,
                horizon=horizon
            )

            if var_features is None:
                continue

            row.update(var_features)

            # Topography interactions
            for var_name in CONFIG.VARIABLES:
                if f'{var_name}_lag1' in row:
                    row[f'elev_x_{var_name}'] = elevation_val * row[f'{var_name}_lag1']

            # Targets
            target_idx = year_idx + horizon
            if target_idx < n_years:
                for var_idx, var_name in enumerate(CONFIG.VARIABLES):
                    row[f'{var_name}_target'] = pixel_data[target_idx, var_idx]

            rows.append(row)

    return pd.DataFrame(rows)


# ==================== BUILD DATASET ====================
print("\nBuilding dataset...")

lat_grid = np.linspace(all_data['bounds'].top, all_data['bounds'].bottom, all_data['shape'][0])
lon_grid = np.linspace(all_data['bounds'].left, all_data['bounds'].right, all_data['shape'][1])

all_rows = []
for pixel_idx in tqdm(range(len(land_pixels_sampled)), desc="Building"):
    i, j = land_pixels_sampled[pixel_idx]
    pixel_data = all_data['data'][:, i, j, :]

    pixel_monthly = np.full((len(CONFIG.YEARS_ALL), 12, len(CONFIG.VARIABLES)), np.nan)
    for var_idx, var in enumerate(CONFIG.VARIABLES):
        raw_idx = CONFIG.VARIABLES_RAW.index(var)
        pixel_monthly[:, :, var_idx] = all_data['monthly_data'][:, :, i, j, raw_idx]

    elev_val = elevation[i, j]

    pixel_df = build_dataset_multistep(pixel_data, pixel_monthly, elev_val, lat_grid[i], lon_grid[j])
    all_rows.append(pixel_df)

df = pd.concat(all_rows, ignore_index=True)
print(f"Dataset: {len(df):,} samples, {df.shape[1]} features")

# Split
train_mask = (df['year'] >= CONFIG.YEARS_TRAIN[0]) & (df['year'] <= CONFIG.YEARS_TRAIN[-1])
val_mask = (df['year'] >= CONFIG.YEARS_VAL[0]) & (df['year'] <= CONFIG.YEARS_VAL[-1])
test_mask = (df['year'] >= CONFIG.YEARS_TEST[0]) & (df['year'] <= CONFIG.YEARS_TEST[-1])

df_train = df[train_mask].copy()
df_val = df[val_mask].copy()
df_test = df[test_mask].copy()

print(f"   Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")


# ==================== TRAIN MODELS ====================
print("\nTraining XGBoost models with optimized parameters...")

target_cols = [f'{var}_target' for var in CONFIG.VARIABLES]
meta_cols = ['year', 'year_idx']
feature_cols = [col for col in df_train.columns if col not in target_cols + meta_cols]

print(f"   Features: {len(feature_cols)}")

X_train = df_train[feature_cols].fillna(0).values
X_val = df_val[feature_cols].fillna(0).values
X_test = df_test[feature_cols].fillna(0).values

models = {}
test_metrics = {}  # store metrics for later reporting

for var in CONFIG.VARIABLES:
    print(f"\n   Training: {var}")

    y_train = df_train[f'{var}_target'].values
    y_val = df_val[f'{var}_target'].values
    y_test = df_test[f'{var}_target'].values

    train_valid = ~np.isnan(y_train)
    val_valid = ~np.isnan(y_val)
    test_valid = ~np.isnan(y_test)

    params = CONFIG.XGBOOST_PARAMS.get(var, CONFIG.XGBOOST_PARAMS['default']).copy()

    print(f"   Using: n_estimators={params['n_estimators']}, "
          f"max_depth={params['max_depth']}, lr={params['learning_rate']}")

    params['early_stopping_rounds'] = CONFIG.EARLY_STOPPING_ROUNDS

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train[train_valid], y_train[train_valid],
        eval_set=[(X_val[val_valid], y_val[val_valid])],
        verbose=CONFIG.VERBOSE_TRAINING
    )
    models[var] = model

    # Model evaluation on test set
    y_pred = model.predict(X_test[test_valid])
    y_true_test = y_test[test_valid]

    mae = mean_absolute_error(y_true_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_test, y_pred))
    r2 = r2_score(y_true_test, y_pred)

    # Baselines on the same test subset
    # Persistence: predict target = lag1 (last observed year)
    lag1_col = f'{var}_lag1'
    if lag1_col in df_test.columns:
        y_pers = df_test.loc[test_valid, lag1_col].fillna(0).values
        mae_pers = mean_absolute_error(y_true_test, y_pers)
        r2_pers = r2_score(y_true_test, y_pers)
    else:
        mae_pers, r2_pers = np.nan, np.nan

    # Climatology: predict target = 10-year rolling mean (proxy for climatology)
    mean10_col = f'{var}_mean_10y'
    if mean10_col in df_test.columns:
        y_clim = df_test.loc[test_valid, mean10_col].fillna(0).values
        mae_clim = mean_absolute_error(y_true_test, y_clim)
        r2_clim = r2_score(y_true_test, y_clim)
    else:
        mae_clim, r2_clim = np.nan, np.nan

    skill_pers = 1 - mae / mae_pers if mae_pers and not np.isnan(mae_pers) else np.nan
    skill_clim = 1 - mae / mae_clim if mae_clim and not np.isnan(mae_clim) else np.nan

    test_metrics[var] = {
        'mae_xgb': mae, 'rmse_xgb': rmse, 'r2_xgb': r2,
        'mae_persistence': mae_pers, 'r2_persistence': r2_pers,
        'mae_climatology': mae_clim, 'r2_climatology': r2_clim,
        'skill_vs_persistence': skill_pers, 'skill_vs_climatology': skill_clim,
    }

    print(f"   XGBoost    -> MAE: {mae:.3f}  RMSE: {rmse:.3f}  R2: {r2:.3f}")
    print(f"   Persistence-> MAE: {mae_pers:.3f}  R2: {r2_pers:.3f}  (skill vs pers: {skill_pers:+.3f})")
    print(f"   Climatology-> MAE: {mae_clim:.3f}  R2: {r2_clim:.3f}  (skill vs clim: {skill_clim:+.3f})")
    print(f"   Best iteration: {model.best_iteration}")

# Per-horizon breakdown
print("\nWalk-forward validation (multi-horizon):")
for horizon in [1, 3, 6]:
    test_horizon = df_test[df_test['horizon'] == horizon]
    if len(test_horizon) > 0:
        print(f"\n   Horizon {horizon} year(s):")
        X_test_h = test_horizon[feature_cols].fillna(0).values

        for var in CONFIG.VARIABLES:
            y_test_h = test_horizon[f'{var}_target'].values
            valid = ~np.isnan(y_test_h)
            if np.sum(valid) > 0:
                y_pred_h = models[var].predict(X_test_h[valid])
                mae_h = mean_absolute_error(y_test_h[valid], y_pred_h)
                r2_h = r2_score(y_test_h[valid], y_pred_h)
                print(f"      {var}: MAE={mae_h:.2f}, R2={r2_h:.3f}")

with open(CONFIG.OUTPUT_DIR / 'models_improved.pkl', 'wb') as f:
    pickle.dump({
        'models': models,
        'feature_cols': feature_cols,
        'test_metrics': test_metrics,
    }, f)

print("\nModels trained.")


# ==================== DIRECT PREDICTION ====================
print("\nDirect multi-step prediction 2025-2050...")


def predict_multiyear_recursive(models, feature_cols, pixel_data, pixel_monthly,
                                elevation_val, lat, lon, base_year_idx, n_years=6,
                                lookback_window=30):
    """
    Predict recursively using a rolling window of the last N years.
    Each prediction uses the most recent 'lookback_window' years
    (real + predicted data).
    """
    predictions = {var: [] for var in CONFIG.VARIABLES}

    extended_data = np.copy(pixel_data)
    extended_monthly = np.copy(pixel_monthly)

    for year_offset in range(n_years):
        current_total_years = len(extended_data)
        window_start = max(0, current_total_years - lookback_window)
        window_end = current_total_years

        data_window = extended_data[window_start:window_end, :]
        monthly_window = extended_monthly[window_start:window_end, :, :]

        current_year_idx = base_year_idx + 1 + year_offset
        horizon = min(year_offset + 1, 6)

        row = {
            'year_idx': current_year_idx,
            'elevation': elevation_val,
            'lat': lat,
            'lon': lon
        }

        var_features = create_improved_features(
            data_window,
            monthly_window,
            len(data_window) - 1,
            horizon=horizon
        )

        if var_features is None:
            for var in CONFIG.VARIABLES:
                predictions[var].append(np.nan)
            continue

        row.update(var_features)

        for var_name in CONFIG.VARIABLES:
            if f'{var_name}_lag1' in row:
                row[f'elev_x_{var_name}'] = elevation_val * row[f'{var_name}_lag1']

        row_df = pd.DataFrame([row])
        for col in feature_cols:
            if col not in row_df.columns:
                row_df[col] = 0

        X = row_df[feature_cols].fillna(0).values

        year_predictions = {}
        for var in CONFIG.VARIABLES:
            pred = models[var].predict(X)[0]

            if var == 'tmax':
                pred = np.clip(pred, *CONFIG.TMAX_RANGE)
            elif var == 'tmin':
                pred = np.clip(pred, *CONFIG.TMIN_RANGE)
            elif var == 'prec':
                pred = max(CONFIG.PREC_MIN, pred)
            elif var == 'vap':
                pred = max(CONFIG.VAP_MIN, pred)
            elif var == 'ws':
                pred = np.clip(pred, *CONFIG.WS_RANGE)
            elif var == 'def':
                pred = max(CONFIG.DEF_MIN, pred)

            predictions[var].append(pred)
            year_predictions[var] = pred

        # Add predicted year to extended data for next iteration
        new_annual_row = np.array([year_predictions[var] for var in CONFIG.VARIABLES])
        extended_data = np.vstack([extended_data, new_annual_row.reshape(1, -1)])

        # Build monthly data for predicted year by scaling 2024 pattern
        reference_pattern = pixel_monthly[-1, :, :]
        new_monthly = np.zeros((12, len(CONFIG.VARIABLES)))

        for var_idx, var in enumerate(CONFIG.VARIABLES):
            if var in ['prec', 'tmax', 'tmin']:
                predicted_annual = year_predictions[var]
                if var == 'prec':
                    reference_annual = np.nansum(reference_pattern[:, var_idx])
                else:
                    reference_annual = np.nanmean(reference_pattern[:, var_idx])

                if reference_annual > 0.01:
                    scale_factor = predicted_annual / reference_annual
                    new_monthly[:, var_idx] = reference_pattern[:, var_idx] * scale_factor
                else:
                    new_monthly[:, var_idx] = predicted_annual / 12
            else:
                new_monthly[:, var_idx] = year_predictions[var]

        extended_monthly = np.vstack([
            extended_monthly.reshape(-1, 12, len(CONFIG.VARIABLES)),
            new_monthly.reshape(1, 12, len(CONFIG.VARIABLES))
        ])

    return predictions


# Predict all pixels
n_future_years = len(CONFIG.YEARS_PREDICT)
shape = all_data['shape']

predictions_all = {
    var: np.full((n_future_years, shape[0], shape[1]), np.nan, dtype=np.float32)
    for var in CONFIG.VARIABLES
}

land_pixels_all = np.argwhere(all_data['land_mask'])
base_year_idx = len(CONFIG.YEARS_ALL) - 1

for pixel_idx in tqdm(range(len(land_pixels_all)), desc="Predicting"):
    i, j = land_pixels_all[pixel_idx]

    pixel_data = all_data['data'][:, i, j, :]

    pixel_monthly = np.full((len(CONFIG.YEARS_ALL), 12, len(CONFIG.VARIABLES)), np.nan)
    for var_idx, var in enumerate(CONFIG.VARIABLES):
        raw_idx = CONFIG.VARIABLES_RAW.index(var)
        pixel_monthly[:, :, var_idx] = all_data['monthly_data'][:, :, i, j, raw_idx]

    elev_val = elevation[i, j]
    lat_val = lat_grid[i]
    lon_val = lon_grid[j]

    preds = predict_multiyear_recursive(
        models, feature_cols, pixel_data, pixel_monthly,
        elev_val, lat_val, lon_val, base_year_idx,
        n_years=n_future_years,
        lookback_window=CONFIG.LOOKBACK_WINDOW
    )

    for var in CONFIG.VARIABLES:
        for future_idx in range(n_future_years):
            predictions_all[var][future_idx, i, j] = preds[var][future_idx]

print("Predictions complete.")


# ==================== MONTHLY DISAGGREGATION ====================
print("\nMonthly disaggregation...")

# 1991-2020 climatological baseline (indices relative to 1958 start of TerraClimate,
# but here we use 1981 indexing consistent with YEARS_ALL)
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


# ==================== EXPORT GEOTIFFS ====================
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


# ==================== PREDICTIONS SUMMARY CSV ====================
# Structured dump of predicted values aggregated by region, with persistence
# and climatology baselines embedded in the same file. The 'observed' column
# stays blank and can be filled in later (e.g. from CHIRPS or TerraClimate)
# as ground truth for each future year becomes available.
print("\nBuilding predictions summary CSV...")

# Build 2D lat/lon grids for regional masking
lat_grid_2d = np.broadcast_to(lat_grid[:, None], shape)
lon_grid_2d = np.broadcast_to(lon_grid[None, :], shape)

# Region masks
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

# Report region sizes
print("   Region pixel counts:")
for r, m in regions.items():
    print(f"      {r}: {int(np.sum(m)):,}")

# Precompute climatology (1981-2014 training-only mean) and 2024 observed fields
# per variable, to avoid recomputing inside the loop.
clim_years_end = 2015 - CONFIG.YEARS_ALL[0]  # exclusive index = first year NOT in climatology
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
        xgb_field = predictions_all[var][future_idx]
        clim_field = climatology_fields[var]
        pers_field = last_observed_fields[var]  # flat persistence: repeat 2024

        for region_name, mask in regions.items():
            if not np.any(mask):
                continue
            summary_rows.append({
                'year': year,
                'variable': var,
                'region': region_name,
                'n_pixels': int(np.sum(mask)),
                'xgb_pred': float(np.nanmean(xgb_field[mask])),
                'persistence_2024': float(np.nanmean(pers_field[mask])),
                'climatology_1981_2014': float(np.nanmean(clim_field[mask])),
                'observed': np.nan,  # fill in later as ground truth becomes available
            })

summary_df = pd.DataFrame(summary_rows)
summary_path = CONFIG.OUTPUT_DIR / CONFIG.VALIDATION_CSV_NAME
summary_df.to_csv(summary_path, index=False)
print(f"   Summary CSV written: {summary_path}")
print(f"   Rows: {len(summary_df):,}")

# Also dump the test-set metrics table for easy reference
metrics_rows = []
for var, m in test_metrics.items():
    metrics_rows.append({'variable': var, **m})
metrics_df = pd.DataFrame(metrics_rows)
metrics_path = CONFIG.OUTPUT_DIR / "test_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"   Test metrics CSV written: {metrics_path}")


# ==================== DONE ====================
print("\n" + "=" * 80)
print("OPTIMIZED PREDICTIONS COMPLETE")
print("=" * 80)
print("\nConfiguration used:")
print(f"   Pixels per strata   : {CONFIG.N_PIXELS_PER_STRATA}")
print(f"   Lag features        : {CONFIG.N_LAG_FEATURES}")
print(f"   XGBoost trees (prec): {CONFIG.XGBOOST_PARAMS['prec']['n_estimators']}")
print(f"   Early stopping      : {CONFIG.EARLY_STOPPING_ROUNDS} rounds")
print(f"   Lookback window     : {CONFIG.LOOKBACK_WINDOW} years")
print(f"\nOutput directory: {CONFIG.OUTPUT_DIR}")
print("=" * 80)