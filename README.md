# 🌍 Climate Prediction — Morocco (2025–2050)

Machine learning pipeline to predict future climate variables across **Morocco**, trained on historical TerraClimate data (1981–2024) and visualized through Köppen-Geiger classification maps.

---

## 📁 Project Structure

```
climate-prediction-morocco/
│
├── ML/                          # XGBoost-based approach
│   ├── projectionML.py          # Main training & prediction script
│   ├── koppenviwer.py           # Tkinter GUI — Köppen-Geiger map viewer (basic)
│   ├── koppen_viewer_extended.py# Tkinter GUI — extended viewer with trends & animation
│   └── data/                   # ⚠️ Local only — not tracked by Git
│       └── {year}/
│           └── {variable}/
│               └── morocco_{variable}_{year}_{month:02d}.tif
│
├── DL/                          # Deep Learning approach
│   ├── projectionDL.py          # LSTM model — same pipeline, DL counterpart
│   └── data/                   # ⚠️ Local only — not tracked by Git
│       └── {year}/
│           └── {variable}/
│               └── morocco_{variable}_{year}_{month:02d}.tif
│
├── .gitignore
└── README.md
```

---

## 🔬 What It Does

| Step | Script | Description |
|---|---|---|
| Train & Predict (ML) | `ML/projectionML.py` | Trains one XGBoost model per climate variable, generates annual + monthly GeoTIFF predictions for 2025–2050 |
| Train & Predict (DL) | `DL/projectionDL.py` | Trains a single multi-output LSTM model for all 6 variables jointly |
| Visualize | `ML/koppen_viewer_extended.py` | Tkinter GUI — load historical or predicted data, explore variables, compare periods, animate over time |

---

## 🌡️ Climate Variables

| Variable | Description | Unit | Aggregation |
|---|---|---|---|
| `tmax` | Maximum temperature | °C | Monthly mean |
| `tmin` | Minimum temperature | °C | Monthly mean |
| `prec` | Precipitation | mm | Monthly sum |
| `vap` | Vapor pressure | kPa | Monthly mean |
| `ws` | Wind speed | m/s | Monthly mean |
| `def` | Climate water deficit | mm | Monthly sum |

---

## 🤖 ML Methodology — XGBoost

### Overview

One **XGBoost regressor** is trained per climate variable (6 models total). Each model predicts the annual value of that variable for a given pixel and future year, using only data available up to the prediction date (no leakage).

### Data & Spatial Filtering

- **Source:** TerraClimate monthly GeoTIFFs at ~4km resolution (1981–2024)
- **Variables loaded:** tmax, tmin, prec, vap, ws, soil, srad, aet, def (9 raw → 6 used)
- **Land mask:** pixels with mean annual precipitation ≥ 50mm and elevation < 4000m only
- **Stratified sampling:** rather than training on all pixels (computationally prohibitive), 150 pixels are sampled per climate stratum. Strata are defined by crossing 3 elevation zones (plain < 500m, hill 500–1500m, mountain > 1500m) with 3 precipitation zones (arid < 200mm, semi-arid 200–400mm, humid > 400mm) → up to 9 strata, ensuring representativeness across Morocco's diverse geography

### Feature Engineering

For each (pixel, year) pair, a flat feature vector is constructed:

| Feature group | What it captures |
|---|---|
| **Lag features** (t-1 to t-5) | Short-term memory — last 5 years of each variable |
| **5-year window stats** | Mean, std, min, max, range, coefficient of variation over the last 5 years |
| **10-year window stats** | Same stats over the last 10 years |
| **Linear trend (10y)** | Slope of linear regression over last 10 years |
| **Quadratic trend (10y)** | Curvature — is the trend accelerating or decelerating? |
| **Long-term trend (20y+)** | Slope over the full available history |
| **Cyclic features** | Sin/cos encoding of 5-year (El Niño-like) and 11-year (solar) cycles |
| **Climate anomaly** | Deviation from 1991–2020 baseline climatology |
| **Static features** | Elevation, latitude, longitude |
| **Horizon** | Number of years ahead being predicted (1–6) |

This gives each model a rich view of both recent conditions and long-term climate trends, while keeping the architecture simple (flat tabular input, no sequence modelling).

### Train / Validation / Test Split

All splits are temporal — no future data leaks into training:

| Split | Years | Purpose |
|---|---|---|
| Train | 1981–2014 | Model fitting |
| Validation | 2015–2017 | Hyperparameter tuning |
| Test | 2018–2024 | Final honest evaluation |

### Evaluation — Baselines

The model is compared against two naive baselines:

- **Persistence:** predict the last observed value (2024) for all future years
- **Climatology:** predict the long-term historical mean

A **skill score** is computed for each: `skill = 1 - MAE_model / MAE_baseline`. Positive skill means the model beats the baseline.

### Recursive Prediction (2025–2050)

Prediction rolls forward year by year. At each step:
1. The last 15 years of data (real + previously predicted) form the input window
2. All 6 models each predict their variable for the next year
3. Those predictions are appended to the history
4. Repeat for the next year

Long-term trend features are always recomputed from the growing history, which prevents drift over the 26-year horizon.

### Physical Constraints

Hard clipping is applied after each prediction to prevent physically impossible outputs:

| Variable | Constraint |
|---|---|
| tmax | [-10, 50] °C |
| tmin | [-20, 40] °C |
| prec | ≥ 0 mm |
| vap | ≥ 0 kPa |
| ws | [0, 20] m/s |
| def | ≥ 0 mm |

---

## 🧠 DL Methodology — LSTM

### Why a Second Model?

XGBoost treats each variable independently and uses a flat feature vector. The LSTM approach offers two architectural advantages for this problem:

1. **Sequence modelling:** the LSTM reads 15 years of raw data in order, learning temporal dynamics directly rather than from hand-crafted lag features
2. **Joint prediction:** a single model predicts all 6 variables simultaneously from a shared representation, which enforces soft physical consistency between outputs (e.g. higher temperatures tend to correlate with lower precipitation in Morocco)

### Architecture

```
Input: sequence (15 years × 6 variables)  +  static features (22 values)
         │                                          │
    ┌────▼────────────────────────────┐             │
    │  LSTM layer 1  (hidden: 128)    │             │
    │  LSTM layer 2  (hidden: 128)    │             │
    │  dropout: 0.2 between layers    │             │
    └────────────┬────────────────────┘             │
                 │  last hidden state (128)          │
                 └──────────┬────────────────────────┘
                            │  concat → (150,)
                       ┌────▼────────────┐
                       │  Dense(150→64)  │
                       │  ReLU + Dropout │
                       │  Dense(64→64)   │
                       │  ReLU + Dropout │
                       │  Dense(64→6)    │
                       └────────────────-┘
                        6 outputs (one per variable)
```

**Total parameters: ~200,000** — fits comfortably in 3GB VRAM (GTX 1060 or better). CPU fallback is automatic.

### Two-Branch Input Design

**Sequence branch (LSTM input):** raw annual values for the last 15 years, shape `(15, 6)`. The LSTM is left to discover temporal patterns on its own — lags, seasonality, momentum.

**Static branch (dense input):** 22 hand-crafted features injected directly into the dense head:

| Feature | Count | Rationale |
|---|---|---|
| Elevation, latitude, longitude | 3 | Geographic context |
| Prediction horizon | 1 | Tells the model how far ahead it's predicting |
| 10-year linear trend per variable | 6 | Hard to learn from a 15-year window alone |
| Trend acceleration per variable | 6 | Is warming speeding up or slowing down? |
| 10-year mean per variable | 6 | Local climatological baseline |

The static branch is intentionally small — if the LSTM can learn it from the raw sequence, it should. We only inject what the LSTM structurally cannot learn: long-range trends computed over the full history.

### Normalization

All inputs and targets are standardized to mean=0, std=1 using statistics computed **only from training data**. Normalization parameters are saved with the model and used at inference time. This is critical — leaking test statistics into normalization would produce overly optimistic metrics.

### Training Details

| Setting | Value | Reason |
|---|---|---|
| Loss | MSE on normalized targets | Standard for regression |
| Optimizer | Adam, lr=1e-3 | Classic default, well-tested |
| Weight decay | 1e-5 | Light L2 regularization |
| Gradient clipping | norm ≤ 1.0 | LSTMs are prone to exploding gradients without this |
| Batch size | 512 | Efficient on GPU without memory issues |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) | Halves LR when val loss plateaus |
| Early stopping | patience=15 epochs | Prevents overfitting, saves best checkpoint |
| Max epochs | 100 | Hard cap; early stopping usually triggers first |

### Train / Validation / Test Split

Same temporal logic as XGBoost, with a wider training window:

| Split | Years |
|---|---|
| Train | 1958–2014 |
| Validation | 2015–2017 |
| Test | 2018–2024 |

The LSTM uses more historical data (back to 1958) because deep learning benefits more from additional samples than XGBoost does.

### Recursive Prediction

Same rolling strategy as XGBoost — each year's prediction is appended to the history and used as input for the next. The horizon is always set to 1 (predict one year ahead), repeated 26 times to reach 2050.

---

## 🌍 Köppen-Geiger Classification

Köppen-Geiger climate types are computed from the **monthly** tmax, tmin, and prec predictions using the standard classification rules. Using monthly data (rather than annual averages) correctly identifies subtypes like Mediterranean dry-summer (`Csa`) vs. all-season humid (`Cfb`).

### Classes Present in Morocco

| Class | Name | Typical location |
|---|---|---|
| `BWh` | Hot desert | South / pre-Saharan |
| `BWk` | Cold desert | High-altitude desert |
| `BSh` | Hot steppe | Interior plains |
| `BSk` | Cold steppe | High plateau |
| `Csa` | Mediterranean hot summer | Atlantic & Rif coasts |
| `Csb` | Mediterranean warm summer | Higher coastal elevations |
| `ET` | Tundra | High Atlas peaks (> 3000m) |

---

## 🗂️ Data

Climate data comes from **TerraClimate** (University of Idaho, monthly ~4km resolution, 1958–2024).

> ⚠️ GeoTIFF files are **not included** in this repository due to size.

Expected path structure:
```
data/{year}/{variable}/morocco_{variable}_{year}_{month:02d}.tif
```

Download: [https://www.climatologylab.org/terraclimate.html](https://www.climatologylab.org/terraclimate.html)

---

## ⚙️ Installation

```bash
git clone https://github.com/YoussefGlb/climate-prediction-morocco-2050.git
cd climate-prediction-morocco-2050
pip install numpy pandas rasterio xgboost scikit-learn scipy tqdm torch matplotlib
```

For the GUI, Tkinter must be available (bundled with standard Python on Windows and macOS).

---

## 🚀 Usage

### Run XGBoost pipeline
```bash
cd ML
python projectionML.py
```
Outputs annual + monthly GeoTIFF predictions to `ML/predictions_maroc_ML_improvedV3entire/`

### Run LSTM pipeline
```bash
cd DL
python projectionDL.py
```
Outputs to `DL/predictions_maroc_DL/`. Requires PyTorch. GPU strongly recommended.

### Visualize
```bash
python koppen_viewer_extended.py
```
Opens the extended GUI. Load your historical folder and/or predicted folder, then explore variables, generate Köppen maps, compare periods, or animate the full 1981–2050 timeline.

---

## 📌 Roadmap

- [x] XGBoost baseline (tmax, tmin, prec, vap, ws, def)
- [x] Köppen-Geiger GUI viewer (basic)
- [x] LSTM deep learning model
- [x] Extended GUI with trends, animation, side-by-side comparison
- [ ] Transformer / ConvLSTM experiment
- [ ] ML vs DL quantitative comparison dashboard
- [ ] Uncertainty quantification (prediction intervals)

---

## 👤 Author

**Youssef Guelloub** — Climate ML project, Morocco