# 🌍 Climate Prediction — Morocco (2025–2050)

Machine learning pipeline to predict future climate variables across **northern Morocco**, trained on historical TerraClimate data (1981–2024) and visualized through Köppen-Geiger classification maps.

---

## 📁 Project Structure

```
climate-prediction-morocco/
│
├── ML/                          # XGBoost-based approach
│   ├── draft3_ML.py             # Main training & prediction script
│   ├── koppemv3ml.py            # Tkinter GUI — Köppen-Geiger map viewer
│   └── data/                   # ⚠️ Local only — not tracked by Git
│       └── TerraClimate_morocco_processed/
│           └── {variable}/
│               └── morocco_{var}_{year}_{month}.tif
│
├── DL/                          # Deep Learning approach (WIP)
│   ├── model_dl.py              # LSTM / Transformer model (coming soon)
│   └── data/                   # ⚠️ Local only — not tracked by Git
│       └── TerraClimate_morocco_processed/
│
├── .gitignore
└── README.md
```

---

## 🔬 What It Does

| Step | Script | Description |
|---|---|---|
| Train & Predict | `ML/draft3_ML.py` | Trains an XGBoost model per climate variable, generates monthly GeoTIFF predictions for 2025–2050 |
| Visualize | `ML/koppemv3ml.py` | Tkinter GUI that loads predictions and renders Köppen-Geiger climate classification maps |
| Deep Learning | `DL/model_dl.py` | *Work in progress* — same pipeline using LSTM/sequence models |

---

## 🌡️ Climate Variables

| Variable | Description | Unit |
|---|---|---|
| `tmax` | Maximum temperature | °C |
| `tmin` | Minimum temperature | °C |
| `prec` | Precipitation | mm |
| `vap` | Vapor pressure | kPa |
| `ws` | Wind speed | m/s |
| `def` | Climate water deficit | mm |

---

## 🤖 ML Model (XGBoost)

- **Training period:** 1981–2024
- **Validation:** 2015–2017
- **Test:** 2018–2024
- **Prediction horizon:** 2025–2050
- **Spatial extent:** Northern Morocco (lat > 27.74°)
- **Stratified sampling** by elevation zones (< 500m, 500–1500m, > 1500m) and precipitation zones

---

## 🗂️ Data

Climate data comes from **TerraClimate** (University of Idaho, monthly 4km resolution).

> ⚠️ GeoTIFF files are **not included** in this repository due to size.  
> Expected path: `ML/data/TerraClimate_morocco_processed/{variable}/morocco_{variable}_{year}_{month}.tif`

Download TerraClimate data: [https://www.climatologylab.org/terraclimate.html](https://www.climatologylab.org/terraclimate.html)

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/climate-prediction-morocco.git
cd climate-prediction-morocco
pip install numpy pandas rasterio xgboost scikit-learn scipy tqdm
```

For the Tkinter GUI, make sure Tkinter is available (included with standard Python on Windows/macOS).

---

## 🚀 Usage

### 1. Train & predict
```bash
cd ML
python draft3_ML.py
```
Outputs monthly GeoTIFF predictions to `ML/predictions_maroc_ML_improvedV3entire/`

### 2. Visualize Köppen-Geiger maps
```bash
cd ML
python koppemv3ml.py
```
Opens a GUI — select your predictions folder, choose a year or range, and generate the climate map.

---

## 🗺️ Köppen-Geiger Classes (Morocco context)

| Class | Climate | Typical zone |
|---|---|---|
| `BWh` | Hot desert | South / pre-Saharan |
| `BSh` | Hot steppe | Interior plains |
| `Csa` | Hot-summer Mediterranean | Atlantic & Rif coasts |
| `ET` | Tundra | High Atlas peaks |

---

## 📌 Roadmap

- [x] XGBoost baseline (tmax, tmin, prec, vap, ws, def)
- [x] Köppen-Geiger GUI viewer
- [ ] LSTM deep learning model (`DL/`)
- [ ] Transformer / ConvLSTM experiment
- [ ] Comparison dashboard: ML vs DL predictions

---

## 👤 Author

**Youssef** — Climate ML project, Morocco  
"# climate-prediction-morocco-2050" 
