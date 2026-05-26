"""
Morocco Climate Viewer - Extended
===================================
Tab 1 : Data Explorer   — visualise any variable (tmax, tmin, prec, vap, ws, def)
                          for historical or predicted data, single year or range average
Tab 2 : Köppen Map      — precise monthly classification, historical or predicted
Tab 3 : Trend Analysis  — which climate zones are growing / shrinking over a range
Tab 4 : Animation       — play through years like a video + side-by-side comparison

File structure expected (same for historical and predicted):
    folder/YEAR/variable/morocco_variable_YEAR_MM.tif   (monthly files)
    folder/YEAR/variable/morocco_variable_YEAR.tif      (annual file, optional)

Historical  : typically 1981-2024  (your data/ folder)
Predicted   : typically 2025-2050  (your predictions_maroc_ML/ or DL/ folder)
"""

import os
import threading
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import numpy as np
import rasterio
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MONTHLY_VARS  = ["tmax", "tmin", "prec"]          # variables with 12 monthly files
ALL_VARS      = ["tmax", "tmin", "prec", "vap", "ws", "def"]

VAR_LABELS = {
    "tmax": "Max Temperature (°C)",
    "tmin": "Min Temperature (°C)",
    "prec": "Precipitation (mm)",
    "vap":  "Vapor Pressure (kPa)",
    "ws":   "Wind Speed (m/s)",
    "def":  "Climate Water Deficit (mm)",
}

VAR_CMAPS = {
    "tmax": "hot",
    "tmin": "coolwarm",
    "prec": "Blues",
    "vap":  "YlGnBu",
    "ws":   "PuRd",
    "def":  "OrRd",
}

KOPPEN_COLORS = {
    'Af':'#0000FF','Am':'#0078FF','Aw':'#46A9FF','As':'#46A9FF',
    'BWh':'#FF0000','BWk':'#FE9695','BSh':'#F5A300','BSk':'#FFDC64',
    'Csa':'#FFFF00','Csb':'#C8C800','Csc':'#969600',
    'Cwa':'#96FF96','Cwb':'#64C864','Cwc':'#329632',
    'Cfa':'#C8FF50','Cfb':'#64FF50','Cfc':'#329632',
    'Dsa':'#FF00FF','Dsb':'#C800C8','Dsc':'#963296','Dsd':'#966496',
    'Dwa':'#AAAAFF','Dwb':'#5555FF','Dwc':'#0000FF','Dwd':'#000080',
    'Dfa':'#00FFFF','Dfb':'#37C8FF','Dfc':'#007D7D','Dfd':'#00465F',
    'ET':'#B2B2B2','EF':'#686868',
}

KOPPEN_DESCRIPTIONS = {
    'Af':'Tropical Rainforest','Am':'Tropical Monsoon','Aw':'Tropical Savanna',
    'As':'Tropical Savanna (dry summer)','BWh':'Hot Desert','BWk':'Cold Desert',
    'BSh':'Hot Semi-Arid','BSk':'Cold Semi-Arid',
    'Csa':'Mediterranean Hot Summer','Csb':'Mediterranean Warm Summer',
    'Csc':'Mediterranean Cold Summer','Cwa':'Humid Subtropical (dry winter)',
    'Cwb':'Subtropical Highland (dry winter)','Cfa':'Humid Subtropical',
    'Cfb':'Oceanic','Cfc':'Subpolar Oceanic','Dsa':'Continental Med. Hot Summer',
    'Dsb':'Continental Med. Warm Summer','ET':'Tundra','EF':'Ice Cap',
}

COLORS = {
    "bg":       "#1a1a2e",
    "panel":    "#16213e",
    "accent1":  "#e94560",
    "accent2":  "#0f3460",
    "accent3":  "#533483",
    "text":     "#eaeaea",
    "subtext":  "#aaaaaa",
    "green":    "#4ade80",
    "orange":   "#fb923c",
    "btn_hist": "#0f3460",
    "btn_pred": "#533483",
}

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
state = {
    # folders
    "hist_folder":    None,
    "pred_folder":    None,
    "hist_years":     [],
    "pred_years":     [],
    # cache: {folder_path: {year: {var: np.array(12,H,W)}}}
    "monthly_cache":  {},
    # cache: {folder_path: {year: {var: np.array(H,W)}}}  (annual)
    "annual_cache":   {},
    # ref geometry from first loaded file
    "ref_profile":    None,
    "ref_mask":       None,
    # animation
    "anim_running":   False,
    "anim_job":       None,
}


# ─────────────────────────────────────────────
# RASTER I/O
# ─────────────────────────────────────────────
def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.profile


def find_monthly_file(folder, year, var, month):
    """Find monthly file with pattern morocco_var_year_MM.tif"""
    var_dir = Path(folder) / str(year) / var
    if not var_dir.exists():
        return None
    month_pat = f"_{month:02d}.tif"
    files = [f for f in var_dir.iterdir() if f.name.endswith(month_pat)]
    return files[0] if files else None


def find_annual_file(folder, year, var):
    """Find annual file morocco_var_year.tif (no month suffix)"""
    var_dir = Path(folder) / str(year) / var
    if not var_dir.exists():
        return None
    # annual file ends with _YEAR.tif  (4 digit year, no _MM)
    candidates = [f for f in var_dir.iterdir()
                  if f.name.endswith(f"_{year}.tif")]
    return candidates[0] if candidates else None


def load_monthly_for_year(folder, year, var):
    """Load 12 monthly arrays for one var/year. Returns (12,H,W) or None."""
    cache = state["monthly_cache"].setdefault(folder, {})
    if year in cache and var in cache[year]:
        return cache[year][var]

    arrays = []
    for m in range(1, 13):
        p = find_monthly_file(folder, year, var, m)
        if p is None:
            return None
        arr, prof = load_raster(p)
        arrays.append(arr)
        if state["ref_profile"] is None:
            state["ref_profile"] = prof
            state["ref_mask"]    = ~np.isnan(arr)

    result = np.array(arrays)   # (12, H, W)
    cache.setdefault(year, {})[var] = result
    return result


def load_annual_for_year(folder, year, var):
    """Load annual array for var/year. Returns (H,W) or None."""
    cache = state["annual_cache"].setdefault(folder, {})
    if year in cache and var in cache[year]:
        return cache[year][var]

    # try explicit annual file first
    p = find_annual_file(folder, year, var)
    if p:
        arr, prof = load_raster(p)
    elif var in MONTHLY_VARS:
        # build from monthly
        monthly = load_monthly_for_year(folder, year, var)
        if monthly is None:
            return None
        arr = np.nansum(monthly, axis=0) if var == "prec" else np.nanmean(monthly, axis=0)
        prof = state["ref_profile"]
    else:
        return None

    if state["ref_profile"] is None:
        state["ref_profile"] = prof
        state["ref_mask"]    = ~np.isnan(arr)

    cache.setdefault(year, {})[var] = arr
    return arr


def detect_years(folder, year_range=(1900, 2100)):
    """Scan folder for year subfolders."""
    years = []
    try:
        for item in os.listdir(folder):
            if item.isdigit():
                y = int(item)
                if year_range[0] <= y <= year_range[1]:
                    years.append(y)
    except Exception:
        pass
    return sorted(years)


# ─────────────────────────────────────────────
# Köppen CLASSIFICATION (unchanged from original)
# ─────────────────────────────────────────────
def classify_koppen_monthly(tmax_m, tmin_m, prec_m):
    tavg_m   = (tmax_m + tmin_m) / 2
    t_annual = np.mean(tavg_m)
    t_cold   = np.min(tavg_m)
    t_warm   = np.max(tmax_m)
    p_annual = np.sum(prec_m)

    if t_warm < 10:
        return 'ET' if t_warm >= 0 else 'EF'

    summer = [3,4,5,6,7,8]
    winter = [9,10,11,0,1,2]
    p_sum  = np.sum(prec_m[summer])
    p_win  = np.sum(prec_m[winter])
    p_sum_pct = p_sum / p_annual if p_annual > 0 else 0

    if p_sum_pct >= 0.7:   C = 280
    elif p_sum_pct < 0.3:  C = 0
    else:                   C = 140

    pth = 20 * t_annual + C
    if p_annual < pth:
        if p_annual < 0.5 * pth:
            return 'BWh' if t_annual >= 18 else 'BWk'
        else:
            return 'BSh' if t_annual >= 18 else 'BSk'

    if t_cold >= 18:
        p_dry = np.min(prec_m)
        if p_dry >= 60:    return 'Af'
        if p_dry >= (100 - p_annual / 25): return 'Am'
        return 'As' if np.argmin(prec_m) in summer else 'Aw'

    ct = 'D' if t_cold < 0 else 'C'

    pw_min = np.min(prec_m[winter])
    ps_max = np.max(prec_m[summer])
    ps_min = np.min(prec_m[summer])
    pw_max = np.max(prec_m[winter])

    if pw_min < ps_max / 10:
        pp = 'w'
    elif ct == 'C':
        pp = 's' if ps_min < 40 and ps_min < pw_max / 3 else 'f'
    else:
        pp = 's' if ps_min < 30 and ps_min < pw_max / 3 else 'f'

    ma10 = np.sum(tavg_m >= 10)
    if ct == 'D' and t_cold < -38: tl = 'd'
    elif t_warm >= 22:              tl = 'a'
    elif ma10 >= 4:                 tl = 'b'
    elif 1 <= ma10 <= 3:            tl = 'c'
    else:                           tl = 'd'

    return ct + pp + tl


def compute_koppen_map(folder, years):
    """
    Compute Köppen map averaged over a list of years.
    Returns (H,W) string array of codes.
    """
    if not years:
        return None

    # accumulate monthly averages
    ref_shape = None
    tmax_acc = tmin_acc = prec_acc = None

    for year in years:
        tm = load_monthly_for_year(folder, year, "tmax")
        tn = load_monthly_for_year(folder, year, "tmin")
        pr = load_monthly_for_year(folder, year, "prec")
        if tm is None or tn is None or pr is None:
            continue
        if ref_shape is None:
            ref_shape = tm.shape[1:]
            tmax_acc = np.zeros((12, *ref_shape))
            tmin_acc = np.zeros((12, *ref_shape))
            prec_acc = np.zeros((12, *ref_shape))
        tmax_acc += tm
        tmin_acc += tn
        prec_acc += pr

    if ref_shape is None:
        return None

    n = len(years)
    tmax_avg = tmax_acc / n
    tmin_avg = tmin_acc / n
    prec_avg = prec_acc / n

    mask = state["ref_mask"] if state["ref_mask"] is not None else np.ones(ref_shape, bool)
    kmap = np.full(ref_shape, '', dtype='U3')

    for i in range(ref_shape[0]):
        for j in range(ref_shape[1]):
            if not mask[i, j]:
                continue
            tm_p = tmax_avg[:, i, j]
            tn_p = tmin_avg[:, i, j]
            pr_p = prec_avg[:, i, j]
            if np.any(np.isnan(tm_p)) or np.any(np.isnan(pr_p)):
                continue
            kmap[i, j] = classify_koppen_monthly(tm_p, tn_p, pr_p)

    return kmap


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def koppen_to_numeric(kmap):
    codes  = sorted([c for c in np.unique(kmap) if c])
    k2n    = {c: i for i, c in enumerate(codes)}
    num    = np.full(kmap.shape, np.nan)
    for c, i in k2n.items():
        num[kmap == c] = i
    colors = [KOPPEN_COLORS.get(c, '#999999') for c in codes]
    return num, codes, colors


def folder_label_short(path):
    if not path:
        return "—"
    return Path(path).name[:35]


def years_from_widgets(start_var, end_var, available):
    try:
        s = int(start_var.get())
        e = int(end_var.get())
    except ValueError:
        return None, "Invalid year selection"
    if s > e:
        return None, "Start year must be ≤ end year"
    years = [y for y in range(s, e+1) if y in available]
    if not years:
        return None, "No available years in that range"
    return years, None


def status_ok(sv, msg):   sv.set("✓  " + msg)
def status_err(sv, msg):  sv.set("✗  " + msg)
def status_info(sv, msg): sv.set("…  " + msg)


# ─────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────
root = tk.Tk()
root.title("Morocco Climate Viewer")
root.geometry("1400x900")
root.configure(bg=COLORS["bg"])

style = ttk.Style()
style.theme_use("clam")
style.configure("TNotebook",           background=COLORS["bg"],  borderwidth=0)
style.configure("TNotebook.Tab",       background=COLORS["panel"], foreground=COLORS["text"],
                padding=[14, 6], font=("Courier New", 10, "bold"))
style.map("TNotebook.Tab",
          background=[("selected", COLORS["accent1"])],
          foreground=[("selected", "white")])
style.configure("TCombobox", fieldbackground=COLORS["panel"],
                background=COLORS["panel"], foreground=COLORS["text"],
                selectbackground=COLORS["accent2"])
style.configure("TFrame", background=COLORS["bg"])
style.configure("TLabelframe", background=COLORS["panel"],
                foreground=COLORS["text"], bordercolor=COLORS["accent2"])
style.configure("TLabelframe.Label", background=COLORS["panel"], foreground=COLORS["accent1"],
                font=("Courier New", 9, "bold"))

def mk_btn(parent, text, cmd, color=None, w=18):
    c = color or COLORS["accent2"]
    return tk.Button(parent, text=text, command=cmd, width=w,
                     bg=c, fg="white", font=("Courier New", 9, "bold"),
                     relief=tk.FLAT, cursor="hand2",
                     activebackground=COLORS["accent1"], activeforeground="white")

def mk_label(parent, text, size=9, bold=False, color=None):
    f = ("Courier New", size, "bold") if bold else ("Courier New", size)
    return tk.Label(parent, text=text, font=f,
                    bg=COLORS["panel"], fg=color or COLORS["text"])

def mk_combo(parent, values, w=10):
    cb = ttk.Combobox(parent, values=values, width=w, state="readonly")
    return cb


# ═══════════════════════════════════════════════════════════
# TOP BAR — folder selection (shared across all tabs)
# ═══════════════════════════════════════════════════════════
top_bar = tk.Frame(root, bg=COLORS["panel"], pady=8, padx=12)
top_bar.pack(fill=tk.X)

tk.Label(top_bar, text="MOROCCO CLIMATE VIEWER",
         font=("Courier New", 13, "bold"),
         bg=COLORS["panel"], fg=COLORS["accent1"]).pack(side=tk.LEFT, padx=10)

# historical folder
hist_lbl = tk.Label(top_bar, text="Historical: —",
                    font=("Courier New", 9), bg=COLORS["panel"], fg=COLORS["subtext"])
hist_lbl.pack(side=tk.LEFT, padx=8)

def browse_hist():
    folder = filedialog.askdirectory(title="Select HISTORICAL data folder (1981-2024)")
    if not folder:
        return
    years = detect_years(folder)
    if not years:
        messagebox.showerror("Error", "No year subfolders found.")
        return
    state["hist_folder"] = folder
    state["hist_years"]  = years
    hist_lbl.config(text=f"Historical: {folder_label_short(folder)}  [{years[0]}–{years[-1]}]",
                    fg=COLORS["green"])
    _update_all_year_combos()
    global_status.set(f"Historical folder loaded: {len(years)} years ({years[0]}–{years[-1]})")

mk_btn(top_bar, "📂 Historical", browse_hist, COLORS["btn_hist"], 15).pack(side=tk.LEFT, padx=4)

pred_lbl = tk.Label(top_bar, text="Predicted: —",
                    font=("Courier New", 9), bg=COLORS["panel"], fg=COLORS["subtext"])
pred_lbl.pack(side=tk.LEFT, padx=8)

def browse_pred():
    folder = filedialog.askdirectory(title="Select PREDICTED data folder (2025-2050)")
    if not folder:
        return
    years = detect_years(folder)
    if not years:
        messagebox.showerror("Error", "No year subfolders found.")
        return
    state["pred_folder"] = folder
    state["pred_years"]  = years
    pred_lbl.config(text=f"Predicted: {folder_label_short(folder)}  [{years[0]}–{years[-1]}]",
                    fg=COLORS["orange"])
    _update_all_year_combos()
    global_status.set(f"Predicted folder loaded: {len(years)} years ({years[0]}–{years[-1]})")

mk_btn(top_bar, "📂 Predicted", browse_pred, COLORS["btn_pred"], 15).pack(side=tk.LEFT, padx=4)

global_status = tk.StringVar(value="Load a historical or predicted folder to begin.")
tk.Label(top_bar, textvariable=global_status,
         font=("Courier New", 8), bg=COLORS["panel"], fg=COLORS["subtext"]).pack(side=tk.RIGHT, padx=10)


# ═══════════════════════════════════════════════════════════
# NOTEBOOK
# ═══════════════════════════════════════════════════════════
nb = ttk.Notebook(root)
nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

# helper — all combo widgets that need updating when folders change
_year_combos_to_update = []   # list of (combo_widget, source_key)  source_key = "hist"|"pred"|"both"

def _update_all_year_combos():
    for cb, src in _year_combos_to_update:
        if src == "hist":
            vals = state["hist_years"]
        elif src == "pred":
            vals = state["pred_years"]
        else:
            vals = sorted(set(state["hist_years"]) | set(state["pred_years"]))
        cb["values"] = vals
        if vals:
            if not cb.get() or int(cb.get()) not in vals:
                cb.set(vals[0])

def register_combo(cb, src):
    _year_combos_to_update.append((cb, src))


# ───────────────────────────────────────────────────────────
# TAB 1 — DATA EXPLORER
# ───────────────────────────────────────────────────────────
tab1 = tk.Frame(nb, bg=COLORS["bg"])
nb.add(tab1, text="  📊 Data Explorer  ")

# -- controls
ctrl1 = tk.Frame(tab1, bg=COLORS["panel"], padx=10, pady=8)
ctrl1.pack(fill=tk.X)

mk_label(ctrl1, "SOURCE:", bold=True).pack(side=tk.LEFT, padx=4)
t1_src = tk.StringVar(value="Historical")
for v, c in [("Historical", COLORS["btn_hist"]), ("Predicted", COLORS["btn_pred"])]:
    tk.Radiobutton(ctrl1, text=v, variable=t1_src, value=v,
                   bg=COLORS["panel"], fg=COLORS["text"], selectcolor=COLORS["accent2"],
                   activebackground=COLORS["panel"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=4)

ttk.Separator(ctrl1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

mk_label(ctrl1, "VARIABLE:", bold=True).pack(side=tk.LEFT, padx=4)
t1_var = mk_combo(ctrl1, ALL_VARS, 8)
t1_var.set("tmax")
t1_var.pack(side=tk.LEFT, padx=4)

ttk.Separator(ctrl1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

mk_label(ctrl1, "MODE:", bold=True).pack(side=tk.LEFT, padx=4)
t1_mode = tk.StringVar(value="Single Year")
for v in ["Single Year", "Range Average"]:
    tk.Radiobutton(ctrl1, text=v, variable=t1_mode, value=v,
                   bg=COLORS["panel"], fg=COLORS["text"], selectcolor=COLORS["accent2"],
                   activebackground=COLORS["panel"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=4)

ttk.Separator(ctrl1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

mk_label(ctrl1, "FROM:", bold=True).pack(side=tk.LEFT, padx=2)
t1_ys = mk_combo(ctrl1, [], 7);  t1_ys.pack(side=tk.LEFT, padx=2); register_combo(t1_ys, "both")
mk_label(ctrl1, "TO:", bold=True).pack(side=tk.LEFT, padx=2)
t1_ye = mk_combo(ctrl1, [], 7);  t1_ye.pack(side=tk.LEFT, padx=2); register_combo(t1_ye, "both")

t1_status = tk.StringVar(value="Load a folder then press Plot.")
mk_btn(ctrl1, "▶  PLOT", lambda: _t1_plot(), COLORS["accent1"], 12).pack(side=tk.LEFT, padx=10)
tk.Label(ctrl1, textvariable=t1_status, font=("Courier New", 8),
         bg=COLORS["panel"], fg=COLORS["subtext"]).pack(side=tk.LEFT, padx=4)

# -- figure
t1_fig, t1_ax = plt.subplots(figsize=(10, 7))
t1_fig.patch.set_facecolor("#1a1a2e")
t1_ax.set_facecolor("#1a1a2e")
t1_canvas = FigureCanvasTkAgg(t1_fig, master=tab1)
t1_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
t1_cbar = [None]

def _t1_plot():
    src    = t1_src.get()
    var    = t1_var.get()
    mode   = t1_mode.get()
    folder = state["hist_folder"] if src == "Historical" else state["pred_folder"]
    avail  = state["hist_years"]  if src == "Historical" else state["pred_years"]

    if not folder:
        t1_status.set(f"✗  {src} folder not loaded.")
        return

    try:
        ys = int(t1_ys.get());  ye = int(t1_ye.get())
    except ValueError:
        t1_status.set("✗  Invalid year selection.")
        return

    years = [y for y in range(ys, ye+1) if y in avail]
    if not years:
        t1_status.set("✗  No available years in that range.")
        return

    status_info(t1_status, f"Loading {var} …")
    root.update_idletasks()

    arrays = []
    for year in years:
        arr = load_annual_for_year(folder, year, var)
        if arr is not None:
            arrays.append(arr)

    if not arrays:
        t1_status.set("✗  Could not load data.")
        return

    if mode == "Single Year":
        data = arrays[0]
        title = f"{VAR_LABELS[var]}  —  {years[0]}"
    else:
        data = np.nanmean(np.stack(arrays), axis=0)
        title = f"{VAR_LABELS[var]}  —  Average {years[0]}–{years[-1]}"

    mask = state["ref_mask"]

    t1_ax.clear()
    if t1_cbar[0] is not None:
        try: t1_cbar[0].remove()
        except: pass

    disp = np.where(mask, data, np.nan) if mask is not None else data
    im = t1_ax.imshow(disp, cmap=VAR_CMAPS[var], interpolation="nearest")
    t1_ax.set_title(title, color="white", fontsize=12, fontweight="bold")
    t1_ax.axis("off")
    t1_ax.set_facecolor("#1a1a2e")

    cb = t1_fig.colorbar(im, ax=t1_ax, shrink=0.75, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="white")
    cb.ax.set_ylabel(VAR_LABELS[var], color="white", fontsize=9)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    t1_cbar[0] = cb

    t1_fig.tight_layout()
    t1_canvas.draw_idle()
    status_ok(t1_status, f"{src} | {var} | {years[0]}{'–'+str(years[-1]) if len(years)>1 else ''}")


# ───────────────────────────────────────────────────────────
# TAB 2 — KÖPPEN MAP
# ───────────────────────────────────────────────────────────
tab2 = tk.Frame(nb, bg=COLORS["bg"])
nb.add(tab2, text="  🌍 Köppen Map  ")

ctrl2 = tk.Frame(tab2, bg=COLORS["panel"], padx=10, pady=8)
ctrl2.pack(fill=tk.X)

mk_label(ctrl2, "SOURCE:", bold=True).pack(side=tk.LEFT, padx=4)
t2_src = tk.StringVar(value="Historical")
for v, c in [("Historical", COLORS["btn_hist"]), ("Predicted", COLORS["btn_pred"])]:
    tk.Radiobutton(ctrl2, text=v, variable=t2_src, value=v,
                   bg=COLORS["panel"], fg=COLORS["text"], selectcolor=COLORS["accent2"],
                   activebackground=COLORS["panel"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=4)

ttk.Separator(ctrl2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

mk_label(ctrl2, "MODE:", bold=True).pack(side=tk.LEFT, padx=4)
t2_mode = tk.StringVar(value="Single Year")
for v in ["Single Year", "Range Average"]:
    tk.Radiobutton(ctrl2, text=v, variable=t2_mode, value=v,
                   bg=COLORS["panel"], fg=COLORS["text"], selectcolor=COLORS["accent2"],
                   activebackground=COLORS["panel"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=4)

ttk.Separator(ctrl2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

mk_label(ctrl2, "FROM:", bold=True).pack(side=tk.LEFT, padx=2)
t2_ys = mk_combo(ctrl2, [], 7); t2_ys.pack(side=tk.LEFT, padx=2); register_combo(t2_ys, "both")
mk_label(ctrl2, "TO:", bold=True).pack(side=tk.LEFT, padx=2)
t2_ye = mk_combo(ctrl2, [], 7); t2_ye.pack(side=tk.LEFT, padx=2); register_combo(t2_ye, "both")

t2_status = tk.StringVar(value="Load a folder then press Compute.")
mk_btn(ctrl2, "▶  COMPUTE", lambda: _t2_compute(), COLORS["accent1"], 14).pack(side=tk.LEFT, padx=10)

t2_export_btn = mk_btn(ctrl2, "💾 Export GeoTIFF", lambda: _t2_export(), COLORS["accent3"], 16)
t2_export_btn.pack(side=tk.LEFT, padx=4)
t2_export_btn.config(state=tk.DISABLED)

tk.Label(ctrl2, textvariable=t2_status, font=("Courier New", 8),
         bg=COLORS["panel"], fg=COLORS["subtext"]).pack(side=tk.LEFT, padx=4)

# stats panel right side
t2_right = tk.Frame(tab2, bg=COLORS["bg"])
t2_right.pack(side=tk.RIGHT, fill=tk.Y, padx=4, pady=4)
t2_stats_text = tk.Text(t2_right, width=22, bg=COLORS["panel"], fg=COLORS["text"],
                         font=("Courier New", 8), relief=tk.FLAT, state=tk.DISABLED)
t2_stats_text.pack(fill=tk.BOTH, expand=True)

t2_fig, t2_ax = plt.subplots(figsize=(9, 7))
t2_fig.patch.set_facecolor("#1a1a2e")
t2_ax.set_facecolor("#1a1a2e")
t2_canvas = FigureCanvasTkAgg(t2_fig, master=tab2)
t2_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=4)
t2_cbar = [None]
t2_kmap = [None]    # store current koppen map for export

def _t2_compute():
    src    = t2_src.get()
    folder = state["hist_folder"] if src == "Historical" else state["pred_folder"]
    avail  = state["hist_years"]  if src == "Historical" else state["pred_years"]

    if not folder:
        t2_status.set(f"✗  {src} folder not loaded.")
        return

    try:
        ys = int(t2_ys.get());  ye = int(t2_ye.get())
    except ValueError:
        t2_status.set("✗  Invalid year selection.")
        return

    if t2_mode.get() == "Single Year":
        years = [ys] if ys in avail else []
    else:
        years = [y for y in range(ys, ye+1) if y in avail]

    if not years:
        t2_status.set("✗  No available years.")
        return

    status_info(t2_status, f"Computing Köppen for {len(years)} year(s)…")
    root.update_idletasks()

    kmap = compute_koppen_map(folder, years)
    if kmap is None:
        t2_status.set("✗  Could not load monthly data (need tmax, tmin, prec).")
        return

    t2_kmap[0] = kmap
    _t2_draw(kmap, years, src)
    t2_export_btn.config(state=tk.NORMAL)
    status_ok(t2_status, f"{src} | Köppen | {years[0]}{'–'+str(years[-1]) if len(years)>1 else ''}")


def _t2_draw(kmap, years, src_label):
    num, codes, colors = koppen_to_numeric(kmap)
    cmap = ListedColormap(colors)
    mask = state["ref_mask"]

    t2_ax.clear()
    if t2_cbar[0]:
        try: t2_cbar[0].remove()
        except: pass

    disp = np.where(mask, num, np.nan) if mask is not None else num
    im = t2_ax.imshow(disp, cmap=cmap, interpolation="nearest",
                       vmin=0, vmax=max(len(codes)-1, 1))

    label = years[0] if len(years) == 1 else f"{years[0]}–{years[-1]}"
    t2_ax.set_title(f"Köppen-Geiger  —  Morocco  {label}  [{src_label}]",
                    color="white", fontsize=12, fontweight="bold")
    t2_ax.axis("off")

    cb = t2_fig.colorbar(im, ax=t2_ax, shrink=0.75, ticks=range(len(codes)))
    cb.ax.set_yticklabels(codes, fontsize=9, color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    t2_cbar[0] = cb

    t2_fig.tight_layout()
    t2_canvas.draw_idle()

    # update stats panel
    total = int(np.sum(mask)) if mask is not None else np.sum(kmap != '')
    t2_stats_text.config(state=tk.NORMAL)
    t2_stats_text.delete("1.0", tk.END)
    t2_stats_text.insert(tk.END, f"{'CODE':<5} {'%':>6}  DESCRIPTION\n")
    t2_stats_text.insert(tk.END, "─"*40 + "\n")
    for code in sorted(codes):
        cnt = int(np.sum(kmap == code))
        pct = 100 * cnt / total if total else 0
        desc = KOPPEN_DESCRIPTIONS.get(code, code)[:18]
        t2_stats_text.insert(tk.END, f"{code:<5} {pct:>5.1f}%  {desc}\n")
    t2_stats_text.config(state=tk.DISABLED)


def _t2_export():
    kmap = t2_kmap[0]
    if kmap is None:
        return
    out = filedialog.asksaveasfilename(defaultextension=".tif",
                                       filetypes=[("GeoTIFF","*.tif")])
    if not out:
        return
    codes = sorted([c for c in np.unique(kmap) if c])
    c2n   = {c: i+1 for i, c in enumerate(codes)}
    num   = np.zeros(kmap.shape, dtype=np.int16)
    for c, n in c2n.items():
        num[kmap == c] = n
    mask = state["ref_mask"]
    if mask is not None:
        num[~mask] = -9999
    prof = state["ref_profile"].copy()
    prof.update(dtype=rasterio.int16, nodata=-9999)
    with rasterio.open(out, 'w', **prof) as dst:
        dst.write(num, 1)
    messagebox.showinfo("Exported", f"GeoTIFF saved:\n{out}")


# ───────────────────────────────────────────────────────────
# TAB 3 — TREND ANALYSIS
# ───────────────────────────────────────────────────────────
tab3 = tk.Frame(nb, bg=COLORS["bg"])
nb.add(tab3, text="  📈 Trend Analysis  ")

ctrl3 = tk.Frame(tab3, bg=COLORS["panel"], padx=10, pady=8)
ctrl3.pack(fill=tk.X)

mk_label(ctrl3, "PERIOD A  FROM:", bold=True).pack(side=tk.LEFT, padx=4)
t3_a_ys = mk_combo(ctrl3, [], 7); t3_a_ys.pack(side=tk.LEFT, padx=2); register_combo(t3_a_ys, "both")
mk_label(ctrl3, "TO:", bold=True).pack(side=tk.LEFT, padx=2)
t3_a_ye = mk_combo(ctrl3, [], 7); t3_a_ye.pack(side=tk.LEFT, padx=2); register_combo(t3_a_ye, "both")
t3_a_src = tk.StringVar(value="Historical")
for v in ["Historical","Predicted"]:
    tk.Radiobutton(ctrl3, text=v, variable=t3_a_src, value=v,
                   bg=COLORS["panel"], fg=COLORS["text"], selectcolor=COLORS["accent2"],
                   activebackground=COLORS["panel"], font=("Courier New", 8)).pack(side=tk.LEFT, padx=2)

ttk.Separator(ctrl3, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

mk_label(ctrl3, "PERIOD B  FROM:", bold=True).pack(side=tk.LEFT, padx=4)
t3_b_ys = mk_combo(ctrl3, [], 7); t3_b_ys.pack(side=tk.LEFT, padx=2); register_combo(t3_b_ys, "both")
mk_label(ctrl3, "TO:", bold=True).pack(side=tk.LEFT, padx=2)
t3_b_ye = mk_combo(ctrl3, [], 7); t3_b_ye.pack(side=tk.LEFT, padx=2); register_combo(t3_b_ye, "both")
t3_b_src = tk.StringVar(value="Predicted")
for v in ["Historical","Predicted"]:
    tk.Radiobutton(ctrl3, text=v, variable=t3_b_src, value=v,
                   bg=COLORS["panel"], fg=COLORS["text"], selectcolor=COLORS["accent2"],
                   activebackground=COLORS["panel"], font=("Courier New", 8)).pack(side=tk.LEFT, padx=2)

t3_status = tk.StringVar(value="Set two periods then press Compare.")
mk_btn(ctrl3, "▶  COMPARE", lambda: _t3_compare(), COLORS["accent1"], 12).pack(side=tk.LEFT, padx=10)
tk.Label(ctrl3, textvariable=t3_status, font=("Courier New", 8),
         bg=COLORS["panel"], fg=COLORS["subtext"]).pack(side=tk.LEFT, padx=4)

t3_fig = plt.figure(figsize=(13, 6))
t3_fig.patch.set_facecolor("#1a1a2e")
t3_canvas = FigureCanvasTkAgg(t3_fig, master=tab3)
t3_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

def _t3_compare():
    def get_folder_years(src_var, ys_var, ye_var):
        src = src_var.get()
        folder = state["hist_folder"] if src == "Historical" else state["pred_folder"]
        avail  = state["hist_years"]  if src == "Historical" else state["pred_years"]
        if not folder:
            return None, None, None, f"{src} folder not loaded"
        try:
            ys = int(ys_var.get()); ye = int(ye_var.get())
        except:
            return None, None, None, "Invalid year"
        years = [y for y in range(ys, ye+1) if y in avail]
        return folder, years, src, None

    fa, ya, la, ea = get_folder_years(t3_a_src, t3_a_ys, t3_a_ye)
    fb, yb, lb, eb = get_folder_years(t3_b_src, t3_b_ys, t3_b_ye)

    if ea: t3_status.set(f"✗  Period A: {ea}"); return
    if eb: t3_status.set(f"✗  Period B: {eb}"); return
    if not ya: t3_status.set("✗  Period A: no years."); return
    if not yb: t3_status.set("✗  Period B: no years."); return

    status_info(t3_status, "Computing Köppen maps…"); root.update_idletasks()

    kmap_a = compute_koppen_map(fa, ya)
    kmap_b = compute_koppen_map(fb, yb)
    if kmap_a is None or kmap_b is None:
        t3_status.set("✗  Failed to compute Köppen (missing tmax/tmin/prec)."); return

    mask = state["ref_mask"]
    total = int(np.sum(mask)) if mask is not None else kmap_a.size

    # count pixels per code in each period
    all_codes = sorted(set(
        [c for c in np.unique(kmap_a) if c] +
        [c for c in np.unique(kmap_b) if c]
    ))

    cnt_a = {c: int(np.sum(kmap_a == c)) for c in all_codes}
    cnt_b = {c: int(np.sum(kmap_b == c)) for c in all_codes}
    deltas = {c: cnt_b[c] - cnt_a.get(c, 0) for c in all_codes}

    t3_fig.clear()
    gs = t3_fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.35)
    ax_bar  = t3_fig.add_subplot(gs[0])
    ax_heat = t3_fig.add_subplot(gs[1])

    # bar chart — sorted by delta
    sorted_codes = sorted(all_codes, key=lambda c: deltas[c])
    bar_colors   = [KOPPEN_COLORS.get(c, '#999') for c in sorted_codes]
    delta_vals   = [deltas[c] for c in sorted_codes]
    bar_x        = range(len(sorted_codes))

    bars = ax_bar.bar(bar_x, delta_vals, color=bar_colors, edgecolor="#333")
    ax_bar.axhline(0, color="white", linewidth=0.8, alpha=0.6)
    ax_bar.set_xticks(list(bar_x))
    ax_bar.set_xticklabels(sorted_codes, rotation=45, ha="right",
                            fontsize=8, color="white")
    ax_bar.set_ylabel("Pixel change (B − A)", color="white", fontsize=9)
    ax_bar.set_title(f"Climate Zone Change\n"
                     f"A: {la} {ya[0]}–{ya[-1]}   →   B: {lb} {yb[0]}–{yb[-1]}",
                     color="white", fontsize=10, fontweight="bold")
    ax_bar.set_facecolor("#16213e")
    ax_bar.tick_params(colors="white")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#333")

    # heatmap of change — per pixel: same=0, lost=−1, gained=+1 per code bucket
    change_map = np.zeros(kmap_a.shape)
    if mask is not None:
        change_map[~mask] = np.nan
    change_map[kmap_a != kmap_b] = 1     # changed
    change_map[kmap_a == kmap_b] = 0     # stable
    change_map[(kmap_a == '') & (kmap_b == '')] = np.nan

    im = ax_heat.imshow(change_map, cmap="RdYlGn_r", interpolation="nearest", vmin=0, vmax=1)
    ax_heat.set_title("Changed pixels (red)", color="white", fontsize=9)
    ax_heat.axis("off")
    ax_heat.set_facecolor("#1a1a2e")

    changed = int(np.nansum(change_map))
    pct_chg = 100 * changed / total if total else 0
    ax_heat.set_xlabel(f"{changed:,} pixels changed ({pct_chg:.1f}%)",
                       color=COLORS["subtext"], fontsize=8)

    t3_fig.patch.set_facecolor("#1a1a2e")
    t3_canvas.draw_idle()
    status_ok(t3_status, f"A: {la} {ya[0]}–{ya[-1]}  vs  B: {lb} {yb[0]}–{yb[-1]}")


# ───────────────────────────────────────────────────────────
# TAB 4 — ANIMATION
# ───────────────────────────────────────────────────────────
tab4 = tk.Frame(nb, bg=COLORS["bg"])
nb.add(tab4, text="  🎬 Animation  ")

ctrl4 = tk.Frame(tab4, bg=COLORS["panel"], padx=10, pady=8)
ctrl4.pack(fill=tk.X)

mk_label(ctrl4, "SOURCE:", bold=True).pack(side=tk.LEFT, padx=4)
t4_src = tk.StringVar(value="Historical")
for v in ["Historical","Predicted","Both (combined)"]:
    tk.Radiobutton(ctrl4, text=v, variable=t4_src, value=v,
                   bg=COLORS["panel"], fg=COLORS["text"], selectcolor=COLORS["accent2"],
                   activebackground=COLORS["panel"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=4)

ttk.Separator(ctrl4, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

mk_label(ctrl4, "MODE:", bold=True).pack(side=tk.LEFT, padx=4)
t4_mode = tk.StringVar(value="Köppen")
for v in ["Köppen","Variable"]:
    tk.Radiobutton(ctrl4, text=v, variable=t4_mode, value=v,
                   bg=COLORS["panel"], fg=COLORS["text"], selectcolor=COLORS["accent2"],
                   activebackground=COLORS["panel"], font=("Courier New", 9)).pack(side=tk.LEFT, padx=4)
t4_var = mk_combo(ctrl4, ALL_VARS, 7); t4_var.set("tmax"); t4_var.pack(side=tk.LEFT, padx=4)

ttk.Separator(ctrl4, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

mk_label(ctrl4, "FROM:", bold=True).pack(side=tk.LEFT, padx=2)
t4_ys = mk_combo(ctrl4, [], 7); t4_ys.pack(side=tk.LEFT, padx=2); register_combo(t4_ys, "both")
mk_label(ctrl4, "TO:", bold=True).pack(side=tk.LEFT, padx=2)
t4_ye = mk_combo(ctrl4, [], 7); t4_ye.pack(side=tk.LEFT, padx=2); register_combo(t4_ye, "both")

ttk.Separator(ctrl4, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

mk_label(ctrl4, "SPEED (ms):", bold=True).pack(side=tk.LEFT, padx=2)
t4_speed = ttk.Scale(ctrl4, from_=100, to=2000, orient=tk.HORIZONTAL, length=100)
t4_speed.set(600)
t4_speed.pack(side=tk.LEFT, padx=4)

t4_play_btn  = mk_btn(ctrl4, "▶  PLAY",  lambda: _t4_play(),  COLORS["green"],  10)
t4_stop_btn  = mk_btn(ctrl4, "■  STOP",  lambda: _t4_stop(),  COLORS["accent1"], 10)
t4_play_btn.pack(side=tk.LEFT, padx=4)
t4_stop_btn.pack(side=tk.LEFT, padx=4)

# compare row
ctrl4b = tk.Frame(tab4, bg=COLORS["panel"], padx=10, pady=4)
ctrl4b.pack(fill=tk.X)
mk_label(ctrl4b, "SIDE-BY-SIDE: Year A:", bold=True).pack(side=tk.LEFT, padx=4)
t4_cmp_a = mk_combo(ctrl4b, [], 7); t4_cmp_a.pack(side=tk.LEFT, padx=2); register_combo(t4_cmp_a, "both")
t4_cmp_a_src = tk.StringVar(value="Historical")
for v in ["Historical","Predicted"]:
    tk.Radiobutton(ctrl4b, text=v, variable=t4_cmp_a_src, value=v,
                   bg=COLORS["panel"], fg=COLORS["text"], selectcolor=COLORS["accent2"],
                   activebackground=COLORS["panel"], font=("Courier New", 8)).pack(side=tk.LEFT, padx=2)
mk_label(ctrl4b, "  Year B:", bold=True).pack(side=tk.LEFT, padx=8)
t4_cmp_b = mk_combo(ctrl4b, [], 7); t4_cmp_b.pack(side=tk.LEFT, padx=2); register_combo(t4_cmp_b, "both")
t4_cmp_b_src = tk.StringVar(value="Predicted")
for v in ["Historical","Predicted"]:
    tk.Radiobutton(ctrl4b, text=v, variable=t4_cmp_b_src, value=v,
                   bg=COLORS["panel"], fg=COLORS["text"], selectcolor=COLORS["accent2"],
                   activebackground=COLORS["panel"], font=("Courier New", 8)).pack(side=tk.LEFT, padx=2)
mk_btn(ctrl4b, "◉ COMPARE", lambda: _t4_compare(), COLORS["accent3"], 12).pack(side=tk.LEFT, padx=10)

t4_status = tk.StringVar(value="Set source and range, then press Play.")
tk.Label(ctrl4b, textvariable=t4_status, font=("Courier New", 8),
         bg=COLORS["panel"], fg=COLORS["subtext"]).pack(side=tk.LEFT, padx=8)

# year label
t4_year_lbl = tk.Label(tab4, text="", font=("Courier New", 16, "bold"),
                        bg=COLORS["bg"], fg=COLORS["accent1"])
t4_year_lbl.pack()

# figure (1 or 2 panels)
t4_fig = plt.figure(figsize=(13, 7))
t4_fig.patch.set_facecolor("#1a1a2e")
t4_canvas = FigureCanvasTkAgg(t4_fig, master=tab4)
t4_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=2)

# animation state
_anim = {"running": False, "job": None, "years": [], "idx": 0,
         "folders": [], "cbar": None}


def _t4_get_years_folders():
    src = t4_src.get()
    try:
        ys = int(t4_ys.get()); ye = int(t4_ye.get())
    except:
        return [], []
    if src == "Historical":
        avail = state["hist_years"]; folders = [state["hist_folder"]] * 200
    elif src == "Predicted":
        avail = state["pred_years"]; folders = [state["pred_folder"]] * 200
    else:  # Both
        avail = sorted(set(state["hist_years"]) | set(state["pred_years"]))
        folders = []
        for y in avail:
            if y in state["hist_years"]:
                folders.append(state["hist_folder"])
            else:
                folders.append(state["pred_folder"])

    year_folder_pairs = [(y, folders[i]) for i, y in enumerate(avail)
                         if ys <= y <= ye and folders[i]]
    years   = [p[0] for p in year_folder_pairs]
    fldrs   = [p[1] for p in year_folder_pairs]
    return years, fldrs


def _t4_render_year(year, folder):
    t4_fig.clear()
    ax = t4_fig.add_subplot(111)
    ax.set_facecolor("#1a1a2e")
    mode = t4_mode.get()

    if mode == "Köppen":
        kmap = compute_koppen_map(folder, [year])
        if kmap is None: return
        num, codes, colors = koppen_to_numeric(kmap)
        cmap = ListedColormap(colors)
        mask = state["ref_mask"]
        disp = np.where(mask, num, np.nan) if mask is not None else num
        im = ax.imshow(disp, cmap=cmap, interpolation="nearest",
                        vmin=0, vmax=max(len(codes)-1, 1))
        cb = t4_fig.colorbar(im, ax=ax, shrink=0.65, ticks=range(len(codes)))
        cb.ax.set_yticklabels(codes, fontsize=7, color="white")
        cb.ax.yaxis.set_tick_params(color="white")
    else:
        var = t4_var.get()
        arr = load_annual_for_year(folder, year, var)
        if arr is None: return
        mask = state["ref_mask"]
        disp = np.where(mask, arr, np.nan) if mask is not None else arr
        im = ax.imshow(disp, cmap=VAR_CMAPS[var], interpolation="nearest")
        cb = t4_fig.colorbar(im, ax=ax, shrink=0.65)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    src_label = "Hist" if folder == state["hist_folder"] else "Pred"
    ax.set_title(f"{year}  [{src_label}]", color="white", fontsize=13, fontweight="bold")
    ax.axis("off")
    t4_fig.tight_layout()
    t4_canvas.draw_idle()
    t4_year_lbl.config(text=str(year))


def _t4_step():
    if not _anim["running"]:
        return
    idx = _anim["idx"]
    years  = _anim["years"]
    folders = _anim["folders"]
    if idx >= len(years):
        _anim["running"] = False
        t4_status.set("✓  Animation complete.")
        return
    year   = years[idx]
    folder = folders[idx]
    _t4_render_year(year, folder)
    _anim["idx"] = idx + 1
    delay = max(100, int(t4_speed.get()))
    _anim["job"] = root.after(delay, _t4_step)


def _t4_play():
    years, folders = _t4_get_years_folders()
    if not years:
        t4_status.set("✗  No years available. Check folders and range."); return
    _t4_stop()
    _anim["years"] = years
    _anim["folders"] = folders
    _anim["idx"] = 0
    _anim["running"] = True
    status_info(t4_status, f"Playing {len(years)} years…")
    _t4_step()


def _t4_stop():
    _anim["running"] = False
    if _anim["job"]:
        root.after_cancel(_anim["job"])
        _anim["job"] = None


def _t4_compare():
    try:
        ya = int(t4_cmp_a.get()); yb = int(t4_cmp_b.get())
    except:
        t4_status.set("✗  Invalid comparison years."); return

    fa = state["hist_folder"] if t4_cmp_a_src.get() == "Historical" else state["pred_folder"]
    fb = state["hist_folder"] if t4_cmp_b_src.get() == "Historical" else state["pred_folder"]
    if not fa or not fb:
        t4_status.set("✗  Folder not loaded."); return

    status_info(t4_status, "Loading comparison…"); root.update_idletasks()

    mode = t4_mode.get()
    t4_fig.clear()
    gs = t4_fig.add_gridspec(1, 2, wspace=0.08)

    for col, (year, folder, src_label) in enumerate([
            (ya, fa, t4_cmp_a_src.get()), (yb, fb, t4_cmp_b_src.get())]):
        ax = t4_fig.add_subplot(gs[col])
        ax.set_facecolor("#1a1a2e")

        if mode == "Köppen":
            kmap = compute_koppen_map(folder, [year])
            if kmap is None: continue
            num, codes, colors = koppen_to_numeric(kmap)
            cmap = ListedColormap(colors)
            mask = state["ref_mask"]
            disp = np.where(mask, num, np.nan) if mask is not None else num
            im = ax.imshow(disp, cmap=cmap, interpolation="nearest",
                            vmin=0, vmax=max(len(codes)-1, 1))
            if col == 1:
                cb = t4_fig.colorbar(im, ax=ax, shrink=0.7, ticks=range(len(codes)))
                cb.ax.set_yticklabels(codes, fontsize=7, color="white")
                cb.ax.yaxis.set_tick_params(color="white")
        else:
            var = t4_var.get()
            arr = load_annual_for_year(folder, year, var)
            if arr is None: continue
            mask = state["ref_mask"]
            disp = np.where(mask, arr, np.nan) if mask is not None else arr
            im = ax.imshow(disp, cmap=VAR_CMAPS[var], interpolation="nearest")
            if col == 1:
                cb = t4_fig.colorbar(im, ax=ax, shrink=0.7)
                cb.ax.yaxis.set_tick_params(color="white")
                plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

        ax.set_title(f"{year}  [{src_label}]", color="white", fontsize=12, fontweight="bold")
        ax.axis("off")

    t4_fig.patch.set_facecolor("#1a1a2e")
    t4_fig.tight_layout()
    t4_canvas.draw_idle()
    t4_year_lbl.config(text=f"{ya}  vs  {yb}")
    status_ok(t4_status, f"Comparing {ya} ({t4_cmp_a_src.get()}) vs {yb} ({t4_cmp_b_src.get()})")


root.mainloop()
