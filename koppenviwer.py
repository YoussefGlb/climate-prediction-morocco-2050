"""
Köppen-Geiger Classification - Monthly Data (PRECISE)
✅ Uses 12 monthly files per variable (tmax_MM, tmin_MM, prec_MM)
✅ Accurate Köppen classification with real seasonal patterns
✅ Flexible year range selection OR single year
✅ Refresh button to reload data
"""

import os
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from pathlib import Path

# ==================================================
# CONFIG
# ==================================================
MONTHLY_VARS = ["tmax", "tmin", "prec"]  # Variables with 12 monthly files

# Köppen-Geiger standard colors
KOPPEN_COLORS = {
    'Af': '#0000FF',  'Am': '#0078FF',  'Aw': '#46A9FF',  'As': '#46A9FF',
    'BWh': '#FF0000', 'BWk': '#FE9695', 'BSh': '#F5A300', 'BSk': '#FFDC64',
    'Csa': '#FFFF00', 'Csb': '#C8C800', 'Csc': '#969600',
    'Cwa': '#96FF96', 'Cwb': '#64C864', 'Cwc': '#329632',
    'Cfa': '#C8FF50', 'Cfb': '#64FF50', 'Cfc': '#329632',
    'Dsa': '#FF00FF', 'Dsb': '#C800C8', 'Dsc': '#963296', 'Dsd': '#966496',
    'Dwa': '#AAAAFF', 'Dwb': '#5555FF', 'Dwc': '#0000FF', 'Dwd': '#000080',
    'Dfa': '#00FFFF', 'Dfb': '#37C8FF', 'Dfc': '#007D7D', 'Dfd': '#00465F',
    'ET': '#B2B2B2',  'EF': '#686868',
}

# ==================================================
# GLOBALS
# ==================================================
monthly_data = {}  # {year: {var: [12 months]}}
koppen_map = None
ref_mask = None
ref_profile = None
predictions_folder = None
selected_years = []
available_years = []
colorbar_ref = None
is_single_year_mode = False

# ==================================================
# REFRESH FUNCTION
# ==================================================
def refresh_all():
    """Reset everything and start fresh"""
    global monthly_data, koppen_map, ref_mask, ref_profile
    global predictions_folder, selected_years, available_years, colorbar_ref
    
    monthly_data = {}
    koppen_map = None
    ref_mask = None
    ref_profile = None
    predictions_folder = None
    selected_years = []
    available_years = []
    colorbar_ref = None
    
    # Reset GUI
    folder_label.config(text="No folder selected")
    year_range_label.config(text="", fg="#666")
    single_year_label.config(text="", fg="#666")
    status.set("Select predictions folder to start")
    
    year_start_combo.set('')
    year_start_combo['values'] = []
    year_end_combo.set('')
    year_end_combo['values'] = []
    single_year_combo.set('')
    single_year_combo['values'] = []
    
    year_range_frame.pack_forget()
    single_year_frame.pack_forget()
    
    validate_btn.config(state=tk.DISABLED)
    validate_single_btn.config(state=tk.DISABLED)
    load_btn.config(state=tk.DISABLED)
    compute_btn.config(state=tk.DISABLED)
    export_btn.config(state=tk.DISABLED)
    
    # Clear plot
    ax.clear()
    if colorbar_ref is not None:
        colorbar_ref.remove()
        colorbar_ref = None
    canvas.draw_idle()
    
    print("\n🔄 Application refreshed - Ready to start\n")

# ==================================================
# BROWSE FUNCTION
# ==================================================
def browse_predictions_folder():
    """Browse ML predictions base folder"""
    global predictions_folder, available_years
    
    folder = filedialog.askdirectory(
        title="Select ML Predictions Base Folder (containing year subfolders)"
    )
    
    if not folder:
        return
    
    # Detect available years
    available_years = []
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path) and item.isdigit():
            year = int(item)
            if 2020 <= year <= 2100:
                available_years.append(year)
    
    available_years.sort()
    
    if not available_years:
        messagebox.showerror(
            "No Years Found",
            f"No valid year folders found in:\n{folder}\n\n"
            f"Expected structure: folder/YEAR/ (e.g., 2025/, 2026/, etc.)"
        )
        return
    
    predictions_folder = folder
    folder_label.config(text=f"📁 {Path(folder).name}")
    
    # Update ALL year selectors
    year_start_combo['values'] = available_years
    year_end_combo['values'] = available_years
    single_year_combo['values'] = available_years
    
    if len(available_years) >= 2:
        year_start_combo.set(available_years[0])
        year_end_combo.set(available_years[-1])
    elif len(available_years) == 1:
        year_start_combo.set(available_years[0])
        year_end_combo.set(available_years[0])
    
    # Show both options
    year_range_frame.pack(pady=5)
    single_year_frame.pack(pady=5)
    
    status.set(f"Found {len(available_years)} years: {available_years[0]}-{available_years[-1]} - Select mode")
    validate_btn.config(state=tk.NORMAL)
    validate_single_btn.config(state=tk.NORMAL)

# ==================================================
# YEAR RANGE MODE
# ==================================================
def validate_year_range():
    """Validate and confirm year range selection (AVERAGE mode)"""
    global selected_years, is_single_year_mode
    
    try:
        year_start = int(year_start_combo.get())
        year_end = int(year_end_combo.get())
    except:
        messagebox.showerror("Error", "Please select valid start and end years")
        return
    
    if year_start > year_end:
        messagebox.showerror("Error", "Start year must be <= end year")
        return
    
    if year_start not in available_years or year_end not in available_years:
        messagebox.showerror("Error", "Selected years not available in folder")
        return
    
    selected_years = list(range(year_start, year_end + 1))
    is_single_year_mode = False
    
    # Verify all years exist
    missing_years = []
    for year in selected_years:
        year_path = os.path.join(predictions_folder, str(year))
        if not os.path.exists(year_path):
            missing_years.append(year)
    
    if missing_years:
        messagebox.showerror(
            "Missing Years",
            f"Missing year folders: {', '.join(map(str, missing_years))}"
        )
        return
    
    year_range_label.config(
        text=f"✅ AVERAGE Mode: {year_start}-{year_end} ({len(selected_years)} years)",
        fg="green"
    )
    single_year_label.config(text="", fg="#666")
    
    status.set(f"Average mode: {year_start}-{year_end} - Click 'Load Data'")
    load_btn.config(state=tk.NORMAL)

# ==================================================
# SINGLE YEAR MODE
# ==================================================
def validate_single_year():
    """Validate single year selection"""
    global selected_years, is_single_year_mode
    
    try:
        year = int(single_year_combo.get())
    except:
        messagebox.showerror("Error", "Please select a valid year")
        return
    
    if year not in available_years:
        messagebox.showerror("Error", "Selected year not available in folder")
        return
    
    year_path = os.path.join(predictions_folder, str(year))
    if not os.path.exists(year_path):
        messagebox.showerror("Error", f"Year folder not found: {year_path}")
        return
    
    selected_years = [year]
    is_single_year_mode = True
    
    single_year_label.config(
        text=f"✅ SINGLE Year Mode: {year}",
        fg="blue"
    )
    year_range_label.config(text="", fg="#666")
    
    status.set(f"Single year mode: {year} - Click 'Load Data'")
    load_btn.config(state=tk.NORMAL)

# ==================================================
# LOAD MONTHLY DATA
# ==================================================
def load_raster(path):
    """Load a single GeoTIFF file"""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = src.profile
    return arr, profile

def load_multiyear_monthly_data():
    """Load 12 monthly files for each variable for selected years"""
    global monthly_data, ref_mask, ref_profile
    
    if not predictions_folder or not selected_years:
        messagebox.showerror("Error", "No folder or year range selected!")
        return
    
    monthly_data = {}
    ref_mask = None
    ref_profile = None
    
    year_start = selected_years[0]
    year_end = selected_years[-1]
    
    mode_text = "SINGLE YEAR" if is_single_year_mode else "AVERAGE"
    print(f"\n📂 Loading {year_start}-{year_end} monthly predictions ({mode_text} mode)...\n")
    
    try:
        for year in selected_years:
            year_folder = os.path.join(predictions_folder, str(year))
            monthly_data[year] = {}
            
            print(f"Loading {year}...")
            
            for var in MONTHLY_VARS:
                var_folder = os.path.join(year_folder, var)
                
                if not os.path.exists(var_folder):
                    raise ValueError(f"Missing variable folder: {var_folder}")
                
                # ✅ Load 12 monthly files
                monthly_data[year][var] = []
                
                for month in range(1, 13):
                    # Find file with pattern: *_MM.tif (e.g., morocco_prec_2029_01.tif)
                    month_pattern = f"_{month:02d}.tif"
                    
                    matching_files = [f for f in os.listdir(var_folder) 
                                     if f.endswith(month_pattern)]
                    
                    if len(matching_files) == 0:
                        raise ValueError(
                            f"Missing monthly file for {var} {year} month {month:02d}\n"
                            f"Expected pattern: *_{month:02d}.tif in {var_folder}"
                        )
                    elif len(matching_files) > 1:
                        raise ValueError(
                            f"Multiple files found for {var} {year} month {month:02d}: {matching_files}"
                        )
                    
                    path = os.path.join(var_folder, matching_files[0])
                    arr, profile = load_raster(path)
                    
                    monthly_data[year][var].append(arr)
                    
                    if ref_profile is None:
                        ref_profile = profile
                        ref_mask = ~np.isnan(arr)
                
                # Convert to numpy array (12, rows, cols)
                monthly_data[year][var] = np.array(monthly_data[year][var])
                
                print(f"  ✓ {var}: {monthly_data[year][var].shape} (12 months)")
        
        print(f"\n✅ Monthly data for {year_start}-{year_end} loaded!")
        print(f"Shape per month: {monthly_data[selected_years[0]]['tmax'][0].shape}")
        print(f"Valid pixels: {np.sum(ref_mask):,}\n")
        
        mode_text = f"Single year {year_start}" if is_single_year_mode else f"Average {year_start}-{year_end}"
        status.set(f"Monthly data loaded ({mode_text}) - Ready to compute Köppen")
        compute_btn.config(state=tk.NORMAL)
        
    except Exception as e:
        messagebox.showerror("Error Loading Data", str(e))
        print(f"❌ Error: {e}")

# ==================================================
# KÖPPEN-GEIGER CLASSIFICATION (MONTHLY DATA)
# ==================================================
def classify_koppen_monthly(tmax_months, tmin_months, prec_months):
    """
    Köppen classification using MONTHLY data (PRECISE)
    CORRECTED to match reference implementation
    
    Parameters:
    - tmax_months: array of 12 monthly tmax values
    - tmin_months: array of 12 monthly tmin values
    - prec_months: array of 12 monthly prec values
    
    Returns:
    - Köppen code (e.g., 'BSk', 'Csa', etc.)
    """
    
    # ✅ CORRECTION 1: Calculate tavg and use it for t_coldest
    tavg_months = (tmax_months + tmin_months) / 2
    t_annual = np.mean(tavg_months)
    t_coldest = np.min(tavg_months)  # ✅ Changed from tmin_months to tavg_months
    t_warmest = np.max(tmax_months)
    p_annual = np.sum(prec_months)
    
    # ===== 1. POLAR CLIMATES (E) =====
    if t_warmest < 10:
        return 'ET' if t_warmest >= 0 else 'EF'
    
    # ===== 2. ARID CLIMATES (B) =====
    # Summer/Winter months (Northern Hemisphere - Morocco)
    summer_months = [3, 4, 5, 6, 7, 8]  # April-September
    winter_months = [9, 10, 11, 0, 1, 2]  # October-March
    
    p_summer = np.sum(prec_months[summer_months])
    p_winter = np.sum(prec_months[winter_months])
    
    # Calculate arid threshold
    p_summer_pct = p_summer / p_annual if p_annual > 0 else 0
    
    if p_summer_pct >= 0.7:
        C = 280
    elif p_summer_pct < 0.3:
        C = 0
    else:
        C = 140
    
    pth = 20 * t_annual + C
    
    # Arid check
    if p_annual < pth:
        if p_annual < 0.5 * pth:
            return 'BWh' if t_annual >= 18 else 'BWk'
        else:
            return 'BSh' if t_annual >= 18 else 'BSk'
    
    # ===== 3. TROPICAL (A) =====
    if t_coldest >= 18:  # ✅ Now uses tavg_coldest instead of tmin_coldest
        p_driest = np.min(prec_months)
        
        if p_driest >= 60:
            return 'Af'
        elif p_driest >= (100 - p_annual / 25):
            return 'Am'
        else:
            # Determine dry season
            driest_month_idx = np.argmin(prec_months)
            if driest_month_idx in summer_months:
                return 'As'
            else:
                return 'Aw'
    
    # ===== 4. TEMPERATE (C) vs CONTINENTAL (D) =====
    # ✅ CORRECTION 2: Changed threshold from -3 to 0
    if t_coldest < 0:  # ✅ Changed from <= -3 to < 0
        climate_type = 'D'
    else:
        climate_type = 'C'
    
    # ===== 5. PRECIPITATION PATTERN (s/w/f) =====
    p_winter_min = np.min(prec_months[winter_months])
    p_summer_max = np.max(prec_months[summer_months])
    p_summer_min = np.min(prec_months[summer_months])
    p_winter_max = np.max(prec_months[winter_months])
    
    # Check for dry winter first
    if p_winter_min < p_summer_max / 10:
        precip_pattern = 'w'
    # ✅ CORRECTION 3: Different thresholds for C and D climates
    elif climate_type == 'C':
        if p_summer_min < 40 and p_summer_min < p_winter_max / 3:
            precip_pattern = 's'
        else:
            precip_pattern = 'f'
    else:  # climate_type == 'D'
        if p_summer_min < 30 and p_summer_min < p_winter_max / 3:  # ✅ 30 instead of 40 for D
            precip_pattern = 's'
        else:
            precip_pattern = 'f'
    
    # ===== 6. TEMPERATURE SUBTYPE (a/b/c/d) =====
    months_above_10 = np.sum(tavg_months >= 10)
    
    # ✅ CORRECTION 4: Priority check for very cold D climates
    if climate_type == 'D' and t_coldest < -38:
        temp_letter = 'd'
    elif t_warmest >= 22:
        temp_letter = 'a'
    elif months_above_10 >= 4:
        temp_letter = 'b'
    elif 1 <= months_above_10 <= 3:
        temp_letter = 'c'
    else:
        temp_letter = 'd'
    
    return climate_type + precip_pattern + temp_letter

def compute_koppen():
    """Compute Köppen classification (single year OR average)"""
    global koppen_map
    
    if not monthly_data or not selected_years:
        messagebox.showerror("Error", "Load data first!")
        return
    
    year_start = selected_years[0]
    year_end = selected_years[-1]
    
    mode_text = f"SINGLE YEAR {year_start}" if is_single_year_mode else f"AVERAGE {year_start}-{year_end}"
    print(f"\n🌍 Computing Köppen-Geiger ({mode_text})...\n")
    
    # Get shape
    shape = monthly_data[selected_years[0]]['tmax'][0].shape
    
    # Average monthly data over all selected years (or just use single year)
    tmax_avg_monthly = np.zeros((12, shape[0], shape[1]))
    tmin_avg_monthly = np.zeros((12, shape[0], shape[1]))
    prec_avg_monthly = np.zeros((12, shape[0], shape[1]))
    
    for year in selected_years:
        tmax_avg_monthly += monthly_data[year]['tmax']
        tmin_avg_monthly += monthly_data[year]['tmin']
        prec_avg_monthly += monthly_data[year]['prec']
    
    tmax_avg_monthly /= len(selected_years)
    tmin_avg_monthly /= len(selected_years)
    prec_avg_monthly /= len(selected_years)
    
    if is_single_year_mode:
        print(f"📊 Using single year {year_start} monthly data")
    else:
        print(f"📊 Averaged monthly data over {len(selected_years)} years")
    
    print(f"  tmax shape: {tmax_avg_monthly.shape} (12 months)")
    print(f"  Example: Jan tmax mean = {np.nanmean(tmax_avg_monthly[0]):.1f}°C")
    print(f"  Example: Jul prec mean = {np.nanmean(prec_avg_monthly[6]):.1f} mm\n")
    
    # Classify each pixel
    koppen_map = np.full(shape, '', dtype='U3')
    
    total_pixels = np.sum(ref_mask)
    processed = 0
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            if not ref_mask[i, j]:
                continue
            
            # Extract 12-month timeseries for this pixel
            tmax_pixel = tmax_avg_monthly[:, i, j]
            tmin_pixel = tmin_avg_monthly[:, i, j]
            prec_pixel = prec_avg_monthly[:, i, j]
            
            if np.any(np.isnan(tmax_pixel)) or np.any(np.isnan(prec_pixel)):
                continue
            
            # ✅ Classify with PRECISE monthly data
            koppen_code = classify_koppen_monthly(tmax_pixel, tmin_pixel, prec_pixel)
            koppen_map[i, j] = koppen_code
            
            processed += 1
            if processed % 10000 == 0:
                print(f"  Progress: {processed}/{total_pixels} ({100*processed/total_pixels:.1f}%)")
    
    print(f"\n✅ Köppen classification complete!\n")
    
    # Statistics
    unique_codes = np.unique(koppen_map[ref_mask])
    print(f"📊 Climate types in {mode_text}:")
    for code in sorted(unique_codes):
        if code:
            count = np.sum(koppen_map == code)
            pct = 100 * count / total_pixels
            print(f"  {code}: {count:,} pixels ({pct:.1f}%)")
    
    draw_koppen_map()
    status.set(f"Köppen computed ({mode_text}) - Ready to export")
    export_btn.config(state=tk.NORMAL)

# ==================================================
# VISUALIZATION (FIXED)
# ==================================================
def draw_koppen_map():
    """Draw Köppen map with fixed size and no legend accumulation"""
    global colorbar_ref
    
    ax.clear()
    if colorbar_ref is not None:
        colorbar_ref.remove()
        colorbar_ref = None
    
    if koppen_map is None:
        canvas.draw_idle()
        return
    
    year_start = selected_years[0]
    year_end = selected_years[-1]
    
    koppen_codes = [code for code in np.unique(koppen_map) if code != '']
    koppen_codes.sort()
    
    koppen_numeric = np.full(koppen_map.shape, np.nan)
    koppen_to_num = {code: i for i, code in enumerate(koppen_codes)}
    
    for code in koppen_codes:
        mask = koppen_map == code
        koppen_numeric[mask] = koppen_to_num[code]
    
    colors = [KOPPEN_COLORS.get(code, '#CCCCCC') for code in koppen_codes]
    cmap = ListedColormap(colors)
    
    im = ax.imshow(koppen_numeric, cmap=cmap, interpolation="nearest",
                   vmin=0, vmax=len(koppen_codes)-1)
    
    if is_single_year_mode:
        title = f"Köppen-Geiger – Morocco {year_start} (ML Monthly Data)"
    else:
        title = f"Köppen-Geiger – Morocco {year_start}-{year_end} Average (ML Monthly Data)"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis("off")
    
    colorbar_ref = fig.colorbar(im, ax=ax, shrink=0.75, ticks=range(len(koppen_codes)))
    colorbar_ref.set_label('Köppen Climate Type', fontsize=10)
    colorbar_ref.ax.set_yticklabels(koppen_codes, fontsize=9)
    
    fig.tight_layout()
    canvas.draw_idle()

def export_koppen_geotiff():
    if koppen_map is None:
        messagebox.showwarning("Warning", "No Köppen map to export!")
        return
    
    year_start = selected_years[0]
    year_end = selected_years[-1]
    
    if is_single_year_mode:
        default_name = f"morocco_koppen_{year_start}_monthly.tif"
    else:
        default_name = f"morocco_koppen_{year_start}-{year_end}_monthly.tif"
    
    out_path = filedialog.asksaveasfilename(
        defaultextension=".tif",
        filetypes=[("GeoTIFF", "*.tif")],
        initialfile=default_name
    )
    
    if not out_path:
        return
    
    unique_codes = sorted([code for code in np.unique(koppen_map) if code != ''])
    code_to_num = {code: i+1 for i, code in enumerate(unique_codes)}
    
    numeric_map = np.zeros(koppen_map.shape, dtype=np.int16)
    for code, num in code_to_num.items():
        numeric_map[koppen_map == code] = num
    
    numeric_map[~ref_mask] = -9999
    
    profile = ref_profile.copy()
    profile.update(dtype=rasterio.int16, nodata=-9999)
    
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(numeric_map, 1)
    
    legend_path = out_path.replace('.tif', '_legend.txt')
    with open(legend_path, 'w') as f:
        if is_single_year_mode:
            f.write(f"Köppen-Geiger Climate Classification - Morocco {year_start}\n")
        else:
            f.write(f"Köppen-Geiger Climate Classification - Morocco {year_start}-{year_end} Average\n")
        f.write("="*60 + "\n\n")
        f.write("Based on ML predictions (MONTHLY data - PRECISE)\n")
        if is_single_year_mode:
            f.write(f"Year: {year_start}\n\n")
        else:
            f.write(f"Years averaged: {', '.join(map(str, selected_years))}\n")
            f.write(f"Number of years: {len(selected_years)}\n\n")
        f.write("Value | Köppen Code | Description\n")
        f.write("-" * 60 + "\n")
        
        descriptions = {
            'Af': 'Tropical Rainforest', 'Am': 'Tropical Monsoon', 
            'Aw': 'Tropical Savanna', 'As': 'Tropical Savanna (dry summer)',
            'BWh': 'Hot Desert', 'BWk': 'Cold Desert',
            'BSh': 'Hot Semi-Arid', 'BSk': 'Cold Semi-Arid',
            'Csa': 'Mediterranean Hot Summer', 'Csb': 'Mediterranean Warm Summer',
            'Csc': 'Mediterranean Cold Summer',
            'Cwa': 'Humid Subtropical (dry winter)', 
            'Cwb': 'Subtropical Highland (dry winter)',
            'Cfa': 'Humid Subtropical', 'Cfb': 'Oceanic', 'Cfc': 'Subpolar Oceanic',
            'Dsa': 'Continental Mediterranean Hot Summer',
            'Dsb': 'Continental Mediterranean Warm Summer',
            'Dwa': 'Continental Humid Hot Summer (dry winter)',
            'Dwb': 'Continental Humid Warm Summer (dry winter)',
            'Dfa': 'Continental Humid Hot Summer',
            'Dfb': 'Continental Humid Warm Summer',
            'Dfc': 'Subarctic', 'Dfd': 'Subarctic (severe winter)',
            'ET': 'Tundra', 'EF': 'Ice Cap'
        }
        
        for code, num in sorted(code_to_num.items(), key=lambda x: x[1]):
            desc = descriptions.get(code, 'Unknown')
            f.write(f"{num:5d} | {code:3s} | {desc}\n")
    
    mode_text = f"Single year {year_start}" if is_single_year_mode else f"Average {year_start}-{year_end} ({len(selected_years)} years)"
    
    messagebox.showinfo("Success", 
        f"✅ GeoTIFF saved:\n{out_path}\n\n"
        f"📄 Legend:\n{legend_path}\n\n"
        f"📊 Period: {mode_text}\n"
        f"✅ PRECISE classification (monthly data)"
    )

# ==================================================
# GUI
# ==================================================
root = tk.Tk()
root.title("Köppen-Geiger - Monthly Data (PRECISE) + Refresh")
root.geometry("1200x900")

top = tk.Frame(root, bg="#f0f0f0", pady=10)
top.pack(fill=tk.X)

# Step 0: Refresh button
refresh_frame = tk.Frame(top, bg="#f0f0f0")
refresh_frame.pack(pady=5)

tk.Button(refresh_frame, text="🔄 REFRESH / RESTART", 
          command=refresh_all,
          width=20, bg="#E91E63", fg="white", 
          font=("Arial", 10, "bold")).pack(pady=5)

# Step 1: Browse
step1_frame = tk.Frame(top, bg="#f0f0f0")
step1_frame.pack(pady=5)

tk.Label(step1_frame, text="1. Select ML Predictions Base Folder:", 
         font=("Arial", 10, "bold"), bg="#f0f0f0").pack(side=tk.LEFT, padx=5)

tk.Button(step1_frame, text="📁 Browse Predictions Folder", 
          command=browse_predictions_folder,
          width=25, bg="#4CAF50", fg="white", 
          font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)

folder_label = tk.Label(step1_frame, text="No folder selected", 
                        font=("Arial", 9), bg="#f0f0f0", fg="#666")
folder_label.pack(side=tk.LEFT, padx=10)

# Step 2A: Year range (AVERAGE mode)
year_range_frame = tk.Frame(top, bg="#f0f0f0")

tk.Label(year_range_frame, text="2A. AVERAGE Mode - Select Year Range:", 
         font=("Arial", 10, "bold"), bg="#f0f0f0").pack(side=tk.LEFT, padx=5)

tk.Label(year_range_frame, text="From:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
year_start_combo = ttk.Combobox(year_range_frame, width=8, state="readonly")
year_start_combo.pack(side=tk.LEFT, padx=5)

tk.Label(year_range_frame, text="To:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
year_end_combo = ttk.Combobox(year_range_frame, width=8, state="readonly")
year_end_combo.pack(side=tk.LEFT, padx=5)

validate_btn = tk.Button(year_range_frame, text="✓ Validate Range", 
                        command=validate_year_range,
                        width=15, bg="#FF5722", fg="white", 
                        font=("Arial", 9, "bold"), state=tk.DISABLED)
validate_btn.pack(side=tk.LEFT, padx=5)

year_range_label = tk.Label(year_range_frame, text="", 
                            font=("Arial", 9), bg="#f0f0f0", fg="#666")
year_range_label.pack(side=tk.LEFT, padx=10)

# Step 2B: Single year mode
single_year_frame = tk.Frame(top, bg="#f0f0f0")

tk.Label(single_year_frame, text="2B. SINGLE YEAR Mode - Select One Year:", 
         font=("Arial", 10, "bold"), bg="#f0f0f0").pack(side=tk.LEFT, padx=5)

tk.Label(single_year_frame, text="Year:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
single_year_combo = ttk.Combobox(single_year_frame, width=8, state="readonly")
single_year_combo.pack(side=tk.LEFT, padx=5)

validate_single_btn = tk.Button(single_year_frame, text="✓ Validate Year", 
                               command=validate_single_year,
                               width=15, bg="#2196F3", fg="white", 
                               font=("Arial", 9, "bold"), state=tk.DISABLED)
validate_single_btn.pack(side=tk.LEFT, padx=5)

single_year_label = tk.Label(single_year_frame, text="", 
                             font=("Arial", 9), bg="#f0f0f0", fg="#666")
single_year_label.pack(side=tk.LEFT, padx=10)

# Step 3-5: Actions
step3_frame = tk.Frame(top, bg="#f0f0f0")
step3_frame.pack(pady=5)

load_btn = tk.Button(step3_frame, text="3. Load Monthly Data", 
                     command=load_multiyear_monthly_data,
                     width=20, bg="#FF9800", fg="white", 
                     font=("Arial", 9, "bold"), state=tk.DISABLED)
load_btn.pack(side=tk.LEFT, padx=5)

compute_btn = tk.Button(step3_frame, text="4. Compute Köppen (Precise)", 
                        command=compute_koppen,
                        width=25, bg="#00BCD4", fg="white", 
                        font=("Arial", 9, "bold"), state=tk.DISABLED)
compute_btn.pack(side=tk.LEFT, padx=5)

export_btn = tk.Button(step3_frame, text="5. Export GeoTIFF", 
                       command=export_koppen_geotiff,
                       width=18, bg="#673AB7", fg="white", 
                       font=("Arial", 9, "bold"), state=tk.DISABLED)
export_btn.pack(side=tk.LEFT, padx=5)

status = tk.StringVar(value="Select predictions folder to start")
status_label = tk.Label(root, textvariable=status, font=("Arial", 10), 
                       fg="#555", bg="#e8f5e9", pady=8)
status_label.pack(fill=tk.X)

info_frame = tk.Frame(root, bg="#fff3e0", pady=5)
info_frame.pack(fill=tk.X)

info_text = """📁 Expected Structure: predictions_folder/
  ├── 2025/
  │   ├── tmax/
  │   │   ├── morocco_tmax_2025_01.tif
  │   │   ├── morocco_tmax_2025_02.tif
  │   │   └── ... (12 monthly files)
  │   ├── tmin/ (12 monthly files)
  │   └── prec/ (12 monthly files)
  └── 2026/ ...

✅ Two modes: AVERAGE (multiple years) OR SINGLE YEAR
✅ Uses MONTHLY data for PRECISE Köppen classification
✅ Detects summer-dry (Csa), winter-dry (Cwa), etc.
✅ REFRESH button to start over"""

tk.Label(info_frame, text=info_text, font=("Courier", 8), 
         bg="#fff3e0", fg="#555", justify=tk.LEFT).pack()

fig, ax = plt.subplots(figsize=(10, 7))
fig.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

root.mainloop()