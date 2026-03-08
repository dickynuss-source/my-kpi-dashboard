import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings
import re 
import os
import gdown 
import pyarrow.parquet as pq
import gc

warnings.simplefilter(action='ignore', category=FutureWarning)

# ================= PAGE CONFIGURATION =================
st.set_page_config(page_title="Multi-RAT KPI Dashboard", page_icon="📶", layout="wide")
st.title("📶 Multi-RAT (2G/4G/5G) Daily KPI Dashboard")
st.markdown("---")

HD_CONFIG = {
    'toImageButtonOptions': {'format': 'png', 'filename': 'KPI_Chart_Export', 'height': 700, 'width': 1200, 'scale': 2}
}

# ================= GOOGLE DRIVE IDs MAPPING =================
GDRIVE_FILE_IDS = {
    "Master_2GDaily.parquet": "1-NT-NtuVoyxvdZw-A8ypl4jhJRgOs2xy",
    "Master_4GDaily.parquet": "1PxhJRu9ruYS8SfJ7gMbhcs-4xVOouEw3",
    "Master_No_PLMN.parquet": "1Simg9uithTM5sRF5IqeSchhGagd1Yczf", # Link 5G Fix
    "Master_4GBH.parquet":    "1dDY3d3pJ1WxfJQVzjeFx0Z-T35bLpB61",
    "Master_LTE.parquet":     "1hY4B6ZfMJbAG8lIgt5LAp6n4jD11hEMm",
    "Master_GSM.parquet":     "1haxfl2PF3Q-haQVIYad5k48w8TPow8Rx"
}

# ================= HELPER FUNCTIONS =================
def format_x_axis(fig, num_days=10):
    if num_days <= 14: interval = 1
    elif num_days <= 28: interval = 2
    elif num_days <= 42: interval = 3
    elif num_days <= 70: interval = 5
    else: interval = max(1, num_days // 14)
    
    tick_ms = 86400000 * interval 
    fig.update_xaxes(tickangle=-45, dtick=tick_ms, tickformat="%Y-%m-%d", tickfont=dict(color='#2c3e50', size=24), showgrid=True, gridwidth=1, gridcolor='#e5e8e8', linecolor='#bdc3c7', linewidth=1) 
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e8e8', linecolor='#bdc3c7', linewidth=1)
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#2c3e50'), margin=dict(l=10, r=10, t=40, b=80))
    return fig

def get_col(df, possible_names):
    if df.empty: return possible_names[0]
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    for n in possible_names:
        n_lower = n.lower().strip()
        if n_lower in df_cols_lower: return df_cols_lower[n_lower]
    return possible_names[0]

# ================= DATA LOADING (DIET RAM + ONLINE VECTORIZATION) =================
@st.cache_data(ttl=timedelta(hours=12)) 
def load_data():
    dfs = {}
    # Hanya sedot kolom yang relevan agar memori server Cloud tidak meledak
    keep_keywords = [
        'date', 'begintime', 'operator', 'opr', 'cluster', 'project_cluster', 
        'tower', 'site', 'cell', 'moentity', 'payload', 'rrc', 'traffic', 
        'nav', 'avail', 'thp', 'throughput', 'prb', 'ta', 'cqi'
    ]
    
    for filename, file_id in GDRIVE_FILE_IDS.items():
        if not os.path.exists(filename):
            try:
                gdown.download(f'https://drive.google.com/uc?id={file_id}', filename, quiet=True)
            except Exception:
                pass
            
        try:
            parquet_file = pq.ParquetFile(filename)
            existing_cols = parquet_file.schema.names
            cols_to_load = [c for c in existing_cols if any(k in c.lower() for k in keep_keywords)]
            
            df = pd.read_parquet(filename, columns=cols_to_load) if cols_to_load else pd.read_parquet(filename)
            
            if not df.empty:
                date_col = next((c for c in df.columns if c.lower().strip() in ['date', 'begintime']), None)
                if date_col:
                    df['Date_Temp'] = pd.to_datetime(df[date_col], errors='coerce').dt.date
                    if date_col != 'Date': 
                        df = df.drop(columns=[date_col])
                    df['Date'] = df['Date_Temp']
                    df = df.drop(columns=['Date_Temp'])
            dfs[filename] = df
        except Exception:
            dfs[filename] = pd.DataFrame()
            
        gc.collect()

    df_5g = dfs.get("Master_No_PLMN.parquet", pd.DataFrame())
    if not df_5g.empty:
        c_pay = get_col(df_5g, ['dlulpayload'])
        c_rrc = get_col(df_5g, ['rrcusermax'])
        c_avail = get_col(df_5g, ['cellavailability'])
        for col in [c_pay, c_rrc, c_avail]:
            if col in df_5g.columns:
                df_5g[col] = pd.to_numeric(df_5g[col], errors='coerce').fillna(0)

    return (
        dfs.get("Master_2GDaily.parquet", pd.DataFrame()),
        dfs.get("Master_4GDaily.parquet", pd.DataFrame()),
        df_5g,
        dfs.get("Master_4GBH.parquet", pd.DataFrame()),
        dfs.get("Master_LTE.parquet", pd.DataFrame()),
        dfs.get("Master_GSM.parquet", pd.DataFrame())
    )

with st.spinner("Downloading & Extracting Data..."):
    raw_2g, raw_4g, raw_5g, raw_4g_bh, raw_lte, raw_gsm = load_data()

gc.collect()

# ================= GLOBAL FILTER & MENU =================
st.sidebar.markdown("### 🧭 MENU NAVIGASI")
# LAZY LOADING: Hanya satu menu yang akan dihitung di memori dalam satu waktu
menu = st.sidebar.radio("Pilih Analisa:", [
    "📊 Komparasi MOCN", 
    "🏢 Komparasi Operator", 
    "📈 Trend Harian (MOCN)", 
    "⏳ Trend Busy Hour"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ GLOBAL FILTERS**")
mocn_filter = st.sidebar.radio("Select MOCN Status", ["All", "Pre MOCN (XL & SF)", "Post MOCN (XLSMART)"])

def apply_mocn_filter(df):
    if df.empty: return df
    op_col = get_col(df, ['operator', 'opr'])
    if op_col not in df.columns: return df
    df_op_clean = df[op_col].astype(str).str.lower().str.strip()
    if mocn_filter == "Pre MOCN (XL & SF)": return df[df_op_clean.isin(['xl', 'sf'])]
    elif mocn_filter == "Post MOCN (XLSMART)": return df[df_op_clean == 'xlsmart']
    return df

base_2g = apply_mocn_filter(raw_2g)
base_4g = apply_mocn_filter(raw_4g)
base_5g = apply_mocn_filter(raw_5g)
base_4g_bh = apply_mocn_filter(raw_4g_bh)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Filter Area & Site")

def apply_filter(df, col, selected_vals):
    if not selected_vals or df.empty or col not in df.columns: return df
    return df[df[col].isin(selected_vals)]

all_clusters = set()
for df in [base_2g, base_4g, base_5g, base_4g_bh]:
    c_clust = get_col(df, ['cluster', 'project_cluster'])
    if not df.empty and c_clust in df.columns: all_clusters.update(df[c_clust].dropna().unique())
selected_clusters = st.sidebar.multiselect("1. Select Cluster", sorted([c for c in all_clusters if str(c) != 'nan']))

f_2g = apply_filter(base_2g, get_col(base_2g, ['cluster', 'project_cluster']), selected_clusters)
f_4g = apply_filter(base_4g, get_col(base_4g, ['cluster', 'project_cluster']), selected_clusters)
f_5g = apply_filter(base_5g, get_col(base_5g, ['cluster', 'project_cluster']), selected_clusters)
f_4g_bh = apply_filter(base_4g_bh, get_col(base_4g_bh, ['cluster', 'project_cluster']), selected_clusters)

all_towers = set()
for df in [f_2g, f_4g, f_5g, f_4g_bh]:
    c_tow = get_col(df, ['towerid', 'siteid'])
    if not df.empty and c_tow in df.columns: all_towers.update(df[c_tow].dropna().unique())

selected_towers_ms = st.sidebar.multiselect("2. Select Site / Tower ID", sorted([t for t in all_towers if str(t) != 'nan']))
pasted_towers = st.sidebar.text_area("✏️ Or paste Site IDs:", height=68)
p_towers = [s.strip() for s in re.split(r'[,\n\t]+', pasted_towers) if s.strip()] if pasted_towers else []
selected_towers = list(set(selected_towers_ms + p_towers))

f_2g = apply_filter(f_2g, get_col(f_2g, ['towerid', 'siteid']), selected_towers)
f_4g = apply_filter(f_4g, get_col(f_4g, ['towerid', 'siteid']), selected_towers)
f_5g = apply_filter(f_5g, get_col(f_5g, ['towerid', 'siteid']), selected_towers)
f_4g_bh = apply_filter(f_4g_bh, get_col(f_4g_bh, ['towerid', 'siteid']), selected_towers)

all_tower_sectors = set()
for df in [f_2g, f_4g, f_5g, f_4g_bh]:
    c_ts = get_col(df, ['tower_sector'])
    if not df.empty and c_ts in df.columns: all_tower_sectors.update(df[c_ts].dropna().unique())
selected_tower_sectors = st.sidebar.multiselect("3. Select Tower Sector", sorted([s for s in all_tower_sectors if str(s) != 'nan']))

f_2g = apply_filter(f_2g, get_col(f_2g, ['tower_sector']), selected_tower_sectors)
f_4g = apply_filter(f_4g, get_col(f_4g, ['tower_sector']), selected_tower_sectors)
f_5g = apply_filter(f_5g, get_col(f_5g, ['tower_sector']), selected_tower_sectors)
f_4g_bh = apply_filter(f_4g_bh, get_col(f_4g_bh, ['tower_sector']), selected_tower_sectors)

all_cells = set()
for df in [f_2g, f_4g, f_5g, f_4g_bh]:
    c_cell = get_col(df, ['cellname', 'moentity'])
    if not df.empty and c_cell in df.columns: all_cells.update(df[c_cell].dropna().unique())

selected_cells_ms = st.sidebar.multiselect("4. Select Cell (MOEntity)", sorted([c for c in all_cells if str(c) != 'nan']))
pasted_cells = st.sidebar.text_area("✏️ Or paste Cell Names:", height=68)
p_cells = [s.strip() for s in re.split(r'[,\n\t]+', pasted_cells) if s.strip()] if pasted_cells else []
selected_cells = list(set(selected_cells_ms + p_cells))

f_2g = apply_filter(f_2g, get_col(f_2g, ['cellname', 'moentity']), selected_cells)
f_4g = apply_filter(f_4g, get_col(f_4g, ['cellname', 'moentity']), selected_cells)
f_5g = apply_filter(f_5g, get_col(f_5g, ['cellname', 'moentity']), selected_cells)
f_4g_bh = apply_filter(f_4g_bh, get_col(f_4g_bh, ['cellname', 'moentity']), selected_cells)

st.sidebar.markdown("---")

all_dates = []
for df in [f_2g, f_4g, f_5g, f_4g_bh]:
    if not df.empty and 'Date' in df.columns: all_dates.extend(df['Date'].dropna().unique())

if all_dates: 
    min_date, max_date = min(all_dates), max(all_dates)
    if min_date == max_date: max_date = min_date + timedelta(days=1)
else: 
    min_date, max_date = pd.Timestamp('today').date(), pd.Timestamp('today').date() + timedelta(days=1)


# ================= GENERAL CALCULATION FUNCTIONS (PURE BASELINE) =================
def get_sum(df_filtered, col_names):
    if df_filtered.empty: return 0
    col = get_col(df_filtered, col_names)
    if col not in df_filtered.columns: return 0
    return float(pd.to_numeric(df_filtered[col], errors='coerce').sum())

def get_mean(df_filtered, col_names):
    if df_filtered.empty: return 0
    col = get_col(df_filtered, col_names)
    if col not in df_filtered.columns: return 0
    val = pd.to_numeric(df_filtered[col], errors='coerce').mean()
    return float(val) if pd.notna(val) else 0

def get_op_sum(df, col_names, op_name):
    if df.empty: return 0
    op_col = get_col(df, ['operator', 'opr'])
    if op_col not in df.columns: return 0
    df_op = df[df[op_col].astype(str).str.lower().str.strip() == op_name]
    return get_sum(df_op, col_names)

def calc_delta(pre, post):
    if pre == 0 and post == 0: return 0
    if pre == 0: return 100.0
    return ((post - pre) / pre) * 100
    
def calc_delta_abs(pre, post): return post - pre
def color_delta(val): return 'color: green' if val > 0 else ('color: red' if val < 0 else '')


# ================= MENU 1: MOCN RAT LEVEL COMPARISON =================
if menu == "📊 Komparasi MOCN":
    st.header("📊 Pre vs Post Comparison - MOCN RAT Level")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        all_clusters_comp = set()
        for df in [raw_2g, raw_4g, raw_5g]: 
            c_clust = get_col(df, ['cluster', 'project_cluster'])
            if not df.empty and c_clust in df.columns: all_clusters_comp.update(df[c_clust].dropna().unique())
        selected_clusters_comp = st.multiselect("Select Cluster (MOCN)", sorted([c for c in all_clusters_comp if str(c) != 'nan']), key='cl_mocn')

    comp_2g_c = apply_filter(raw_2g, get_col(raw_2g, ['cluster', 'project_cluster']), selected_clusters_comp) if selected_clusters_comp else raw_2g
    comp_4g_c = apply_filter(raw_4g, get_col(raw_4g, ['cluster', 'project_cluster']), selected_clusters_comp) if selected_clusters_comp else raw_4g
    comp_5g_c = apply_filter(raw_5g, get_col(raw_5g, ['cluster', 'project_cluster']), selected_clusters_comp) if selected_clusters_comp else raw_5g

    with col_f2:
        all_towers_comp = set()
        for df in [comp_2g_c, comp_4g_c, comp_5g_c]:
            c_tow = get_col(df, ['towerid', 'siteid'])
            if not df.empty and c_tow in df.columns: all_towers_comp.update(df[c_tow].dropna().unique())
        
        selected_towers_comp_ms = st.multiselect("Select Site (MOCN)", sorted([t for t in all_towers_comp if str(t) != 'nan']), key='site_mocn_ms')
        pasted_towers_comp = st.text_area("✏️ Or paste Site IDs:", key='site_mocn_ta', height=68)
        p_towers_comp = [s.strip() for s in re.split(r'[,\n\t]+', pasted_towers_comp) if s.strip()] if pasted_towers_comp else []
        selected_towers_comp = list(set(selected_towers_comp_ms + p_towers_comp))

    comp_2g = apply_filter(comp_2g_c, get_col(comp_2g_c, ['towerid', 'siteid']), selected_towers_comp) if selected_towers_comp else comp_2g_c
    comp_4g = apply_filter(comp_4g_c, get_col(comp_4g_c, ['towerid', 'siteid']), selected_towers_comp) if selected_towers_comp else comp_4g_c
    comp_5g = apply_filter(comp_5g_c, get_col(comp_5g_c, ['towerid', 'siteid']), selected_towers_comp) if selected_towers_comp else comp_5g_c

    col_pre, col_post = st.columns(2)
    default_pre_end = min_date + timedelta(days=4) if max_date > min_date + timedelta(days=4) else max_date
    default_post_start = max_date - timedelta(days=4) if max_date > min_date + timedelta(days=4) else min_date

    with col_pre: pre_dates = st.date_input("📅 Select PRE Date Range", [min_date, default_pre_end], min_value=min_date, max_value=max_date, key='pre_mocn')
    with col_post: post_dates = st.date_input("📅 Select POST Date Range", [default_post_start, max_date], min_value=min_date, max_value=max_date, key='post_mocn')

    if len(pre_dates) == 2 and len(post_dates) == 2:
        pre_start, pre_end = pre_dates
        post_start, post_end = post_dates
        pre_days = (pre_end - pre_start).days + 1
        post_days = (post_end - post_start).days + 1

        pre_2g = comp_2g[(comp_2g['Date'] >= pre_start) & (comp_2g['Date'] <= pre_end)] if not comp_2g.empty else comp_2g
        pre_4g = comp_4g[(comp_4g['Date'] >= pre_start) & (comp_4g['Date'] <= pre_end)] if not comp_4g.empty else comp_4g
        pre_5g = comp_5g[(comp_5g['Date'] >= pre_start) & (comp_5g['Date'] <= pre_end)] if not comp_5g.empty else comp_5g

        post_2g = comp_2g[(comp_2g['Date'] >= post_start) & (comp_2g['Date'] <= post_end)] if not comp_2g.empty else comp_2g
        post_4g = comp_4g[(comp_4g['Date'] >= post_start) & (comp_4g['Date'] <= post_end)] if not comp_4g.empty else comp_4g
        post_5g = comp_5g[(comp_5g['Date'] >= post_start) & (comp_5g['Date'] <= post_end)] if not comp_5g.empty else comp_5g

        c_2g_traf = ['2g_tch traffic_kpi', 'tchtraffic']
        c_4g_pay  = ['totalpayloadgbkpi', '4g_total payload gb_kpi']
        c_4g_rrc  = ['connectedusermaxkpi']
        c_4g_volte= ['voltetrafficerlkpi']
        c_5g_pay  = ['dlulpayload']
        c_5g_rrc  = ['rrcusermax']
        c_2g_avail = ['2g_nav_kpi(%)', '2g_nav_kpi']
        c_4g_avail = ['navkpi', 'navexcludeprojectkpi', 'navincludeprojectkpi']
        c_5g_avail = ['cellavailability']

        # LOGIKA BASELINE (MURNI TANPA FILTER NAMA OPERATOR)
        pre_payload_4g = get_sum(pre_4g, c_4g_pay)
        pre_payload_5g = get_sum(pre_5g, c_5g_pay)
        pre_payload_total = pre_payload_4g + pre_payload_5g
        pre_rrc_4g = get_sum(pre_4g, c_4g_rrc)
        pre_rrc_5g = get_sum(pre_5g, c_5g_rrc)
        pre_rrc_total = pre_rrc_4g + pre_rrc_5g
        pre_voice_volte = get_sum(pre_4g, c_4g_volte)
        pre_voice_2g = get_sum(pre_2g, c_2g_traf)
        pre_voice_total = pre_voice_volte + pre_voice_2g
        pre_avail_4g = get_mean(pre_4g, c_4g_avail)
        pre_avail_5g = get_mean(pre_5g, c_5g_avail)
        pre_avail_2g = get_mean(pre_2g, c_2g_avail)

        post_payload_4g = get_sum(post_4g, c_4g_pay)
        post_payload_5g = get_sum(post_5g, c_5g_pay)
        post_payload_total = post_payload_4g + post_payload_5g
        post_rrc_4g = get_sum(post_4g, c_4g_rrc)
        post_rrc_5g = get_sum(post_5g, c_5g_rrc)
        post_rrc_total = post_rrc_4g + post_rrc_5g
        post_voice_volte = get_sum(post_4g, c_4g_volte)
        post_voice_2g = get_sum(post_2g, c_2g_traf)
        post_voice_total = post_voice_volte + post_voice_2g
        post_avail_4g = get_mean(post_4g, c_4g_avail)
        post_avail_5g = get_mean(post_5g, c_5g_avail)
        post_avail_2g = get_mean(post_2g, c_2g_avail)

        comp_data = {
            "KPI / Metric": [
                "Total Payload (4G+5G) [GB]", "├─ 4G Payload [GB]", "└─ 5G Payload [GB]", 
                "Total RRC User Max (4G+5G) [Sum]", "├─ 4G RRC User Max", "└─ 5G RRC User Max",
                "Total Voice Traffic (2G+4G) [Erl]", "├─ 4G VoLTE Traffic [Erl]", "└─ 2G Traffic [Erl]",
                "4G Network Availability [%]", "5G Network Availability [%]", "2G Network Availability [%]"
            ],
            f"Pre ({pre_days} Days)": [
                pre_payload_total, pre_payload_4g, pre_payload_5g, 
                pre_rrc_total, pre_rrc_4g, pre_rrc_5g,
                pre_voice_total, pre_voice_volte, pre_voice_2g,
                pre_avail_4g, pre_avail_5g, pre_avail_2g
            ],
            f"Post ({post_days} Days)": [
                post_payload_total, post_payload_4g, post_payload_5g, 
                post_rrc_total, post_rrc_4g, post_rrc_5g,
                post_voice_total, post_voice_volte, post_voice_2g,
                post_avail_4g, post_avail_5g, post_avail_2g
            ],
            "Delta (%)": [
                calc_delta(pre_payload_total, post_payload_total), calc_delta(pre_payload_4g, post_payload_4g), calc_delta(pre_payload_5g, post_payload_5g),
                calc_delta(pre_rrc_total, post_rrc_total), calc_delta(pre_rrc_4g, post_rrc_4g), calc_delta(pre_rrc_5g, post_rrc_5g),
                calc_delta(pre_voice_total, post_voice_total), calc_delta(pre_voice_volte, post_voice_volte), calc_delta(pre_voice_2g, post_voice_2g),
                calc_delta_abs(pre_avail_4g, post_avail_4g), calc_delta_abs(pre_avail_5g, post_avail_5g), calc_delta_abs(pre_avail_2g, post_avail_2g)
            ]
        }
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp.style.format({f"Pre ({pre_days} Days)": "{:,.2f}", f"Post ({post_days} Days)": "{:,.2f}", "Delta (%)": "{:,.2f} %"}).map(color_delta, subset=['Delta (%)']), width='stretch')
    else:
        st.warning("Silakan pilih rentang tanggal yang lengkap untuk Pre dan Post kalender.")


# ================= MENU 2: OPERATOR LEVEL =================
elif menu == "🏢 Komparasi Operator":
    st.header("🏢 Pre vs Post Comparison - Operator Level")

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        all_clusters_op = set()
        for df in [raw_lte, raw_gsm]:
            c_clust = get_col(df, ['cluster', 'project_cluster'])
            if not df.empty and c_clust in df.columns: all_clusters_op.update(df[c_clust].dropna().unique())
        selected_clusters_op = st.multiselect("Select Cluster (Operator)", sorted([c for c in all_clusters_op if str(c) != 'nan']), key='cl_op')

    comp_lte_c = apply_filter(raw_lte, get_col(raw_lte, ['cluster', 'project_cluster']), selected_clusters_op) if selected_clusters_op else raw_lte
    comp_gsm_c = apply_filter(raw_gsm, get_col(raw_gsm, ['cluster', 'project_cluster']), selected_clusters_op) if selected_clusters_op else raw_gsm

    with col_l2:
        all_towers_op = set()
        for df in [comp_lte_c, comp_gsm_c]:
            c_tow = get_col(df, ['towerid', 'siteid'])
            if not df.empty and c_tow in df.columns: all_towers_op.update(df[c_tow].dropna().unique())
        
        selected_towers_op_ms = st.multiselect("Select Site (Operator)", sorted([t for t in all_towers_op if str(t) != 'nan']), key='site_op_ms')
        pasted_towers_op = st.text_area("✏️ Or paste Site IDs:", key='site_op_ta', height=68)
        p_towers_op = [s.strip() for s in re.split(r'[,\n\t]+', pasted_towers_op) if s.strip()] if pasted_towers_op else []
        selected_towers_op = list(set(selected_towers_op_ms + p_towers_op))

    comp_lte = apply_filter(comp_lte_c, get_col(comp_lte_c, ['towerid', 'siteid']), selected_towers_op) if selected_towers_op else comp_lte_c
    comp_gsm = apply_filter(comp_gsm_c, get_col(comp_gsm_c, ['towerid', 'siteid']), selected_towers_op) if selected_towers_op else comp_gsm_c

    col_pre_op, col_post_op = st.columns(2)
    with col_pre_op: pre_dates_op = st.date_input("📅 PRE Date Range", [min_date, min_date + timedelta(days=4)], min_value=min_date, max_value=max_date, key='pre_op')
    with col_post_op: post_dates_op = st.date_input("📅 POST Date Range", [max_date - timedelta(days=4), max_date], min_value=min_date, max_value=max_date, key='post_op')

    if len(pre_dates_op) == 2 and len(post_dates_op) == 2:
        pre_start_o, pre_end_o = pre_dates_op
        post_start_o, post_end_o = post_dates_op
        pre_days_o = (pre_end_o - pre_start_o).days + 1
        post_days_o = (post_end_o - post_start_o).days + 1

        c_lte_pay  = ['totalpayloadgbkpi', '4g_total payload gb_kpi']
        c_lte_rrc  = ['connectedusermaxkpi']
        c_lte_volte= ['voltetrafficerlkpi']
        c_gsm_traf = ['tchtraffic', '2g_tch traffic_kpi']

        pre_lte_df = comp_lte[(comp_lte['Date'] >= pre_start_o) & (comp_lte['Date'] <= pre_end_o)] if not comp_lte.empty else comp_lte
        post_lte_df = comp_lte[(comp_lte['Date'] >= post_start_o) & (comp_lte['Date'] <= post_end_o)] if not comp_lte.empty else comp_lte
        pre_gsm_df = comp_gsm[(comp_gsm['Date'] >= pre_start_o) & (comp_gsm['Date'] <= pre_end_o)] if not comp_gsm.empty else comp_gsm
        post_gsm_df = comp_gsm[(comp_gsm['Date'] >= post_start_o) & (comp_gsm['Date'] <= post_end_o)] if not comp_gsm.empty else comp_gsm

        pre_pay_xl = get_op_sum(pre_lte_df, c_lte_pay, 'xl')
        pre_pay_sf = get_op_sum(pre_lte_df, c_lte_pay, 'sf')
        pre_pay_xls = get_op_sum(pre_lte_df, c_lte_pay, 'xlsmart')
        pre_pay_tot = pre_pay_xl + pre_pay_sf + pre_pay_xls

        pre_rrc_xl = get_op_sum(pre_lte_df, c_lte_rrc, 'xl')
        pre_rrc_sf = get_op_sum(pre_lte_df, c_lte_rrc, 'sf')
        pre_rrc_xls = get_op_sum(pre_lte_df, c_lte_rrc, 'xlsmart')
        pre_rrc_tot = pre_rrc_xl + pre_rrc_sf + pre_rrc_xls

        pre_volte_xl = get_op_sum(pre_lte_df, c_lte_volte, 'xl') + get_op_sum(pre_gsm_df, c_gsm_traf, 'xl')
        pre_volte_sf = get_op_sum(pre_lte_df, c_lte_volte, 'sf') + get_op_sum(pre_gsm_df, c_gsm_traf, 'sf')
        pre_volte_xls = get_op_sum(pre_lte_df, c_lte_volte, 'xlsmart') + get_op_sum(pre_gsm_df, c_gsm_traf, 'xlsmart')
        pre_volte_tot = pre_volte_xl + pre_volte_sf + pre_volte_xls

        post_pay_xl = get_op_sum(post_lte_df, c_lte_pay, 'xl')
        post_pay_sf = get_op_sum(post_lte_df, c_lte_pay, 'sf')
        post_pay_xls = get_op_sum(post_lte_df, c_lte_pay, 'xlsmart')
        post_pay_tot = post_pay_xl + post_pay_sf + post_pay_xls

        post_rrc_xl = get_op_sum(post_lte_df, c_lte_rrc, 'xl')
        post_rrc_sf = get_op_sum(post_lte_df, c_lte_rrc, 'sf')
        post_rrc_xls = get_op_sum(post_lte_df, c_lte_rrc, 'xlsmart')
        post_rrc_tot = post_rrc_xl + post_rrc_sf + post_rrc_xls

        post_volte_xl = get_op_sum(post_lte_df, c_lte_volte, 'xl') + get_op_sum(post_gsm_df, c_gsm_traf, 'xl')
        post_volte_sf = get_op_sum(post_lte_df, c_lte_volte, 'sf') + get_op_sum(post_gsm_df, c_gsm_traf, 'sf')
        post_volte_xls = get_op_sum(post_lte_df, c_lte_volte, 'xlsmart') + get_op_sum(post_gsm_df, c_gsm_traf, 'xlsmart')
        post_volte_tot = post_volte_xl + post_volte_sf + post_volte_xls

        comp_lte_data = {
            "KPI / Metric (Operator Level)": [
                "Total 4G Payload [GB]", "├─ XL 4G Payload", "├─ SF 4G Payload", "└─ XLSMART 4G Payload", 
                "Total 4G RRC User Max [Sum]", "├─ XL RRC User Max", "├─ SF RRC User Max", "└─ XLSMART RRC User Max",
                "Total Voice (VoLTE + 2G GSM) [Erl]", "├─ XL Voice Traffic", "├─ SF Voice Traffic", "└─ XLSMART Voice Traffic"
            ],
            f"Pre ({pre_days_o} Days)": [
                pre_pay_tot, pre_pay_xl, pre_pay_sf, pre_pay_xls,
                pre_rrc_tot, pre_rrc_xl, pre_rrc_sf, pre_rrc_xls,
                pre_volte_tot, pre_volte_xl, pre_volte_sf, pre_volte_xls
            ],
            f"Post ({post_days_o} Days)": [
                post_pay_tot, post_pay_xl, post_pay_sf, post_pay_xls,
                post_rrc_tot, post_rrc_xl, post_rrc_sf, post_rrc_xls,
                post_volte_tot, post_volte_xl, post_volte_sf, post_volte_xls
            ],
            "Delta (%)": [
                calc_delta(pre_pay_tot, post_pay_tot), calc_delta(pre_pay_xl, post_pay_xl), calc_delta(pre_pay_sf, post_pay_sf), calc_delta(pre_pay_xls, post_pay_xls),
                calc_delta(pre_rrc_tot, post_rrc_tot), calc_delta(pre_rrc_xl, post_rrc_xl), calc_delta(pre_rrc_sf, post_rrc_sf), calc_delta(pre_rrc_xls, post_rrc_xls),
                calc_delta(pre_volte_tot, post_volte_tot), calc_delta(pre_volte_xl, post_volte_xl), calc_delta(pre_volte_sf, post_volte_sf), calc_delta(pre_volte_xls, post_volte_xls)
            ]
        }
        df_comp_lte = pd.DataFrame(comp_lte_data)
        st.dataframe(df_comp_lte.style.format({f"Pre ({pre_days_o} Days)": "{:,.2f}", f"Post ({post_days_o} Days)": "{:,.2f}", "Delta (%)": "{:,.2f} %"}).map(color_delta, subset=['Delta (%)']), width='stretch')


# ================= MENU 3: TREND HARIAN =================
elif menu == "📈 Trend Harian (MOCN)":
    st.header("📅 Daily Performance (Area Stacked)")
    
    date_range = st.sidebar.date_input("5. Global Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    if len(date_range) == 2:
        start_d, end_d = date_range
        trend_days = (end_d - start_d).days + 1 
        chart_2g = f_2g[(f_2g['Date'] >= start_d) & (f_2g['Date'] <= end_d)] if not f_2g.empty else f_2g
        chart_4g = f_4g[(f_4g['Date'] >= start_d) & (f_4g['Date'] <= end_d)] if not f_4g.empty else f_4g
        chart_5g = f_5g[(f_5g['Date'] >= start_d) & (f_5g['Date'] <= end_d)] if not f_5g.empty else f_5g
        
        col_2g_traf = get_col(chart_2g, ['2g_tch traffic_kpi', 'tchtraffic'])
        col_2g_avail = get_col(chart_2g, ['2g_nav_kpi(%)', '2g_nav_kpi'])
        col_4g_pay  = get_col(chart_4g, ['totalpayloadgbkpi', '4g_total payload gb_kpi'])
        col_4g_rrc  = get_col(chart_4g, ['connectedusermaxkpi'])
        col_4g_volte= get_col(chart_4g, ['voltetrafficerlkpi'])
        col_4g_avail= get_col(chart_4g, ['navkpi', 'navexcludeprojectkpi', 'navincludeprojectkpi'])
        col_5g_pay  = get_col(chart_5g, ['dlulpayload'])
        col_5g_rrc  = get_col(chart_5g, ['rrcusermax'])
        col_5g_avail= get_col(chart_5g, ['cellavailability'])

        dict_2g = {}
        if col_2g_traf in chart_2g.columns: dict_2g[col_2g_traf] = 'sum'
        if col_2g_avail in chart_2g.columns: dict_2g[col_2g_avail] = 'mean'
        agg_2g = chart_2g.groupby('Date').agg(dict_2g).reset_index() if not chart_2g.empty and dict_2g else pd.DataFrame(columns=['Date'])

        dict_4g = {}
        for c in [col_4g_pay, col_4g_rrc, col_4g_volte]:
            if c in chart_4g.columns: dict_4g[c] = 'sum'
        if col_4g_avail in chart_4g.columns: dict_4g[col_4g_avail] = 'mean'
        agg_4g = chart_4g.groupby('Date').agg(dict_4g).reset_index() if not chart_4g.empty and dict_4g else pd.DataFrame(columns=['Date'])

        dict_5g = {}
        for c in [col_5g_pay, col_5g_rrc]:
            if c in chart_5g.columns: dict_5g[c] = 'sum'
        if col_5g_avail in chart_5g.columns: dict_5g[col_5g_avail] = 'mean'
        agg_5g = chart_5g.groupby('Date').agg(dict_5g).reset_index() if not chart_5g.empty and dict_5g else pd.DataFrame(columns=['Date'])

        df_trend = pd.merge(agg_4g, agg_5g, on='Date', how='outer')
        df_trend = pd.merge(df_trend, agg_2g, on='Date', how='outer').fillna(0)

        if not df_trend.empty:
            df_trend = df_trend.sort_values('Date')
            
            rename_dict = {}
            if col_4g_pay in df_trend.columns: rename_dict[col_4g_pay] = '4G Payload (GB)'
            if col_5g_pay in df_trend.columns: rename_dict[col_5g_pay] = '5G Payload (GB)'
            if col_4g_rrc in df_trend.columns: rename_dict[col_4g_rrc] = '4G RRC (User Max)'
            if col_5g_rrc in df_trend.columns: rename_dict[col_5g_rrc] = '5G RRC (User Max)'
            if col_4g_volte in df_trend.columns: rename_dict[col_4g_volte] = '4G VoLTE Traffic (Erl)'
            if col_2g_traf in df_trend.columns: rename_dict[col_2g_traf] = '2G Traffic (Erl)'
            if col_4g_avail in df_trend.columns: rename_dict[col_4g_avail] = '4G Availability (%)'
            if col_5g_avail in df_trend.columns: rename_dict[col_5g_avail] = '5G Availability (%)'
            if col_2g_avail in df_trend.columns: rename_dict[col_2g_avail] = '2G Availability (%)'
            
            df_trend.rename(columns=rename_dict, inplace=True)
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                if '4G Payload (GB)' in df_trend.columns and '5G Payload (GB)' in df_trend.columns:
                    fig_payload = px.area(df_trend, x='Date', y=['4G Payload (GB)', '5G Payload (GB)'], title='Daily Payload Trend', color_discrete_sequence=['#1f77b4', '#00cc96'])
                    st.plotly_chart(format_x_axis(fig_payload, trend_days), width='stretch', config=HD_CONFIG)
                if '4G VoLTE Traffic (Erl)' in df_trend.columns and '2G Traffic (Erl)' in df_trend.columns:
                    fig_traf = px.area(df_trend, x='Date', y=['4G VoLTE Traffic (Erl)', '2G Traffic (Erl)'], title='Daily Voice Traffic Trend', color_discrete_sequence=['#ef553b', '#636efa'])
                    st.plotly_chart(format_x_axis(fig_traf, trend_days), width='stretch', config=HD_CONFIG)
                    
            with col_d2:
                if '4G RRC (User Max)' in df_trend.columns and '5G RRC (User Max)' in df_trend.columns:
                    fig_rrc = px.area(df_trend, x='Date', y=['4G RRC (User Max)', '5G RRC (User Max)'], title='Daily RRC User Trend', color_discrete_sequence=['#ff7f0e', '#ab63fa'])
                    st.plotly_chart(format_x_axis(fig_rrc, trend_days), width='stretch', config=HD_CONFIG)
                    
                avail_cols = [c for c in ['4G Availability (%)', '5G Availability (%)', '2G Availability (%)'] if c in df_trend.columns]
                if avail_cols:
                    fig_avail = px.line(df_trend, x='Date', y=avail_cols, title='Daily Network Availability (%)', markers=True)
                    st.plotly_chart(format_x_axis(fig_avail, trend_days), width='stretch', config=HD_CONFIG)
        else:
            st.warning("Data trend kosong untuk tanggal tersebut.")


# ================= MENU 4: BUSY HOUR =================
elif menu == "⏳ Trend Busy Hour":
    st.header("⏳ Busy Hour Performance (4G Only)")
    
    date_range = st.sidebar.date_input("5. Global Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    if not f_4g_bh.empty and len(date_range) == 2:
        start_d, end_d = date_range
        trend_days = (end_d - start_d).days + 1 
        chart_4g_bh = f_4g_bh[(f_4g_bh['Date'] >= start_d) & (f_4g_bh['Date'] <= end_d)]
        
        b_dl_num = get_col(chart_4g_bh, ['4g_cell_downlink user throughput_num'])
        b_dl_den = get_col(chart_4g_bh, ['4g_cell_downlink user throughput_den'])
        b_ul_num = get_col(chart_4g_bh, ['4g_cell_uplink user throughput_num'])
        b_ul_den = get_col(chart_4g_bh, ['4g_cell_uplink user throughput_den'])
        b_pay    = get_col(chart_4g_bh, ['4g_total payload gb_kpi'])
        b_prb    = get_col(chart_4g_bh, ['dl prb', 'dlprbutil'])
        b_ta_num = get_col(chart_4g_bh, ['4g_average ta num_mpi'])
        b_ta_den = get_col(chart_4g_bh, ['4g_average ta den_mpi'])
        b_cqi_num= get_col(chart_4g_bh, ['4g_cell_average cqi_num'])
        b_cqi_den= get_col(chart_4g_bh, ['4g_cell_average cqi_den'])
        b_avail  = get_col(chart_4g_bh, ['4g_nav_kpi(%)', '4g_nav_kpi', 'navexcludeprojectkpi'])

        bh_agg_dict = {
            b_dl_num: 'sum', b_dl_den: 'sum', b_ul_num: 'sum', b_ul_den: 'sum',
            b_pay: 'sum', b_prb: 'mean', b_ta_num: 'sum', b_ta_den: 'sum',
            b_cqi_num: 'sum', b_cqi_den: 'sum'
        }
        if b_avail in chart_4g_bh.columns: bh_agg_dict[b_avail] = 'mean'

        bh_rename_dict = {b_pay: 'BH Payload (GB)', b_prb: 'BH DL PRB Util (%)'}
        if b_avail in chart_4g_bh.columns: bh_rename_dict[b_avail] = 'BH Availability (%)'

        valid_bh_agg = {k: v for k, v in bh_agg_dict.items() if k in chart_4g_bh.columns}

        if valid_bh_agg:
            agg_bh = chart_4g_bh.groupby('Date').agg(valid_bh_agg).reset_index()
            for col in [b_dl_num, b_dl_den, b_ul_num, b_ul_den, b_ta_num, b_ta_den, b_cqi_num, b_cqi_den]:
                if col not in agg_bh.columns: agg_bh[col] = 0

            agg_bh['BH DL User Thp (Mbps)'] = np.where(agg_bh[b_dl_den] > 0, agg_bh[b_dl_num] / agg_bh[b_dl_den], 0)
            agg_bh['BH UL User Thp (Mbps)'] = np.where(agg_bh[b_ul_den] > 0, agg_bh[b_ul_num] / agg_bh[b_ul_den], 0)
            agg_bh['BH Average TA'] = np.where(agg_bh[b_ta_den] > 0, agg_bh[b_ta_num] / agg_bh[b_ta_den], 0)
            agg_bh['BH Average CQI'] = np.where(agg_bh[b_cqi_den] > 0, agg_bh[b_cqi_num] / agg_bh[b_cqi_den], 0)
            
            agg_bh.rename(columns=bh_rename_dict, inplace=True)
            agg_bh = agg_bh.sort_values('Date')

            st.markdown("#### 1. Aggregated Performance")
            col_bh1, col_bh2 = st.columns(2)
            with col_bh1:
                fig_bh_dl = px.line(agg_bh, x='Date', y='BH DL User Thp (Mbps)', title='BH DL User Throughput', markers=True)
                fig_bh_dl.add_hline(y=3.0, line_dash="dash", line_color="green")
                st.plotly_chart(format_x_axis(fig_bh_dl, trend_days), width='stretch', config=HD_CONFIG)

                fig_bh_pay = px.line(agg_bh, x='Date', y='BH Payload (GB)', title='BH Payload', markers=True, color_discrete_sequence=['#2ca02c'])
                st.plotly_chart(format_x_axis(fig_bh_pay, trend_days), width='stretch', config=HD_CONFIG)
                
                fig_bh_ta = px.line(agg_bh, x='Date', y='BH Average TA', title='BH Average TA', markers=True, color_discrete_sequence=['#9467bd'])
                st.plotly_chart(format_x_axis(fig_bh_ta, trend_days), width='stretch', config=HD_CONFIG)

            with col_bh2:
                fig_bh_ul = px.line(agg_bh, x='Date', y='BH UL User Thp (Mbps)', title='BH UL User Throughput', markers=True, color_discrete_sequence=['#ff7f0e'])
                st.plotly_chart(format_x_axis(fig_bh_ul, trend_days), width='stretch', config=HD_CONFIG)

                fig_bh_prb = px.line(agg_bh, x='Date', y='BH DL PRB Util (%)', title='BH DL PRB Util (%)', markers=True, color_discrete_sequence=['#d62728'])
                st.plotly_chart(format_x_axis(fig_bh_prb, trend_days), width='stretch', config=HD_CONFIG)

                fig_bh_cqi = px.line(agg_bh, x='Date', y='BH Average CQI', title='BH Average CQI', markers=True, color_discrete_sequence=['#e377c2'])
                st.plotly_chart(format_x_axis(fig_bh_cqi, trend_days), width='stretch', config=HD_CONFIG)

            st.markdown("---")
            st.markdown("#### 2. Sector & Cell Analysis")
            c_ts_bh = get_col(chart_4g_bh, ['tower_sector'])
            unique_sectors = chart_4g_bh[c_ts_bh].nunique() if c_ts_bh in chart_4g_bh.columns else 0
            
            if unique_sectors > 30: st.info(f"Terdapat {unique_sectors} Tower_Sectors. Silakan gunakan filter di kiri agar lebih ringan (< 30 Sectors).")
            elif unique_sectors == 0: st.warning("Data Tower_Sector kosong.")
            else:
                agg_bh_ts = chart_4g_bh.groupby(['Date', c_ts_bh]).agg(valid_bh_agg).reset_index()
                agg_bh_ts.rename(columns={c_ts_bh: 'Tower_Sector'}, inplace=True)

                for col in [b_dl_num, b_dl_den, b_ul_num, b_ul_den, b_ta_num, b_ta_den, b_cqi_num, b_cqi_den]:
                    if col not in agg_bh_ts.columns: agg_bh_ts[col] = 0

                agg_bh_ts['BH DL User Thp (Mbps)'] = np.where(agg_bh_ts[b_dl_den] > 0, agg_bh_ts[b_dl_num] / agg_bh_ts[b_dl_den], 0)
                agg_bh_ts['BH UL User Thp (Mbps)'] = np.where(agg_bh_ts[b_ul_den] > 0, agg_bh_ts[b_ul_num] / agg_bh_ts[b_ul_den], 0)
                agg_bh_ts['BH Average TA'] = np.where(agg_bh_ts[b_ta_den] > 0, agg_bh_ts[b_ta_num] / agg_bh_ts[b_ta_den], 0)
                agg_bh_ts['BH Average CQI'] = np.where(agg_bh_ts[b_cqi_den] > 0, agg_bh_ts[b_cqi_num] / agg_bh_ts[b_cqi_den], 0)
                
                agg_bh_ts.rename(columns=bh_rename_dict, inplace=True)
                agg_bh_ts = agg_bh_ts.sort_values(['Date', 'Tower_Sector'])

                col_ts1, col_ts2 = st.columns(2)
                with col_ts1:
                    fig_ts_dl = px.line(agg_bh_ts, x='Date', y='BH DL User Thp (Mbps)', color='Tower_Sector', title='Sector Level - DL User Thp', markers=True)
                    st.plotly_chart(format_x_axis(fig_ts_dl, trend_days), width='stretch', config=HD_CONFIG)

                    fig_ts_pay = px.area(agg_bh_ts, x='Date', y='BH Payload (GB)', color='Tower_Sector', title='Sector Level - Payload')
                    st.plotly_chart(format_x_axis(fig_ts_pay, trend_days), width='stretch', config=HD_CONFIG)

                with col_ts2:
                    fig_ts_ul = px.line(agg_bh_ts, x='Date', y='BH UL User Thp (Mbps)', color='Tower_Sector', title='Sector Level - UL User Thp', markers=True)
                    st.plotly_chart(format_x_axis(fig_ts_ul, trend_days), width='stretch', config=HD_CONFIG)

                    fig_ts_prb = px.line(agg_bh_ts, x='Date', y='BH DL PRB Util (%)', color='Tower_Sector', title='Sector Level - DL PRB Util', markers=True)
                    st.plotly_chart(format_x_axis(fig_ts_prb, trend_days), width='stretch', config=HD_CONFIG)
