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
    "Master_No_PLMN.parquet": "1DXBXH_dDdEIZPSRqryNk0MUGWu-cX9YC",
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
    
    fig.update_xaxes(
        tickangle=-45, dtick=tick_ms, tickformat="%Y-%m-%d",
        tickfont=dict(color='#2c3e50', size=24), 
        showgrid=True, gridwidth=1, gridcolor='#e5e8e8', linecolor='#bdc3c7', linewidth=1
    ) 
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e8e8', linecolor='#bdc3c7', linewidth=1)
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#2c3e50'), margin=dict(l=10, r=10, t=40, b=80))
    return fig

def get_col(df, possible_names):
    if df.empty: return possible_names[0]
    for n in possible_names:
        if n in df.columns: return n
    return possible_names[0]

# ================= DATA LOADING FROM GOOGLE DRIVE =================
@st.cache_data(ttl=timedelta(hours=12)) 
def load_data():
    dfs = {}
    for filename, file_id in GDRIVE_FILE_IDS.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, filename, quiet=True)
            except Exception as e:
                st.error(f"Failed to download {filename}: {e}")
            
        try:
            df = pd.read_parquet(filename)
            if not df.empty and 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.date
            dfs[filename] = df
        except:
            dfs[filename] = pd.DataFrame()

    df_5g = dfs.get("Master_No_PLMN.parquet", pd.DataFrame())
    if not df_5g.empty:
        for col in ['dlulpayload', 'rrcusermax', 'cellavailability']:
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

with st.spinner("Downloading & Preparing Data from Cloud..."):
    raw_2g, raw_4g, raw_5g, raw_4g_bh, raw_lte, raw_gsm = load_data()

# ================= GLOBAL FILTER (MOCN ONLY) =================
st.sidebar.markdown("**⚙️ GLOBAL FILTERS**")
mocn_filter = st.sidebar.radio("Select MOCN Status", ["All", "Pre MOCN (XL & SF)", "Post MOCN (XLSMART)"])

def apply_mocn_filter(df):
    if df.empty or 'Operator' not in df.columns: return df
    df_op_clean = df['Operator'].astype(str).str.lower().str.strip()
    if mocn_filter == "Pre MOCN (XL & SF)": return df[df_op_clean.isin(['xl', 'sf'])]
    elif mocn_filter == "Post MOCN (XLSMART)": return df[df_op_clean == 'xlsmart']
    return df

base_2g = apply_mocn_filter(raw_2g)
base_4g = apply_mocn_filter(raw_4g)
base_5g = apply_mocn_filter(raw_5g)
base_4g_bh = apply_mocn_filter(raw_4g_bh)

st.sidebar.markdown("---")

# ================= PATH A: MAIN CHART FILTERS (SIDEBAR) =================
st.sidebar.header("🎯 Filter Panel (MOCN Trend Charts)")

def apply_filter(df, col, selected_vals):
    if not selected_vals or df.empty or col not in df.columns: return df
    return df[df[col].isin(selected_vals)]

all_clusters = set()
for df in [base_2g, base_4g, base_5g, base_4g_bh]:
    if not df.empty and 'Cluster' in df.columns: all_clusters.update(df['Cluster'].dropna().unique())
selected_clusters = st.sidebar.multiselect("1. Select Cluster", sorted([c for c in all_clusters if str(c) != 'nan']))

f_2g = apply_filter(base_2g, 'Cluster', selected_clusters)
f_4g = apply_filter(base_4g, 'Cluster', selected_clusters)
f_5g = apply_filter(base_5g, 'Cluster', selected_clusters)
f_4g_bh = apply_filter(base_4g_bh, 'Cluster', selected_clusters)

all_towers = set()
for df in [f_2g, f_4g, f_5g, f_4g_bh]:
    if not df.empty and 'TowerID' in df.columns: all_towers.update(df['TowerID'].dropna().unique())

selected_towers_ms = st.sidebar.multiselect("2. Select Site / Tower ID", sorted([t for t in all_towers if str(t) != 'nan']))
pasted_towers = st.sidebar.text_area("✏️ Or paste Site IDs (Comma / Excel paste):", height=68)
p_towers = [s.strip() for s in re.split(r'[,\n\t]+', pasted_towers) if s.strip()] if pasted_towers else []
selected_towers = list(set(selected_towers_ms + p_towers))

f_2g = apply_filter(f_2g, 'TowerID', selected_towers)
f_4g = apply_filter(f_4g, 'TowerID', selected_towers)
f_5g = apply_filter(f_5g, 'TowerID', selected_towers)
f_4g_bh = apply_filter(f_4g_bh, 'TowerID', selected_towers)

all_tower_sectors = set()
for df in [f_2g, f_4g, f_5g, f_4g_bh]:
    if not df.empty and 'Tower_Sector' in df.columns: all_tower_sectors.update(df['Tower_Sector'].dropna().unique())
selected_tower_sectors = st.sidebar.multiselect("3. Select Tower Sector", sorted([s for s in all_tower_sectors if str(s) != 'nan']))

f_2g = apply_filter(f_2g, 'Tower_Sector', selected_tower_sectors)
f_4g = apply_filter(f_4g, 'Tower_Sector', selected_tower_sectors)
f_5g = apply_filter(f_5g, 'Tower_Sector', selected_tower_sectors)
f_4g_bh = apply_filter(f_4g_bh, 'Tower_Sector', selected_tower_sectors)

all_cells = set()
for df in [f_2g, f_4g, f_5g, f_4g_bh]:
    if not df.empty and 'CellName' in df.columns: all_cells.update(df['CellName'].dropna().unique())

selected_cells_ms = st.sidebar.multiselect("4. Select Cell (MOEntity)", sorted([c for c in all_cells if str(c) != 'nan']))
pasted_cells = st.sidebar.text_area("✏️ Or paste Cell Names (Comma / Excel paste):", height=68)
p_cells = [s.strip() for s in re.split(r'[,\n\t]+', pasted_cells) if s.strip()] if pasted_cells else []
selected_cells = list(set(selected_cells_ms + p_cells))

f_2g = apply_filter(f_2g, 'CellName', selected_cells)
f_4g = apply_filter(f_4g, 'CellName', selected_cells)
f_5g = apply_filter(f_5g, 'CellName', selected_cells)
f_4g_bh = apply_filter(f_4g_bh, 'CellName', selected_cells)

st.sidebar.markdown("---")

all_dates = []
for df in [f_2g, f_4g, f_5g, f_4g_bh]:
    if not df.empty and 'Date' in df.columns: all_dates.extend(df['Date'].dropna().unique())

if all_dates: 
    min_date, max_date = min(all_dates), max(all_dates)
    if min_date == max_date: max_date = min_date + timedelta(days=1)
else: 
    min_date, max_date = pd.Timestamp('today').date(), pd.Timestamp('today').date() + timedelta(days=1)

date_range = st.sidebar.date_input("5. Global Date Range (Trend Charts)", [min_date, max_date], min_value=min_date, max_value=max_date)

if len(date_range) == 2:
    start_d, end_d = date_range
    trend_days = (end_d - start_d).days + 1 
    chart_2g = f_2g[(f_2g['Date'] >= start_d) & (f_2g['Date'] <= end_d)] if not f_2g.empty else f_2g
    chart_4g = f_4g[(f_4g['Date'] >= start_d) & (f_4g['Date'] <= end_d)] if not f_4g.empty else f_4g
    chart_5g = f_5g[(f_5g['Date'] >= start_d) & (f_5g['Date'] <= end_d)] if not f_5g.empty else f_5g
    chart_4g_bh = f_4g_bh[(f_4g_bh['Date'] >= start_d) & (f_4g_bh['Date'] <= end_d)] if not f_4g_bh.empty else f_4g_bh
else:
    trend_days = 10 
    chart_2g, chart_4g, chart_5g, chart_4g_bh = f_2g, f_4g, f_5g, f_4g_bh

# ================= GENERAL CALCULATION FUNCTIONS =================
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

def calc_delta(pre, post):
    if pre == 0 and post == 0: return 0
    if pre == 0: return 100.0
    return ((post - pre) / pre) * 100
    
def calc_delta_abs(pre, post): return post - pre
def color_delta(val): return 'color: green' if val > 0 else ('color: red' if val < 0 else '')

# ================= PATH B: PRE VS POST COMPARISON (MOCN RAT) =================
st.header("📊 Pre vs Post Comparison - MOCN RAT Level")

col_f1, col_f2 = st.columns(2)
with col_f1:
    all_clusters_comp = set()
    for df in [base_2g, base_4g, base_5g]: 
        if not df.empty and 'Cluster' in df.columns: all_clusters_comp.update(df['Cluster'].dropna().unique())
    selected_clusters_comp = st.multiselect("Select Cluster (MOCN Comparison)", sorted([c for c in all_clusters_comp if str(c) != 'nan']), key='cl_mocn')

comp_2g_c = apply_filter(base_2g, 'Cluster', selected_clusters_comp) if selected_clusters_comp else base_2g
comp_4g_c = apply_filter(base_4g, 'Cluster', selected_clusters_comp) if selected_clusters_comp else base_4g
comp_5g_c = apply_filter(base_5g, 'Cluster', selected_clusters_comp) if selected_clusters_comp else base_5g

with col_f2:
    all_towers_comp = set()
    for df in [comp_2g_c, comp_4g_c, comp_5g_c]:
        if not df.empty and 'TowerID' in df.columns: all_towers_comp.update(df['TowerID'].dropna().unique())
    
    selected_towers_comp_ms = st.multiselect("Select Site (MOCN Comparison)", sorted([t for t in all_towers_comp if str(t) != 'nan']), key='site_mocn_ms')
    pasted_towers_comp = st.text_area("✏️ Or paste Site IDs here (Comma / Excel paste):", key='site_mocn_ta', height=68)
    p_towers_comp = [s.strip() for s in re.split(r'[,\n\t]+', pasted_towers_comp) if s.strip()] if pasted_towers_comp else []
    selected_towers_comp = list(set(selected_towers_comp_ms + p_towers_comp))

comp_2g = apply_filter(comp_2g_c, 'TowerID', selected_towers_comp) if selected_towers_comp else comp_2g_c
comp_4g = apply_filter(comp_4g_c, 'TowerID', selected_towers_comp) if selected_towers_comp else comp_4g_c
comp_5g = apply_filter(comp_5g_c, 'TowerID', selected_towers_comp) if selected_towers_comp else comp_5g_c

col_pre, col_post = st.columns(2)
default_pre_end = min_date + timedelta(days=4) if max_date > min_date + timedelta(days=4) else max_date
default_post_start = max_date - timedelta(days=4) if max_date > min_date + timedelta(days=4) else min_date

with col_pre: pre_dates = st.date_input("📅 Select PRE Date Range (MOCN)", [min_date, default_pre_end], min_value=min_date, max_value=max_date, key='pre_mocn')
with col_post: post_dates = st.date_input("📅 Select POST Date Range (MOCN)", [default_post_start, max_date], min_value=min_date, max_value=max_date, key='post_mocn')

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
    c_4g_pay  = ['totalpayloadgbkpi']
    c_4g_rrc  = ['connectedusermaxkpi']
    c_4g_volte= ['voltetrafficerlkpi']
    c_5g_pay  = ['dlulpayload']
    c_5g_rrc  = ['rrcusermax']
    c_2g_avail = ['2g_nav_kpi(%)', '2g_nav_kpi']
    c_4g_avail = ['navkpi', 'navexcludeprojectkpi', 'navincludeprojectkpi']
    c_5g_avail = ['cellavailability']

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
    st.warning("Please select a complete date range for both Pre and Post calendars above.")

st.markdown("---")

# ================= PATH C: PRE VS POST COMPARISON & TREND (OPERATOR LEVEL) =================
st.header("🏢 Pre vs Post Comparison - Operator Level (LTE & GSM)")

col_l1, col_l2 = st.columns(2)
with col_l1:
    all_clusters_op = set()
    for df in [raw_lte, raw_gsm]:
        if not df.empty and 'Cluster' in df.columns: all_clusters_op.update(df['Cluster'].dropna().unique())
    selected_clusters_op = st.multiselect("Select Cluster (Operator Level)", sorted([c for c in all_clusters_op if str(c) != 'nan']), key='cl_op')

comp_lte_c = apply_filter(raw_lte, 'Cluster', selected_clusters_op) if selected_clusters_op else raw_lte
comp_gsm_c = apply_filter(raw_gsm, 'Cluster', selected_clusters_op) if selected_clusters_op else raw_gsm

with col_l2:
    all_towers_op = set()
    for df in [comp_lte_c, comp_gsm_c]:
        if not df.empty and 'TowerID' in df.columns: all_towers_op.update(df['TowerID'].dropna().unique())
    
    selected_towers_op_ms = st.multiselect("Select Site (Operator Level)", sorted([t for t in all_towers_op if str(t) != 'nan']), key='site_op_ms')
    pasted_towers_op = st.text_area("✏️ Or paste Site IDs here (Comma / Excel paste):", key='site_op_ta', height=68)
    p_towers_op = [s.strip() for s in re.split(r'[,\n\t]+', pasted_towers_op) if s.strip()] if pasted_towers_op else []
    selected_towers_op = list(set(selected_towers_op_ms + p_towers_op))

comp_lte = apply_filter(comp_lte_c, 'TowerID', selected_towers_op) if selected_towers_op else comp_lte_c
comp_gsm = apply_filter(comp_gsm_c, 'TowerID', selected_towers_op) if selected_towers_op else comp_gsm_c

col_pre_op, col_post_op = st.columns(2)
with col_pre_op: pre_dates_op = st.date_input("📅 Select PRE Date Range (Op Level)", [min_date, default_pre_end], min_value=min_date, max_value=max_date, key='pre_op')
with col_post_op: post_dates_op = st.date_input("📅 Select POST Date Range (Op Level)", [default_post_start, max_date], min_value=min_date, max_value=max_date, key='post_op')

if len(pre_dates_op) == 2 and len(post_dates_op) == 2:
    pre_start_o, pre_end_o = pre_dates_op
    post_start_o, post_end_o = post_dates_op
    pre_days_o = (pre_end_o - pre_start_o).days + 1
    post_days_o = (post_end_o - post_start_o).days + 1

    def get_op_sum(df, col_names, op_name):
        if df.empty or 'Operator' not in df.columns: return 0
        df_op = df[df['Operator'].astype(str).str.lower().str.strip() == op_name]
        return get_sum(df_op, col_names)

    c_lte_pay  = ['totalpayloadgbkpi']
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

# ----------------- CHART TREND KHUSUS OPERATOR LEVEL (DUAL AXIS XL vs SF) -----------------
st.markdown("#### 📈 Daily Trend - Operator Level (XL & SF Only)")

if len(date_range) == 2:
    start_d, end_d = date_range
    chart_lte = comp_lte[(comp_lte['Date'] >= start_d) & (comp_lte['Date'] <= end_d)] if not comp_lte.empty else comp_lte
    chart_gsm = comp_gsm[(comp_gsm['Date'] >= start_d) & (comp_gsm['Date'] <= end_d)] if not comp_gsm.empty else comp_gsm
else:
    chart_lte = comp_lte
    chart_gsm = comp_gsm

def plot_dual_axis(df, val_col, title_text, t_days):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    df_xl = df[df['Operator'] == 'XL']
    df_sf = df[df['Operator'] == 'SF']
    
    if not df_xl.empty:
        fig.add_trace(go.Scatter(x=df_xl['Date'], y=df_xl[val_col], mode='lines+markers', name='XL (Left Axis)', line=dict(color='#1f77b4')), secondary_y=False)
    if not df_sf.empty:
        fig.add_trace(go.Scatter(x=df_sf['Date'], y=df_sf[val_col], mode='lines+markers', name='SF (Right Axis)', line=dict(color='#ff7f0e')), secondary_y=True)
        
    fig.update_layout(title_text=title_text, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig = format_x_axis(fig, t_days)
    fig.update_yaxes(title_text="XL", showgrid=True, secondary_y=False)
    fig.update_yaxes(title_text="SF", showgrid=False, secondary_y=True)
    return fig

if not chart_lte.empty or not chart_gsm.empty:
    c_lte_pay = get_col(chart_lte, ['totalpayloadgbkpi'])
    c_lte_volte = get_col(chart_lte, ['voltetrafficerlkpi'])
    c_lte_rrc = get_col(chart_lte, ['connectedusermaxkpi'])
    c_lte_dlnum = get_col(chart_lte, ['celldluserthpnum'])
    c_lte_dlden = get_col(chart_lte, ['celldluserthpden'])
    c_gsm_traf = get_col(chart_gsm, ['tchtraffic', '2g_tch traffic_kpi'])
    
    if not chart_lte.empty:
        valid_lte_agg = {k: 'sum' for k in [c_lte_pay, c_lte_volte, c_lte_rrc, c_lte_dlnum, c_lte_dlden] if k in chart_lte.columns}
        if valid_lte_agg:
            agg_lte = chart_lte.groupby(['Date', 'Operator']).agg(valid_lte_agg).reset_index()
        else:
            agg_lte = pd.DataFrame(columns=['Date', 'Operator'])
            
        for c in [c_lte_dlnum, c_lte_dlden]:
            if c not in agg_lte.columns: agg_lte[c] = 0
            
        agg_lte['DL Thp (Mbps)'] = np.where(agg_lte[c_lte_dlden] > 0, agg_lte[c_lte_dlnum] / agg_lte[c_lte_dlden], 0)
    else:
        agg_lte = pd.DataFrame(columns=['Date', 'Operator', c_lte_pay, c_lte_volte, c_lte_rrc, 'DL Thp (Mbps)'])

    if not chart_gsm.empty:
        valid_gsm_agg = {c_gsm_traf: 'sum'} if c_gsm_traf in chart_gsm.columns else {}
        if valid_gsm_agg:
            agg_gsm = chart_gsm.groupby(['Date', 'Operator']).agg(valid_gsm_agg).reset_index()
        else:
            agg_gsm = pd.DataFrame(columns=['Date', 'Operator'])
    else:
        agg_gsm = pd.DataFrame(columns=['Date', 'Operator', c_gsm_traf])

    agg_op_trend = pd.merge(agg_lte, agg_gsm, on=['Date', 'Operator'], how='outer').fillna(0)
    
    if c_lte_volte in agg_op_trend.columns and c_gsm_traf in agg_op_trend.columns:
        agg_op_trend['VoLTE + 2G Traf (Erl)'] = agg_op_trend[c_lte_volte] + agg_op_trend[c_gsm_traf]
    
    if not agg_op_trend.empty:
        agg_op_trend['Operator'] = agg_op_trend['Operator'].astype(str).str.upper()
        agg_op_trend = agg_op_trend[agg_op_trend['Operator'].isin(['XL', 'SF'])]
    
    if agg_op_trend.empty:
        st.warning("XL and SF data are empty for this range.")
    else:
        agg_op_trend = agg_op_trend.sort_values(['Date', 'Operator'])

        col_op1, col_op2 = st.columns(2)
        with col_op1:
            if c_lte_pay in agg_op_trend.columns:
                fig_op_pay = plot_dual_axis(agg_op_trend, c_lte_pay, '1. Payload 4G (GB)', trend_days)
                st.plotly_chart(fig_op_pay, width='stretch', config=HD_CONFIG)
                
            if c_lte_rrc in agg_op_trend.columns:
                fig_op_rrc = plot_dual_axis(agg_op_trend, c_lte_rrc, '3. Max RRC User', trend_days)
                st.plotly_chart(fig_op_rrc, width='stretch', config=HD_CONFIG)
                
        with col_op2:
            if 'VoLTE + 2G Traf (Erl)' in agg_op_trend.columns:
                fig_op_traf = plot_dual_axis(agg_op_trend, 'VoLTE + 2G Traf (Erl)', '2. Voice (VoLTE + 2G) Traffic (Erl)', trend_days)
                st.plotly_chart(fig_op_traf, width='stretch', config=HD_CONFIG)
                
            if 'DL Thp (Mbps)' in agg_op_trend.columns:
                fig_op_thp = plot_dual_axis(agg_op_trend, 'DL Thp (Mbps)', '4. DL Throughput (Mbps)', trend_days)
                st.plotly_chart(fig_op_thp, width='stretch', config=HD_CONFIG)

st.markdown("---")

# ================= PART 2: DAILY AREA CHARTS (MOCN RAT) =================
st.header("📅 Daily Performance (Area Stacked) - MOCN RAT")

col_2g_traf = get_col(chart_2g, ['2g_tch traffic_kpi', 'tchtraffic'])
col_2g_avail = get_col(chart_2g, ['2g_nav_kpi(%)', '2g_nav_kpi'])
col_4g_pay  = get_col(chart_4g, ['totalpayloadgbkpi'])
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

if not df_trend.empty and len(date_range) == 2:
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
    st.warning("Daily data is not available.")

st.markdown("---")

# ================= PART 3 & 4: BUSY HOUR AGGREGATED & SECTOR =================
if not chart_4g_bh.empty and len(date_range) == 2:
    b_dl_num = get_col(chart_4g_bh, ['4g_cell_downlink user throughput_num'])
    b_dl_den = get_col(chart_4g_bh, ['4g_cell_downlink user throughput_den'])
    b_ul_num = get_col(chart_4g_bh, ['4g_cell_uplink user throughput_num'])
    b_ul_den = get_col(chart_4g_bh, ['4g_cell_uplink user throughput_den'])
    b_pay    = get_col(chart_4g_bh, ['4g_total payload gb_kpi'])
    b_prb    = get_col(chart_4g_bh, ['dl prb'])
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

    st.header("⏳ Busy Hour Performance (Aggregated)")
    
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

        col_bh1, col_bh2 = st.columns(2)
        with col_bh1:
            fig_bh_dl = px.line(agg_bh, x='Date', y='BH DL User Thp (Mbps)', title='1. BH DL User Throughput (Mbps)', markers=True)
            fig_bh_dl.add_hline(y=3.0, line_dash="dash", line_color="green", annotation_text="Target 3 Mbps")
            fig_bh_dl.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Limit 1.5 Mbps")
            st.plotly_chart(format_x_axis(fig_bh_dl, trend_days), width='stretch', config=HD_CONFIG)

            fig_bh_pay = px.line(agg_bh, x='Date', y='BH Payload (GB)', title='3. Busy Hour Payload (GB)', markers=True, color_discrete_sequence=['#2ca02c'])
            st.plotly_chart(format_x_axis(fig_bh_pay, trend_days), width='stretch', config=HD_CONFIG)
            
            fig_bh_ta = px.line(agg_bh, x='Date', y='BH Average TA', title='5. BH Average TA', markers=True, color_discrete_sequence=['#9467bd'])
            st.plotly_chart(format_x_axis(fig_bh_ta, trend_days), width='stretch', config=HD_CONFIG)

            if 'BH Availability (%)' in agg_bh.columns:
                fig_bh_avail = px.line(agg_bh, x='Date', y='BH Availability (%)', title='7. BH Availability (%)', markers=True, color_discrete_sequence=['#17becf'])
                st.plotly_chart(format_x_axis(fig_bh_avail, trend_days), width='stretch', config=HD_CONFIG)

        with col_bh2:
            fig_bh_ul = px.line(agg_bh, x='Date', y='BH UL User Thp (Mbps)', title='2. BH UL User Throughput (Mbps)', markers=True, color_discrete_sequence=['#ff7f0e'])
            st.plotly_chart(format_x_axis(fig_bh_ul, trend_days), width='stretch', config=HD_CONFIG)

            fig_bh_prb = px.line(agg_bh, x='Date', y='BH DL PRB Util (%)', title='4. BH DL PRB Utilization (%)', markers=True, color_discrete_sequence=['#d62728'])
            st.plotly_chart(format_x_axis(fig_bh_prb, trend_days), width='stretch', config=HD_CONFIG)

            fig_bh_cqi = px.line(agg_bh, x='Date', y='BH Average CQI', title='6. BH Average CQI', markers=True, color_discrete_sequence=['#e377c2'])
            st.plotly_chart(format_x_axis(fig_bh_cqi, trend_days), width='stretch', config=HD_CONFIG)

    st.markdown("---")

    # ================= PART 4: BUSY HOUR TOWER SECTOR =================
    st.header("🏢 Busy Hour - Tower Sector Level Analysis")
    unique_sectors = chart_4g_bh['Tower_Sector'].nunique()
    
    if unique_sectors > 30: st.info(f"There are {unique_sectors} Tower_Sectors selected. Please filter specifically (Maximum 30 Sectors).")
    elif unique_sectors == 0: st.warning("No matching Tower_Sector.")
    elif valid_bh_agg:
        agg_bh_ts = chart_4g_bh.groupby(['Date', 'Tower_Sector']).agg(valid_bh_agg).reset_index()

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
            fig_ts_dl = px.line(agg_bh_ts, x='Date', y='BH DL User Thp (Mbps)', color='Tower_Sector', title='1. Sector Level - DL User Thp (Mbps)', markers=True)
            fig_ts_dl.add_hline(y=3.0, line_dash="dash", line_color="green")
            fig_ts_dl.add_hline(y=1.5, line_dash="dash", line_color="red")
            fig_ts_dl.update_layout(height=700)
            st.plotly_chart(format_x_axis(fig_ts_dl, trend_days), width='stretch', config=HD_CONFIG)

            fig_ts_pay = px.area(agg_bh_ts, x='Date', y='BH Payload (GB)', color='Tower_Sector', title='3. Sector Level - Payload (GB) [Stacked Area]')
            fig_ts_pay.update_layout(height=700)
            st.plotly_chart(format_x_axis(fig_ts_pay, trend_days), width='stretch', config=HD_CONFIG)
            
            fig_ts_ta = px.line(agg_bh_ts, x='Date', y='BH Average TA', color='Tower_Sector', title='5. Sector Level - Average TA', markers=True)
            fig_ts_ta.update_layout(height=700)
            st.plotly_chart(format_x_axis(fig_ts_ta, trend_days), width='stretch', config=HD_CONFIG)
            
            if 'BH Availability (%)' in agg_bh_ts.columns:
                fig_ts_avail = px.line(agg_bh_ts, x='Date', y='BH Availability (%)', color='Tower_Sector', title='7. Sector Level - Availability (%)', markers=True)
                fig_ts_avail.update_layout(height=700)
                st.plotly_chart(format_x_axis(fig_ts_avail, trend_days), width='stretch', config=HD_CONFIG)

        with col_ts2:
            fig_ts_ul = px.line(agg_bh_ts, x='Date', y='BH UL User Thp (Mbps)', color='Tower_Sector', title='2. Sector Level - UL User Thp (Mbps)', markers=True)
            fig_ts_ul.update_layout(height=700)
            st.plotly_chart(format_x_axis(fig_ts_ul, trend_days), width='stretch', config=HD_CONFIG)

            fig_ts_prb = px.line(agg_bh_ts, x='Date', y='BH DL PRB Util (%)', color='Tower_Sector', title='4. Sector Level - DL PRB Util (%)', markers=True)
            fig_ts_prb.update_layout(height=700)
            st.plotly_chart(format_x_axis(fig_ts_prb, trend_days), width='stretch', config=HD_CONFIG)
            
            fig_ts_cqi = px.line(agg_bh_ts, x='Date', y='BH Average CQI', color='Tower_Sector', title='6. Sector Level - Average CQI', markers=True)
            fig_ts_cqi.update_layout(height=700)
            st.plotly_chart(format_x_axis(fig_ts_cqi, trend_days), width='stretch', config=HD_CONFIG)

    st.markdown("---")

    # ================= PART 5: BUSY HOUR CELL LEVEL CHARTS =================
    st.header("🔬 Busy Hour - Cell Level Analysis")
    unique_cells = chart_4g_bh['CellName'].nunique()
    
    if unique_cells > 30: st.info(f"There are {unique_cells} cells selected. Please filter specifically (Maximum 30 Cells).")
    elif unique_cells == 0: st.warning("No matching cell.")
    elif valid_bh_agg:
        agg_bh_cell = chart_4g_bh.groupby(['Date', 'Tower_Sector', 'CellName']).agg(valid_bh_agg).reset_index()

        for col in [b_dl_num, b_dl_den, b_ul_num, b_ul_den, b_ta_num, b_ta_den, b_cqi_num, b_cqi_den]:
            if col not in agg_bh_cell.columns: agg_bh_cell[col] = 0

        agg_bh_cell['BH DL User Thp (Mbps)'] = np.where(agg_bh_cell[b_dl_den] > 0, agg_bh_cell[b_dl_num] / agg_bh_cell[b_dl_den], 0)
        agg_bh_cell['BH UL User Thp (Mbps)'] = np.where(agg_bh_cell[b_ul_den] > 0, agg_bh_cell[b_ul_num] / agg_bh_cell[b_ul_den], 0)
        agg_bh_cell['BH Average TA'] = np.where(agg_bh_cell[b_ta_den] > 0, agg_bh_cell[b_ta_num] / agg_bh_cell[b_ta_den], 0)
        agg_bh_cell['BH Average CQI'] = np.where(agg_bh_cell[b_cqi_den] > 0, agg_bh_cell[b_cqi_num] / agg_bh_cell[b_cqi_den], 0)
        
        agg_bh_cell.rename(columns=bh_rename_dict, inplace=True)
        agg_bh_cell = agg_bh_cell.sort_values(['Date', 'Tower_Sector', 'CellName'])

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_cell_dl = px.line(agg_bh_cell, x='Date', y='BH DL User Thp (Mbps)', color='CellName', hover_data=['Tower_Sector'], title='1. Cell Level - DL User Thp (Mbps)', markers=True)
            fig_cell_dl.add_hline(y=3.0, line_dash="dash", line_color="green")
            fig_cell_dl.add_hline(y=1.5, line_dash="dash", line_color="red")
            fig_cell_dl.update_layout(height=700)
            st.plotly_chart(format_x_axis(fig_cell_dl, trend_days), width='stretch', config=HD_CONFIG)

            fig_cell_pay = px.area(agg_bh_cell, x='Date', y='BH Payload (GB)', color='CellName', hover_data=['Tower_Sector'], title='3. Cell Level - Payload (GB) [Stacked Area]')
            fig_cell_pay.update_layout(height=700)
            st.plotly_chart(format_x_axis(fig_cell_pay, trend_days), width='stretch', config=HD_CONFIG)
            
            fig_cell_ta = px.line(agg_bh_cell, x='Date', y='BH Average TA', color='CellName', hover_data=['Tower_Sector'], title='5. Cell Level - Average TA', markers=True)
            fig_cell_ta.update_layout(height=700)
            st.plotly_chart(format_x_axis(fig_cell_ta, trend_days), width='stretch', config=HD_CONFIG)
            
            if 'BH Availability (%)' in agg_bh_cell.columns:
                fig_cell_avail = px.line(agg_bh_cell, x='Date', y='BH Availability (%)', color='CellName', hover_data=['Tower_Sector'], title='7. Cell Level - Availability (%)', markers=True)
                fig_cell_avail.update_layout(height=700)
                st.plotly_chart(format_x_axis(fig_cell_avail, trend_days), width='stretch', config=HD_CONFIG)

        with col_c2:
            fig_cell_ul = px.line(agg_bh_cell, x='Date', y='BH UL User Thp (Mbps)', color='CellName', hover_data=['Tower_Sector'], title='2. Cell Level - UL User Thp (Mbps)', markers=True)
            fig_cell_ul.update_layout(height=700)
            st.plotly_chart(format_x_axis(fig_cell_ul, trend_days), width='stretch', config=HD_CONFIG)

            fig_cell_prb = px.line(agg_bh_cell, x='Date', y='BH DL PRB Util (%)', color='CellName', hover_data=['Tower_Sector'], title='4. Cell Level - DL PRB Util (%)', markers=True)
            fig_cell_prb.update_layout(height=700)
            st.plotly_chart(format_x_axis(fig_cell_prb, trend_days), width='stretch', config=HD_CONFIG)
            
            fig_cell_cqi = px.line(agg_bh_cell, x='Date', y='BH Average CQI', color='CellName', hover_data=['Tower_Sector'], title='6. Cell Level - Average CQI', markers=True)
            fig_cell_cqi.update_layout(height=700)
            st.plotly_chart(format_x_axis(fig_cell_cqi, trend_days), width='stretch', config=HD_CONFIG)
