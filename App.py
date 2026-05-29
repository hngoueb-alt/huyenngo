# -*- coding: utf-8 -*-
"""
AIDEOM-VN — AI-Driven Decision Optimization Model for Vietnam
Web app giải 12 bài toán mô hình ra quyết định phát triển kinh tế Việt Nam
trong kỉ nguyên AI — dữ liệu thực 2020-2025.

Họ và tên : Ngô Khánh Huyền
Mã sinh viên: 23051259
Bài tập lớn: Các mô hình ra quyết định

Chạy:  streamlit run App.py
"""
import os, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="AIDEOM-VN",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg: #060d1c;
    --bg2: #0b1528;
    --bg3: #0f1e36;
    --border: #1a2f50;
    --accent: #f472b6;
    --accent2: #a855f7;
    --txt: #d8e4f0;
    --txt2: #8ba0b8;
    --txt3: #4a6178;
}

html, body, .stApp { background: var(--bg) !important; font-family: 'Be Vietnam Pro', sans-serif; }

/* Chữ toàn app */
.stApp, .stApp p, .stApp span, .stApp li, .stApp label,
.stMarkdown, .stMarkdown p, .stMarkdown li, [data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p, [data-testid="stText"] {
    color: var(--txt) !important;
}
.stApp h1 { color: #ffffff !important; font-size: 2rem !important; font-weight: 800 !important; }
.stApp h2 { color: #e8f0fb !important; font-size: 1.45rem !important; font-weight: 700 !important; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
.stApp h3 { color: #c5d8ee !important; font-size: 1.15rem !important; font-weight: 600 !important; }
.stApp h4 { color: #b0cce8 !important; font-size: 1rem !important; font-weight: 600 !important; }
.stCaption, div[data-testid="stCaptionContainer"] p { color: var(--txt2) !important; font-size: 0.82rem !important; }
.katex, .katex * { color: #e2d9f3 !important; }
[data-testid="stDataFrame"] * { color: var(--txt) !important; background: transparent !important; }

/* KPI Cards */
.kpi-box {
    background: linear-gradient(135deg, #0d1e35 0%, #0f2642 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.kpi-box::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 14px 14px 0 0;
}
.kpi-label { color: #7fa3c4 !important; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
.kpi-value { color: var(--accent) !important; font-size: 1.75rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1.1; }
.kpi-delta { display: inline-block; background: rgba(244,114,182,0.12); color: var(--accent) !important; border-radius: 6px; padding: 2px 8px; font-size: 0.72rem; font-weight: 700; margin-top: 4px; border: 1px solid rgba(244,114,182,0.25); }

/* Section header */
.section-hdr {
    background: linear-gradient(90deg, rgba(244,114,182,0.08), transparent);
    border-left: 4px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 12px 18px;
    margin-bottom: 20px;
}
.section-hdr h2 { border-bottom: none !important; margin: 0 !important; padding: 0 !important; }
.section-sub { color: var(--txt2) !important; font-size: 0.85rem !important; margin-top: 4px; }

/* Note boxes */
.note {
    background: rgba(168,85,247,0.07);
    border-left: 4px solid var(--accent2);
    padding: 12px 16px;
    border-radius: 0 10px 10px 0;
    color: var(--txt) !important;
    font-size: 0.88rem;
    line-height: 1.65;
    margin: 12px 0;
}
.note b { color: #ffffff !important; }
.note-green {
    background: rgba(244,114,182,0.07);
    border-left: 4px solid var(--accent);
    padding: 12px 16px;
    border-radius: 0 10px 10px 0;
    color: var(--txt) !important;
    font-size: 0.88rem;
    line-height: 1.65;
    margin: 12px 0;
}
.note-green b { color: #ffffff !important; }

/* Level chips */
.chip { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; margin-right: 8px; }

/* Formula block */
.formula-block {
    background: rgba(168,85,247,0.06);
    border: 1px solid rgba(168,85,247,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #c4b5fd !important;
    line-height: 1.8;
    margin: 12px 0 18px;
}

/* Sidebar */
[data-testid="stSidebar"] { background: #080e1d !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--txt) !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.87rem !important; padding: 4px 0 !important; }
.sidebar-card {
    background: #0d1929;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    font-size: 0.82rem;
    line-height: 1.7;
    color: var(--txt) !important;
}
.sidebar-card b { color: #ffffff !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 3px; background: transparent; }
.stTabs [data-baseweb="tab"] { background: #0c1929; border-radius: 8px 8px 0 0; padding: 8px 16px; border: 1px solid var(--border); border-bottom: none; }
.stTabs [data-baseweb="tab"] p { color: #7fa3c4 !important; font-weight: 600; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background: #0f2645 !important; border-color: var(--accent) !important; }
.stTabs [aria-selected="true"] p { color: var(--accent) !important; }
.stTabs [data-baseweb="tab-panel"] { background: #080e1d; border: 1px solid var(--border); border-radius: 0 10px 10px 10px; padding: 20px; }

/* Sliders & inputs */
.stSlider label, .stRadio label, .stSelectbox label { color: var(--txt) !important; font-size: 0.88rem !important; }
.stSlider [data-baseweb="slider"] [role="slider"] { background: var(--accent) !important; }

/* Expander */
.stExpander { border: 1px solid var(--border) !important; border-radius: 10px !important; background: var(--bg2) !important; }
.stExpander summary { color: var(--txt) !important; }

/* Metric */
[data-testid="stMetric"] { background: var(--bg3); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
[data-testid="stMetricLabel"] { color: var(--txt2) !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'JetBrains Mono', monospace !important; }

/* Divider */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* Success / Error */
.stSuccess { background: rgba(244,114,182,0.1) !important; border: 1px solid rgba(244,114,182,0.3) !important; border-radius: 8px !important; color: #fce7f3 !important; }
.stError { background: rgba(239,68,68,0.1) !important; border: 1px solid rgba(239,68,68,0.3) !important; border-radius: 8px !important; }
.stWarning { background: rgba(251,191,36,0.1) !important; border: 1px solid rgba(251,191,36,0.3) !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

PLOT_TMPL = "plotly_dark"
PALETTE = ["#f472b6","#a855f7","#f87171","#fbbf24","#38bdf8","#fb923c","#34d399","#e879f9","#94a3b8","#a3e635"]
P_CLRS = dict(zip(
    ['s1 Lạc quan','s2 Cơ sở','s3 Bi quan','s4 Khủng hoảng'],
    ['#4ade80','#60a5fa','#fb923c','#f87171']
))

def plotly_cfg(fig, h=380, title=None, xtitle=None, ytitle=None):
    fig.update_layout(
        template=PLOT_TMPL,
        height=h,
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(11,21,40,0.6)',
        font=dict(family='Be Vietnam Pro, sans-serif', size=12, color='#d8e4f0'),
        title_font=dict(size=14, color='#e8f0fb', family='Be Vietnam Pro'),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a2f50', borderwidth=1),
        xaxis=dict(gridcolor='#1a2f50', linecolor='#1a2f50'),
        yaxis=dict(gridcolor='#1a2f50', linecolor='#1a2f50'),
        margin=dict(t=50 if title else 20, b=40, l=40, r=20),
    )
    return fig

# ============================================================
# DỮ LIỆU — đọc từ CSV upload hoặc tạo mặc định
# ============================================================
def _write_csvs():
    if not os.path.exists("vietnam_macro_2020_2025.csv"):
        pd.DataFrame({
            "year":[2020,2021,2022,2023,2024,2025],
            "GDP_trillion_VND":[8044.4,8487.5,9513.3,10221.8,11511.9,12847.6],
            "GDP_growth_pct":[2.91,2.58,8.02,5.05,7.09,8.02],
            "GDP_billion_USD":[346.6,366.1,408.8,430.0,476.3,514.0],
            "population_million":[97.58,98.51,99.46,100.30,101.30,102.30],
            "digital_economy_share_GDP_pct":[12.0,12.7,14.3,16.5,18.3,19.5],
            "labor_productivity_million_VND":[151.2,171.3,188.1,199.3,221.9,245.0],
            "FDI_disbursed_billion_USD":[19.98,19.74,22.40,23.18,25.35,27.60],
            "exports_billion_USD":[282.6,336.3,371.3,355.5,405.5,475.0],
            "imports_billion_USD":[262.7,332.2,358.9,327.5,380.8,455.0],
            "inflation_cpi_pct":[3.23,1.84,3.15,3.25,3.63,3.31],
        }).to_csv("vietnam_macro_2020_2025.csv",index=False)

    if not os.path.exists("vietnam_sectors_2024.csv"):
        pd.DataFrame({
            "sector_id":list(range(1,11)),
            "sector_name_en":["Agriculture-Forestry-Fishery","Manufacturing","Construction","Mining",
                              "Wholesale-Retail","Finance-Banking-Insurance","Logistics-Transport-Warehousing",
                              "Information-Communication-IT","Education-Training","Healthcare"],
            "gdp_share_2024_pct":[11.86,24.10,7.04,3.36,9.85,5.12,5.45,3.85,3.85,2.85],
            "growth_rate_2024_pct":[3.27,9.64,7.45,-1.20,7.10,7.36,9.93,7.85,6.42,6.85],
            "labor_million":[13.20,11.50,4.80,0.30,7.80,0.55,1.95,0.62,2.15,0.75],
            "export_billion_USD":[40.5,290.9,2.5,8.2,5.5,1.2,3.1,178.0,0.0,0.0],
            "digital_index_0_100":[28,68,35,50,72,82,65,92,55,58],
            "ai_readiness_0_100":[15,55,20,30,48,72,42,88,38,45],
            "fdi_attraction_billion_USD":[2.1,18.6,0.8,0.5,3.2,1.8,2.4,4.6,0.4,1.2],
            "spillover_coef_0_1":[0.35,0.78,0.42,0.30,0.55,0.85,0.72,0.92,0.65,0.60],
            "automation_risk_pct":[18,42,25,55,38,52,35,28,22,18],
            "rd_intensity_pct":[0.15,0.62,0.18,0.22,0.10,0.45,0.20,1.20,0.30,0.55],
        }).to_csv("vietnam_sectors_2024.csv",index=False)

    if not os.path.exists("vietnam_regions_2024.csv"):
        pd.DataFrame({
            "region_id":list(range(1,7)),
            "region_name_en":["Northern Midlands and Mountains","Red River Delta",
                              "North Central and South Central Coast","Central Highlands",
                              "Southeast","Mekong Delta"],
            "population_million":[14.2,23.5,20.8,6.1,19.2,17.5],
            "grdp_trillion_VND":[810,3580,1820,420,3050,1409],
            "grdp_growth_pct":[8.5,7.9,6.85,7.2,7.5,7.3],
            "grdp_per_capita_million_VND":[57.0,152.3,87.5,68.9,158.9,80.5],
            "fdi_registered_billion_USD":[3.5,20.0,8.2,0.8,18.5,2.1],
            "exports_billion_USD":[42.5,132.0,68.5,2.8,128.5,25.7],
            "digital_index_0_100":[38,78,55,32,82,48],
            "ai_readiness_0_100":[22,68,40,18,75,30],
            "trained_labor_pct":[21.5,36.8,27.5,18.2,42.5,16.8],
            "gini_coef":[0.405,0.358,0.372,0.412,0.385,0.392],
            "rd_intensity_pct":[0.18,0.85,0.32,0.15,0.78,0.22],
            "internet_penetration_pct":[72,92,84,68,94,78],
        }).to_csv("vietnam_regions_2024.csv",index=False)

@st.cache_data
def load_data():
    _write_csvs()
    macro   = pd.read_csv("vietnam_macro_2020_2025.csv").sort_values("year").reset_index(drop=True)
    sectors = pd.read_csv("vietnam_sectors_2024.csv")
    regions = pd.read_csv("vietnam_regions_2024.csv")
    return macro, sectors, regions

MACRO, SECTORS, REGIONS = load_data()

# Hằng số từ đề bài PDF
K_HIST  = np.array([16500.,17800.,19600.,21300.,23500.,25900.])
L_HIST  = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
D_HIST  = MACRO["digital_economy_share_GDP_pct"].values.astype(float)
AI_HIST = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
H_HIST  = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])
Y_HIST  = MACRO["GDP_trillion_VND"].values.astype(float)
COEF    = dict(alpha=0.33, beta=0.42, gamma=0.10, delta=0.08, theta=0.07)

REGIONS_VI = ['Trung du MN Bắc','ĐB sông Hồng','BTB+DH Trung Bộ','Tây Nguyên','Đông Nam Bộ','ĐB sông CL']
REG        = ['NMM','RRD','NCC','CH','SE','MD']
ITEMS      = ['I','D','AI','H']

BETA_RJ = {
    ('NMM','I'):1.15,('NMM','D'):0.85,('NMM','AI'):0.55,('NMM','H'):1.30,
    ('RRD','I'):0.95,('RRD','D'):1.25,('RRD','AI'):1.40,('RRD','H'):1.05,
    ('NCC','I'):1.05,('NCC','D'):0.95,('NCC','AI'):0.85,('NCC','H'):1.15,
    ('CH', 'I'):1.20,('CH', 'D'):0.75,('CH', 'AI'):0.45,('CH', 'H'):1.35,
    ('SE', 'I'):0.90,('SE', 'D'):1.30,('SE', 'AI'):1.55,('SE', 'H'):1.00,
    ('MD', 'I'):1.10,('MD', 'D'):0.85,('MD', 'AI'):0.65,('MD', 'H'):1.25,
}

D0_REG = dict(zip(REG, REGIONS["digital_index_0_100"].values if "digital_index_0_100" in REGIONS.columns
               else [38,78,55,32,82,48]))

# TOPSIS cho Bài 6
TOPSIS_COLS = ["grdp_per_capita_million_VND","fdi_registered_billion_USD","digital_index_0_100",
               "ai_readiness_0_100","trained_labor_pct","rd_intensity_pct","internet_penetration_pct","gini_coef"]
TOPSIS_LBL  = ['GRDP/N','FDI','Digital','AI Ready','LĐ ĐT','R&D','Internet','Gini']
IS_BENEFIT  = np.array([True,True,True,True,True,True,True,False])

# ============================================================
# SIDEBAR
# ============================================================
PAGES = [
    "🏠 Trang chủ",
    "🌱 Bài 1 — Cobb-Douglas + AI",
    "💰 Bài 2 — LP ngân sách số",
    "📊 Bài 3 — Priority 10 ngành",
    "🗺️ Bài 4 — LP ngành-vùng",
    "🎯 Bài 5 — MIP 15 dự án",
    "🏆 Bài 6 — TOPSIS 6 vùng",
    "🌐 Bài 7 — NSGA-II Pareto",
    "⏳ Bài 8 — Động 2026-2035",
    "👷 Bài 9 — Lao động & AI",
    "🎲 Bài 10 — Stochastic SP",
    "🤖 Bài 11 — Q-learning RL",
    "🇻🇳 Bài 12 — AIDEOM tích hợp",
]

with st.sidebar:
    st.markdown("## 🇻🇳 AIDEOM-VN")
    page = st.radio("Mục lục", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        '<div class="sidebar-card">'
        '<b>Họ và tên:</b> Ngô Khánh Huyền<br>'
        '<b>Mã sinh viên:</b> 23051259<br>'
        '<b>Học phần:</b> Các mô hình ra quyết định<br>'
        '<b>Trường:</b> UEB — ĐHQGHN'
        '</div>', unsafe_allow_html=True)
    st.markdown("")
    st.caption("📂 Dữ liệu: NSO · MoST · MIC · MPI · WB · GII 2025")

# ============================================================
# TIỆN ÍCH
# ============================================================
def kpi(col, label, value, delta=None, color=None):
    clr = color or "#f472b6"
    d_html = f'<div class="kpi-delta">↑ {delta}</div>' if delta else ''
    html = (f'<div class="kpi-box">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="color:{clr}!important">{value}</div>'
            f'{d_html}</div>')
    col.markdown(html, unsafe_allow_html=True)

def section(title, sub=None):
    sub_html = f'<div class="section-sub">{sub}</div>' if sub else ''
    st.markdown(
        f'<div class="section-hdr"><h2>{title}</h2>{sub_html}</div>',
        unsafe_allow_html=True)

def note(text, green=False):
    cls = "note-green" if green else "note"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

def formula(text):
    st.markdown(f'<div class="formula-block">{text}</div>', unsafe_allow_html=True)

def _topsis(X, w, isb):
    denom = np.sqrt((X**2).sum(0))
    denom[denom == 0] = 1e-12
    R = X / denom
    V = R * w
    A_s = np.where(isb, V.max(0), V.min(0))
    A_n = np.where(isb, V.min(0), V.max(0))
    S_s = np.sqrt(((V - A_s)**2).sum(1))
    S_n = np.sqrt(((V - A_n)**2).sum(1))
    denom2 = S_s + S_n
    denom2[denom2 == 0] = 1e-12
    return S_n / denom2

def _entropy_w(X):
    col_sum = X.sum(0)
    col_sum[col_sum == 0] = 1e-12
    P = X / col_sum
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), 0)
    d = 1 - E
    d[d < 0] = 0
    s = d.sum()
    return d / s if s > 0 else np.ones(len(d)) / len(d)

# ============================================================
# TRANG CHỦ
# ============================================================
def page_home():
    st.markdown("# 🇻🇳 AIDEOM-VN")
    st.markdown("### *AI-Driven Decision Optimization Model for Vietnam*")
    st.markdown(
        "Web app giải **12 bài toán mô hình ra quyết định** phát triển kinh tế "
        "Việt Nam trong kỷ nguyên AI — dữ liệu thực 2020–2025.")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "GDP 2025", "514 tỷ USD", "8,02%")
    kpi(c2, "Kinh tế số / GDP", "≈19,5%", "1,2 đpt")
    kpi(c3, "FDI giải ngân 2025", "27,6 tỷ USD", "8,9%")
    kpi(c4, "GDP/người 2025", "5.026 USD", "6,9%")

    st.markdown("---")
    st.markdown("## 📚 12 bài toán — 4 cấp độ")
    levels = [
        ("🟢 Cấp độ DỄ — Làm quen mô hình", "#16a34a", [
            ("Bài 1","Cobb-Douglas mở rộng: TFP, growth accounting, dự báo GDP 2030"),
            ("Bài 2","LP 4 hạng mục ngân sách số: scipy.optimize, shadow price, độ nhạy"),
            ("Bài 3","Priority 10 ngành: min-max chuẩn hóa, weighted scoring, heatmap"),
        ]),
        ("🟡 Cấp độ TRUNG BÌNH — Tối ưu cổ điển", "#ca8a04", [
            ("Bài 4","LP 24 biến ngành×vùng: ràng buộc công bằng vùng miền, PuLP+CVXPY"),
            ("Bài 5","MIP 15 dự án: biến nhị phân, tiên quyết, ngân sách đa năm"),
            ("Bài 6","TOPSIS 6 vùng: chuẩn hóa vector, Entropy weight, AHP"),
        ]),
        ("🟠 Cấp độ KHÁ KHÓ — Đa mục tiêu & Động", "#ea580c", [
            ("Bài 7","NSGA-II Pareto 4 mục tiêu: tăng trưởng/bao trùm/môi trường/an ninh"),
            ("Bài 8","Tối ưu động 2026-2035: CRRA utility, quỹ đạo K/D/AI/H, cú sốc"),
            ("Bài 9","AI & lao động: NetJob 8 ngành, Sankey, ngưỡng đào tạo"),
        ]),
        ("🔴 Cấp độ KHÓ — Bất định & Tích hợp", "#dc2626", [
            ("Bài 10","Stochastic 2 giai đoạn: VSS, EVPI, robust minimax regret"),
            ("Bài 11","Q-learning MDP 81 trạng thái: chính sách π*, so sánh rule-based"),
            ("Bài 12","AIDEOM-VN tích hợp: 6 module, 5 kịch bản, dashboard 4 tab"),
        ]),
    ]
    for title, color, items in levels:
        with st.expander(title, expanded=(color=="#16a34a")):
            for code, desc in items:
                st.markdown(
                    f'<span class="chip" style="background:{color}22;color:{color}">{code}</span>{desc}',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📂 Dữ liệu gốc Việt Nam 2020-2025")
    t1, t2, t3 = st.tabs(["📈 Vĩ mô 2020-2025","🏭 10 ngành 2024","🗺️ 6 vùng KT-XH 2024"])
    with t1:
        st.dataframe(MACRO, use_container_width=True, hide_index=True)
        fig = px.line(MACRO, x="year", y="GDP_trillion_VND", markers=True,
                      template=PLOT_TMPL, labels={"GDP_trillion_VND":"GDP (nghìn tỷ VND)","year":"Năm"})
        fig.update_traces(line_color="#f472b6", line_width=2.5, marker_size=8)
        st.plotly_chart(plotly_cfg(fig, title="GDP Việt Nam (nghìn tỷ VND) 2020-2025"), use_container_width=True)
    with t2:
        df_s = SECTORS.copy()
        if "gdp_share_2024_pct" in df_s.columns and "labor_million" in df_s.columns:
            df_s["productivity_tr_VND"] = (df_s["gdp_share_2024_pct"]/100 * 11511.9 / df_s["labor_million"]).round(1)
        st.dataframe(df_s, use_container_width=True, hide_index=True)
    with t3:
        st.dataframe(REGIONS, use_container_width=True, hide_index=True)

# ============================================================
# BÀI 1
# ============================================================
def page_bai1():
    section("🌱 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng",
            "Growth accounting · TFP · dự báo GDP 2030 · dữ liệu VN 2020-2025")
    st.latex(r"Y_t = A_t \cdot K_t^{\alpha} L_t^{\beta} D_t^{\gamma} AI_t^{\delta} H_t^{\theta}, \quad \alpha+\beta+\gamma+\delta+\theta=1")

    a,b,g,d,th = COEF["alpha"],COEF["beta"],COEF["gamma"],COEF["delta"],COEF["theta"]
    years = MACRO["year"].values
    Y,K,L,D,AI,H = Y_HIST, K_HIST, L_HIST, D_HIST, AI_HIST, H_HIST

    # 1.4.1 TFP
    A = Y / (K**a * L**b * D**g * AI**d * H**th)
    A_mean = A.mean()
    # 1.4.2 dự báo
    Y_hat = A_mean * (K**a * L**b * D**g * AI**d * H**th)
    mape = np.mean(np.abs((Y - Y_hat)/Y)) * 100

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,"MAPE dự báo", f"{mape:.2f}%")
    kpi(c2,"TFP trung bình Ā", f"{A_mean:.4f}")
    kpi(c3,"TFP 2025", f"{A[-1]:.4f}")
    kpi(c4,"Tăng TFP 20-25", f"{((A[-1]/A[0])**(1/5)-1)*100:.2f}%/năm", color="#a855f7")
    st.markdown("<br>",unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs(["📌 1.4.1 TFP A_t","📊 1.4.2 Dự báo & MAPE","📉 1.4.3 Phân rã tăng trưởng","🔭 1.4.4 Dự báo 2030"])

    with tab1:
        dfA = pd.DataFrame({"Năm":years,"Y thực tế (ng.tỷ)":Y.round(1),"A_t (TFP)":A.round(5)})
        c1,c2=st.columns([1,1.6])
        c1.dataframe(dfA, use_container_width=True, hide_index=True)
        with c2:
            fig = go.Figure()
            fig.add_scatter(x=years,y=A,mode="lines+markers",name="TFP A_t",
                           line=dict(color="#a78bfa",width=2.5),marker=dict(size=8,color="#a78bfa"))
            fig.add_hline(y=A_mean,line_dash="dash",line_color="#fbbf24",
                         annotation_text=f"Ā={A_mean:.4f}",annotation_font_color="#fbbf24")
            st.plotly_chart(plotly_cfg(fig,title="Năng suất nhân tố tổng hợp A_t 2020-2025",
                                       xtitle="Năm",ytitle="TFP (A)"), use_container_width=True)
        note("<b>Xu hướng TFP:</b> TFP tăng liên tục từ 2020→2025 (27.75→34.91), phản ánh chất lượng tăng trưởng cải thiện. Giai đoạn 2021-2022 (COVID) tăng chậm hơn nhưng không giảm — cho thấy số hóa đã hỗ trợ tốt khả năng phục hồi.")

    with tab2:
        dff = pd.DataFrame({"Năm":years,"Y thực (ng.tỷ)":Y.round(1),
                           "Y dự báo (ng.tỷ)":Y_hat.round(1),"Sai số %":((Y_hat-Y)/Y*100).round(2)})
        c1,c2=st.columns([1,1.6])
        c1.dataframe(dff,use_container_width=True,hide_index=True)
        c1.markdown(f"**MAPE = {mape:.3f}%** — sai số trung bình tuyệt đối phần trăm")
        with c2:
            fig = go.Figure()
            fig.add_bar(x=years,y=Y,name="Y thực tế",marker_color="rgba(0,201,167,0.6)")
            fig.add_scatter(x=years,y=Y_hat,name="Y dự báo",mode="lines+markers",
                          line=dict(color="#f87171",width=2.5,dash="dot"),marker=dict(size=8))
            st.plotly_chart(plotly_cfg(fig,title=f"Y thực tế vs Y dự báo — MAPE={mape:.2f}%",
                                       xtitle="Năm",ytitle="GDP (ng.tỷ VND)"), use_container_width=True)

    with tab3:
        n=5
        gY=(np.log(Y[-1])-np.log(Y[0]))/n
        gs={"TFP (A)":(np.log(A[-1])-np.log(A[0]))/n,
            "Vốn K":a*(np.log(K[-1])-np.log(K[0]))/n,
            "Lao động L":b*(np.log(L[-1])-np.log(L[0]))/n,
            "Số hóa D":g*(np.log(D[-1])-np.log(D[0]))/n,
            "AI capacity":d*(np.log(AI[-1])-np.log(AI[0]))/n,
            "Nhân lực H":th*(np.log(H[-1])-np.log(H[0]))/n}
        dfd=pd.DataFrame({"Yếu tố":list(gs.keys()),
                         "Đóng góp (%/năm)":[v*100 for v in gs.values()],
                         "Tỷ lệ (% tổng)":[v/gY*100 for v in gs.values()]}).round(3)
        c1,c2=st.columns([1,1.5])
        c1.dataframe(dfd, use_container_width=True, hide_index=True)
        c1.markdown(f"**Tăng trưởng GDP bình quân năm = {gY*100:.2f}%**")
        with c2:
            clrs=["#fb923c","#38bdf8","#94a3b8","#a78bfa","#f87171","#fbbf24"]
            fig=px.bar(dfd,x="Yếu tố",y="Tỷ lệ (% tổng)",template=PLOT_TMPL,
                      color="Yếu tố",color_discrete_sequence=clrs)
            fig.update_layout(showlegend=False)
            st.plotly_chart(plotly_cfg(fig,title=f"Đóng góp vào tăng trưởng GDP TB năm = {gY*100:.2f}%",
                                       ytitle="% đóng góp"), use_container_width=True)
        note("<b>Phân tích:</b> TFP chiếm ~49% tăng trưởng — cải thiện chất lượng rõ rệt. Vốn K ~32%. Số hóa D ~10.4% và AI ~6.2% — các yếu tố mới đóng góp ngày càng lớn. Lao động L âm nhẹ do dịch chuyển cơ cấu 2020-2021 (COVID).")

    with tab4:
        cc1,cc2=st.columns(2)
        l_g=cc1.slider("Tăng trưởng L (%/năm)",0.0,3.0,0.5,0.1)
        k_g=cc2.slider("Tăng trưởng K (%/năm)",3.0,10.0,6.0,0.5)
        K30=K[-1]*(1+k_g/100)**5; L30=L[-1]*(1+l_g/100)**5
        A30=A[-1]*1.012**5
        Y30=A30*(K30**a * L30**b * 30.0**g * 100.0**d * 35.0**th)
        cc=st.columns(3)
        kpi(cc[0],"GDP 2030 dự báo",f"{Y30:,.0f} ng.tỷ")
        kpi(cc[1],"Tăng trưởng BQ 25-30",f"{((Y30/Y[-1])**(1/5)-1)*100:.2f}%/năm",color="#fbbf24")
        kpi(cc[2],"GDP/người 2030 (~110tr dân)",f"{Y30*1e12/(110e6)/25000:,.0f} USD",color="#a855f7")
        proj=[Y[-1]]+[Y[-1]*(Y30/Y[-1])**((t+1)/5) for t in range(5)]
        fig=go.Figure()
        fig.add_scatter(x=list(range(2025,2031)),y=proj,mode="lines+markers",name="GDP dự báo",
                       line=dict(color="#f472b6",width=2.5),marker=dict(size=8))
        fig.add_vline(x=2025,line_dash="dash",line_color="#4b6070",
                     annotation_text="Thực tế",annotation_font_color="#4b6070")
        st.plotly_chart(plotly_cfg(fig,title=f"Quỹ đạo GDP 2025→2030 (kịch bản: D=30%, AI=100K DN, H=35%)",
                                   xtitle="Năm",ytitle="GDP (ng.tỷ VND)"), use_container_width=True)
        note("<b>Kịch bản 2030:</b> D=30% GDP số (từ 19.5%), AI=100K DN số (từ 80.1K), H=35% LĐ qua đào tạo (từ 29.2%), K tăng 6%/năm, TFP tăng 1.2%/năm. Mục tiêu 30% KTS/GDP 2030 khả thi nếu duy trì đầu tư D & H song song.",green=True)

# ============================================================
# BÀI 2
# ============================================================
def page_bai2():
    from scipy.optimize import linprog
    section("💰 Bài 2 — Phân bổ ngân sách 4 hạng mục đầu tư số",
            "Quy hoạch tuyến tính · scipy.optimize.linprog · shadow price · độ nhạy ngân sách")
    st.latex(r"\max Z = 0.85x_1 + 1.20x_2 + 0.95x_3 + 1.35x_4")
    formula("x₁ Hạ tầng số · x₂ AI &amp; dữ liệu · x₃ Nhân lực số · x₄ R&amp;D công nghệ (đơn vị: nghìn tỷ VND)<br>"
            "Hệ số phản ánh GDP tăng thêm / 1 nghìn tỷ đầu tư (ước lượng WB Vietnam Digital Economy 2024)")

    def solve(B=100, x3min=20):
        c=[-0.85,-1.20,-0.95,-1.35]
        A_ub=[[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[0.35,-0.65,0.35,-0.65]]
        b_ub=[B,-25,-15,-x3min,-10,0]
        return linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=[(0,None)]*4,method="highs")

    tab1,tab2,tab3=st.tabs(["📌 2.4.1-2 Lời giải & Shadow price","📈 2.4.3 Độ nhạy ngân sách","🧑‍💻 2.4.4 Ưu tiên nhân lực"])
    with tab1:
        res=solve()
        names=["x₁ Hạ tầng số","x₂ AI & dữ liệu","x₃ Nhân lực số","x₄ R&D công nghệ"]
        df=pd.DataFrame({"Hạng mục":names,"Phân bổ tối ưu (ng.tỷ)":res.x.round(2),
                         "Hệ số β":[0.85,1.20,0.95,1.35],
                         "GDP kỳ vọng":(res.x*np.array([0.85,1.20,0.95,1.35])).round(2)})
        shadow=1.35  # dual của ràng buộc ngân sách tổng = hệ số R&D (binding)
        c1,c2,c3=st.columns(3)
        kpi(c1,"Z* GDP tăng thêm",f"{-res.fun:.2f} ng.tỷ")
        kpi(c2,"Shadow price ngân sách",f"{shadow:.2f}","ng.tỷ GDP / ng.tỷ NS")
        kpi(c3,"Tỷ trọng AI+R&D",f"{(res.x[1]+res.x[3])/res.x.sum()*100:.1f}%",color="#a855f7")
        st.markdown("<br>",unsafe_allow_html=True)
        c1,c2=st.columns([1.2,1])
        c1.dataframe(df,use_container_width=True,hide_index=True)
        with c2:
            fig=px.pie(df,names="Hạng mục",values="Phân bổ tối ưu (ng.tỷ)",hole=0.45,
                      template=PLOT_TMPL,color_discrete_sequence=PALETTE)
            fig.update_traces(textfont_size=11,textinfo="percent+label")
            st.plotly_chart(plotly_cfg(fig,h=280), use_container_width=True)
        note("<b>Shadow price ngân sách = 1,35:</b> mỗi nghìn tỷ ngân sách tăng thêm sinh ra 1,35 nghìn tỷ GDP — bằng đúng hệ số R&D vì ràng buộc ngân sách là binding. Đây là cận trên chi phí cơ hội vốn công. "
             "R&D có hệ số cao nhất (1.35) nhưng ràng buộc sàn thấp nhất (10) do rủi ro triển khai cao và độ trễ dài.")

    with tab2:
        Bs=np.arange(100,210,10)
        Zs=[-solve(B).fun for B in Bs]
        df_s=pd.DataFrame({"Ngân sách (ng.tỷ)":Bs,"Z* (ng.tỷ GDP)":np.round(Zs,2)})
        c1,c2=st.columns([1,1.5])
        c1.dataframe(df_s.loc[df_s["Ngân sách (ng.tỷ)"].isin([100,120,140,160])],use_container_width=True,hide_index=True)
        c1.caption(f"Độ tăng Z* / 10 ng.tỷ NS = {(Zs[1]-Zs[0]):.2f} → Shadow price = {(Zs[1]-Zs[0])/10:.2f}")
        with c2:
            fig=go.Figure()
            fig.add_scatter(x=Bs,y=Zs,mode="lines+markers",name="Z*(B)",
                          line=dict(color="#f472b6",width=2.5),marker=dict(size=7))
            fig.add_vline(x=100,line_dash="dash",line_color="#4b6070",
                         annotation_text="Cơ sở B=100",annotation_font_color="#4b6070")
            st.plotly_chart(plotly_cfg(fig,title="Đường cong Z*(B) — GDP gain theo ngân sách",
                                       xtitle="Ngân sách tổng (ng.tỷ VND)",ytitle="Z* (ng.tỷ GDP)"),
                           use_container_width=True)

    with tab3:
        x3min=st.slider("Ràng buộc sàn nhân lực số x₃ ≥",20,45,30,1,
                       help="PDF đề xuất tăng lên 30 khi cần ưu tiên kỹ sư AI")
        res2=solve(x3min=x3min)
        res_base=solve(x3min=20)
        if res2.success:
            delta_z=(-res_base.fun)-(-res2.fun)
            st.success(f"✅ Khả thi — Z* = {-res2.fun:.2f} ng.tỷ (giảm {delta_z:.2f} ng.tỷ so với x₃≥20)")
            df2=pd.DataFrame({"Hạng mục":["x₁","x₂","x₃","x₄"],"x₃≥20":res_base.x.round(2),"x₃≥"+str(x3min):res2.x.round(2)})
            st.dataframe(df2,use_container_width=True,hide_index=True)
            note(f"<b>Tác động:</b> Tăng x₃≥{x3min} từ x₃≥20 giảm Z* = {delta_z:.2f} ng.tỷ. Đây là <b>chi phí kinh tế</b> của chính sách ưu tiên nhân lực số — chấp nhận hi sinh một phần hiệu quả ngắn hạn để bền vững dài hạn.")
        else:
            st.error("⚠️ Bài toán KHÔNG khả thi với ràng buộc này.")

# ============================================================
# BÀI 3
# ============================================================
def page_bai3():
    section("📊 Bài 3 — Chỉ số ưu tiên 10 ngành Việt Nam",
            "Chuẩn hóa min-max · weighted scoring · phân tích độ nhạy · 2 bộ trọng số")
    df=SECTORS.copy()
    c_name="sector_name_en" if "sector_name_en" in df.columns else df.columns[0]
    names_vi=['Nông-Lâm-Thủy sản','CN Chế biến chế tạo','Xây dựng','Khai khoáng',
              'Bán buôn-bán lẻ','Tài chính-Ngân hàng','Logistics-Vận tải',
              'CNTT-Truyền thông','Giáo dục-Đào tạo','Y tế']
    if len(df)==10: df["name_vi"]=names_vi
    else: df["name_vi"]=df[c_name]

    GDP24=11511.9
    df["productivity"]=(df["gdp_share_2024_pct"]/100)*GDP24/df["labor_million"]

    def nm(x):
        r=x.max()-x.min()
        return (x-x.min())/r if r>0 else pd.Series(np.zeros(len(x)),index=x.index)
    def nb(x):
        r=x.max()-x.min()
        return (x.max()-x)/r if r>0 else pd.Series(np.zeros(len(x)),index=x.index)

    Xg=pd.DataFrame({
        "growth":nm(df["growth_rate_2024_pct"]),
        "productivity":nm(df["productivity"]),
        "spillover":nm(df["spillover_coef_0_1"]),
        "export":nm(df["export_billion_USD"]),
        "labor":nm(df["labor_million"]),
        "ai":nm(df["ai_readiness_0_100"]),
    })
    Xb=nb(df["automation_risk_pct"])

    formula("Priorityᵢ = a₁·G̃ᵢ + a₂·P̃ᵢ + a₃·SP̃ᵢ + a₄·EX̃ᵢ + a₅·LB̃ᵢ + a₆·ÃIᵢ − a₇·R̃ISKᵢ<br>"
            "Chuẩn hóa: x̃ᵢ=(xᵢ−min)/(max−min) | Risk đảo: x̃ᵢ=(max−xᵢ)/(max−min)")

    tab0,tab1,tab2,tab3=st.tabs(["🔢 3.4.1 Ma trận chuẩn hóa","🏆 3.4.2 Xếp hạng mặc định",
                                 "🌡️ 3.4.3 Heatmap độ nhạy","⚖️ 3.4.4 Hai bộ trọng số"])
    with tab0:
        nm_df=Xg.copy()
        nm_df.columns=["Tăng trưởng","Năng suất","Lan tỏa","Xuất khẩu","Việc làm","AI Ready"]
        nm_df.insert(0,"Ngành",df["name_vi"].values)
        nm_df["Risk (đảo)"]=(Xb*100).round(1).values
        nm_df[nm_df.columns[1:]]=nm_df[nm_df.columns[1:]].round(4)
        st.dataframe(nm_df,use_container_width=True,hide_index=True)
        note("Chuẩn hóa min-max về [0,1]. Risk <b>đảo dấu</b>: rủi ro tự động hóa thấp → điểm cao. Năng suất = (gdp_share × GDP₂₀₂₄) / labor (từ CSV thực tế).",green=True)

    with tab1:
        # Trọng số theo PDF: a1..a6 cho 6 cột tốt, a7 cho risk — tổng = 1.10 (như PDF)
        # Để chính xác theo PDF công thức, dùng nguyên xi không chuẩn hóa tổng
        w_raw=np.array([0.15,0.15,0.20,0.15,0.10,0.20]); wr=0.15
        pr=Xg.values@w_raw - wr*Xb.values
        rk=pd.DataFrame({"#":range(1,11),"Ngành":df["name_vi"],
                        "Priority":pr.round(4),
                        "AI Ready":df["ai_readiness_0_100"].values,
                        "Rủi ro TĐH (%)":df["automation_risk_pct"].values,
                        "Tăng trưởng (%)":df["growth_rate_2024_pct"].values})\
            .sort_values("Priority",ascending=False).reset_index(drop=True)
        rk["#"]=range(1,11)
        c1,c2=st.columns([1.1,1.5])
        c1.dataframe(rk,use_container_width=True,hide_index=True)
        with c2:
            fig=px.bar(rk,x="Priority",y="Ngành",orientation="h",template=PLOT_TMPL,
                      color="Priority",color_continuous_scale="Tealgrn",
                      title="Xếp hạng ưu tiên 10 ngành (trọng số PDF mặc định)")
            fig.update_layout(yaxis=dict(autorange="reversed"),showlegend=False)
            st.plotly_chart(plotly_cfg(fig,h=380), use_container_width=True)
        note(f"<b>Top 3:</b> {rk.iloc[0]['Ngành']} ({rk.iloc[0]['Priority']:.4f}) → "
             f"{rk.iloc[1]['Ngành']} → {rk.iloc[2]['Ngành']}. "
             "CNTT-TT dẫn đầu nhờ AI Readiness = 88, Spillover = 0.92. "
             "Khai khoáng năng suất cao nhất nhưng tăng trưởng âm (-1.2%) và risk TĐH cao nhất (55%) → xếp cuối. Phù hợp NQ 57-NQ/TW.")

    with tab2:
        w_base=np.array([0.15,0.15,0.20,0.15,0.10]); wr=0.15
        rng=np.arange(0.05,0.45,0.05)
        heat=[]
        for wai in rng:
            rem=1-wai-wr
            wsc=w_base*(rem/w_base.sum()) if w_base.sum()>0 else w_base/w_base.sum()
            w_full=np.append(wsc,wai)
            heat.append(Xg.values@w_full - wr*Xb.values)
        heat=np.array(heat)
        names_short=[n[:8] for n in df["name_vi"].values]
        fig=px.imshow(heat,x=names_short,y=[f"{w:.2f}" for w in rng],
                     aspect="auto",color_continuous_scale="RdYlGn",template=PLOT_TMPL,
                     text_auto=".3f",
                     labels=dict(x="Ngành",y="w_AI (a₆)",color="Priority"),
                     title="Heatmap Priority theo trọng số AI Readiness (a₆)")
        st.plotly_chart(plotly_cfg(fig,h=420), use_container_width=True)
        note("Khi <b>tăng a₆ (AI Readiness)</b>: CNTT-TT càng nổi bật. Tài chính-NH vượt lên mạnh khi a₆≥0.25 nhờ AI Ready=72. Khai khoáng luôn cuối do AI Ready thấp (30) và risk cao.")

    with tab3:
        wg=np.array([0.25,0.25,0.10,0.25,0.05,0.05]); wg_r=0.05
        wi=np.array([0.05,0.10,0.25,0.05,0.25,0.10]); wi_r=0.20
        pg=Xg.values@wg - wg_r*Xb.values
        pi=Xg.values@wi - wi_r*Xb.values
        comp=pd.DataFrame({"Ngành":df["name_vi"],"P. Tăng trưởng":pg.round(4),"P. Bao trùm":pi.round(4)})
        c1,c2=st.columns(2)
        c1.markdown("**Top-3 Định hướng Tăng trưởng**")
        c1.dataframe(comp.nlargest(3,"P. Tăng trưởng")[["Ngành","P. Tăng trưởng"]],
                    hide_index=True,use_container_width=True)
        c2.markdown("**Top-3 Định hướng Bao trùm**")
        c2.dataframe(comp.nlargest(3,"P. Bao trùm")[["Ngành","P. Bao trùm"]],
                    hide_index=True,use_container_width=True)
        fig=go.Figure()
        fig.add_bar(y=comp["Ngành"],x=comp["P. Tăng trưởng"],name="Tăng trưởng",
                   orientation="h",marker_color="#f472b6",opacity=0.85)
        fig.add_bar(y=comp["Ngành"],x=comp["P. Bao trùm"],name="Bao trùm",
                   orientation="h",marker_color="#f87171",opacity=0.85)
        fig.update_layout(barmode="group")
        st.plotly_chart(plotly_cfg(fig,title="So sánh 2 định hướng trọng số",h=420), use_container_width=True)

# ============================================================
# BÀI 4
# ============================================================
def _solve_lp4(with_equity=True):
    import pulp
    gv,lm=0.002,0.6
    m=pulp.LpProblem("LP4",pulp.LpMaximize)
    x=pulp.LpVariable.dicts("x",(REG,ITEMS),lowBound=0)
    m+=pulp.lpSum(BETA_RJ[(r,j)]*x[r][j] for r in REG for j in ITEMS)
    m+=pulp.lpSum(x[r][j] for r in REG for j in ITEMS)<=50000
    for r in REG:
        m+=pulp.lpSum(x[r][j] for j in ITEMS)>=5000
        m+=pulp.lpSum(x[r][j] for j in ITEMS)<=12000
    m+=pulp.lpSum(x[r]["H"] for r in REG)>=12000
    if with_equity:
        M=pulp.LpVariable("Dmax")
        for r in REG:
            m+=D0_REG[r]+gv*x[r]["D"]<=M
            m+=D0_REG[r]+gv*x[r]["D"]>=lm*M
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    mat=np.array([[x[r][j].value() or 0 for j in ITEMS] for r in REG])
    return mat,pulp.value(m.objective) or 0

def page_bai4():
    section("🗺️ Bài 4 — LP phân bổ ngân sách số ngành-vùng",
            "24 biến quyết định · ràng buộc công bằng vùng miền · PuLP+CVXPY")
    with st.spinner("Đang giải LP (PuLP/CBC)..."):
        x_eq,Z_eq=_solve_lp4(True)
        x_ne,Z_ne=_solve_lp4(False)

    # comp dùng cho tab chi phí công bằng
    comp=pd.DataFrame({"Vùng":REGIONS_VI,
                       "Có công bằng":x_eq.sum(1).round(0),
                       "Không công bằng":x_ne.sum(1).round(0)})

    c1,c2,c3=st.columns(3)
    kpi(c1,"Z* (có ràng buộc công bằng)",f"{Z_eq:,.0f} tỷ")
    kpi(c2,"Z* (không ràng buộc công bằng)",f"{Z_ne:,.0f} tỷ",color="#fbbf24")
    kpi(c3,"Chi phí công bằng vùng",f"{Z_ne-Z_eq:,.0f} tỷ",color="#f87171")
    st.markdown("<br>",unsafe_allow_html=True)

    tab1,tabc,tab2=st.tabs(["📌 4.4.1-3 Phân bổ tối ưu (PuLP)","🔄 4.4.2 Đối chiếu CVXPY","⚖️ 4.4.4 Chi phí công bằng"])
    with tab1:
        dfm=pd.DataFrame(x_eq,index=REGIONS_VI,columns=ITEMS).round(0)
        dfm["Tổng"]=dfm.sum(1)
        st.dataframe(dfm,use_container_width=True)
        fig=px.imshow(x_eq,x=ITEMS,y=REGIONS_VI,aspect="auto",text_auto=".0f",
                     color_continuous_scale="YlOrRd",template=PLOT_TMPL,
                     title=f"Heatmap phân bổ tối ưu (Z* = {Z_eq:,.0f} tỷ)",
                     labels=dict(x="Hạng mục",y="Vùng",color="Tỷ VND"))
        st.plotly_chart(plotly_cfg(fig,h=340), use_container_width=True)
        note("<b>Nhận xét:</b> ĐNB và ĐBSH nhận nhiều AI nhất (β_AI cao nhất). Tây Nguyên và MN Bắc được ưu tiên H và I (β_H,β_I cao nhất do gia tốc lớn từ nền thấp). Ràng buộc C5 cân bằng Digital Index đảm bảo không vùng nào tụt hậu quá 40% vùng dẫn đầu.")

    with tabc:
        try:
            import cvxpy as cp
            beta_mat=np.array([[BETA_RJ[(r,j)] for j in ITEMS] for r in REG])
            xv=cp.Variable((6,4),nonneg=True)
            rs=cp.sum(xv,axis=1)
            D0v=np.array([D0_REG[r] for r in REG])
            Dn=D0v+0.002*xv[:,1]; Mc=cp.Variable()
            cons=[cp.sum(xv)<=50000,rs>=5000,rs<=12000,cp.sum(xv[:,3])>=12000,Dn<=Mc,Dn>=0.6*Mc]
            prob=cp.Problem(cp.Maximize(cp.sum(cp.multiply(beta_mat,xv))),cons)
            solved=False
            for sv in ["CLARABEL","ECOS","SCS"]:
                try:
                    prob.solve(solver=getattr(cp,sv))
                    if prob.status and prob.status.startswith("optimal"): solved=True; break
                except Exception: continue
            c1,c2=st.columns(2)
            kpi(c1,"Z* PuLP (CBC)",f"{Z_eq:,.0f}")
            kpi(c2,"Z* CVXPY",f"{prob.value:,.0f}" if solved and prob.value else "N/A")
            if solved and prob.value:
                delta=abs(Z_eq-prob.value)
                st.success(f"Chênh lệch PuLP vs CVXPY = {delta:.2f} tỷ → {'Hai solver trùng khớp ✅' if delta<1 else 'Sai lệch nhỏ ✅'}")
                st.dataframe(pd.DataFrame(xv.value,index=REGIONS_VI,columns=ITEMS).round(0),use_container_width=True)
        except ImportError:
            st.info("💡 CVXPY chưa cài. Kết quả PuLP/CBC: Z* = "+f"{Z_eq:,.0f} tỷ")
            note("Cài CVXPY: <code>pip install cvxpy</code> để so sánh với PuLP. Cả hai solver LP convex sẽ cho kết quả trùng với sai số &lt;1 tỷ.")

    with tab2:
        fig=go.Figure()
        fig.add_bar(x=comp["Vùng"],y=comp["Có công bằng"],name="Có công bằng (C5)",marker_color="#f472b6",opacity=0.85)
        fig.add_bar(x=comp["Vùng"],y=comp["Không công bằng"],name="Không có C5",marker_color="#f87171",opacity=0.75)
        fig.update_layout(barmode="group")
        st.plotly_chart(plotly_cfg(fig,title="Tổng ngân sách theo vùng: có vs không có ràng buộc công bằng C5",
                                  ytitle="Ngân sách (tỷ VND)"), use_container_width=True)
        note(f"<b>Chi phí công bằng = {Z_ne-Z_eq:,.0f} tỷ</b> ({(Z_ne-Z_eq)/Z_ne*100:.1f}% GDP gain). "
             "Không có C5: vốn tập trung ĐNB+ĐBSH do β_AI cao nhất → vùng khó càng tụt hậu. "
             "Đây là chi phí kinh tế của mục tiêu phát triển <b>bao trùm</b>.")

# ============================================================
# BÀI 5
# ============================================================
def page_bai5():
    from pulp import LpProblem,LpMaximize,LpVariable,lpSum,value,PULP_CBC_CMD,LpStatus
    section("🎯 Bài 5 — MIP lựa chọn dự án chuyển đổi số",
            "15 dự án · biến nhị phân · ràng buộc loại trừ, tiên quyết, đa năm")
    P=list(range(1,16))
    C={1:12000,2:11500,3:18000,4:4500,5:3200,6:5800,7:6500,8:15000,
       9:2500,10:7200,11:4800,12:8500,13:20000,14:3800,15:1500}
    C1={1:8500,2:7500,3:12000,4:3500,5:2500,6:4000,7:4500,8:9000,
        9:1800,10:5000,11:3500,12:5500,13:13000,14:2800,15:1200}
    B={1:21500,2:20800,3:32500,4:9200,5:6800,6:11400,7:12200,8:28500,
       9:5800,10:13800,11:8500,12:16200,13:35000,14:7500,15:3800}
    names={1:'TT dữ liệu Hòa Lạc',2:'TT dữ liệu phía Nam',3:'5G toàn quốc',
           4:'VNeID 2.0',5:'Cổng DVC v3',6:'Y tế số',7:'Giáo dục số K-12',
           8:'TT AI + supercomputing',9:'Fintech sandbox',10:'Logistics thông minh',
           11:'Nông nghiệp số ĐBSCL',12:'Đào tạo 50K kỹ sư AI',
           13:'Khu CN bán dẫn BN-BG',14:'An ninh mạng SOC',15:'Open Data quốc gia'}
    fields={1:'ht',2:'ht',3:'ht',4:'cp',5:'cp',6:'yt',7:'gd',8:'ai',
            9:'tc',10:'lg',11:'nn',12:'nl',13:'bd',14:'an',15:'dl'}
    prob_p={'ht':.85,'cp':.75,'ai':.65,'bd':.65,'yt':.8,'gd':.8,'tc':.8,'lg':.8,'nn':.8,'nl':.8,'an':.8,'dl':.8}

    def solve(BT=80000,B12=40000,use_exp=False,force12=False):
        m=LpProblem("sel",LpMaximize)
        y=LpVariable.dicts("y",P,cat="Binary")
        m+=lpSum((prob_p[fields[i]] if use_exp else 1)*B[i]*y[i] for i in P)
        m+=lpSum(C[i]*y[i] for i in P)<=BT
        m+=lpSum(C1[i]*y[i] for i in P)<=B12
        if force12: m+=y[1]>=1; m+=y[2]>=1
        else: m+=y[1]+y[2]<=1
        m+=y[8]<=y[12]; m+=y[13]<=y[12]
        m+=y[4]+y[5]>=1; m+=y[14]>=1
        m+=lpSum(y[i] for i in P)>=7; m+=lpSum(y[i] for i in P)<=11
        m.solve(PULP_CBC_CMD(msg=False))
        sel=[i for i in P if (y[i].value() or 0)>0.5]
        return sel,sum(C[i] for i in sel),value(m.objective) or 0,LpStatus[m.status]

    tab1,tab2,tabf,tab3=st.tabs(["📌 5.4.1 Lời giải cơ sở","💰 5.4.2 Nới ngân sách","🔒 5.4.3 Bắt buộc P1+P2","🎲 5.4.4 Lợi ích kỳ vọng"])
    with tab1:
        sel,tc,Z,_=solve()
        c1,c2,c3,c4=st.columns(4)
        kpi(c1,"Số dự án chọn",f"{len(sel)}/15")
        kpi(c2,"Tổng chi phí",f"{tc:,} tỷ")
        kpi(c3,"Tổng lợi ích Z*",f"{Z:,.0f} tỷ")
        kpi(c4,"NPV biên (B/C)",f"{Z/tc:.2f}x",color="#a855f7")
        st.markdown("<br>",unsafe_allow_html=True)
        df_sel=pd.DataFrame([{"Mã":f"P{i}","Tên dự án":names[i],"Chi phí (tỷ)":C[i],
                              "NPV (tỷ)":B[i],"Năm 1-2":C1[i],"B/C":round(B[i]/C[i],2)} for i in sel])
        st.dataframe(df_sel,use_container_width=True,hide_index=True)
        sel0_ids=set(sel)
        p15_note="P15 (Open Data, B/C=2.53) bị bỏ qua do <b>ràng buộc số lượng dự án ≤11</b> — các dự án khác có NPV tuyệt đối cao hơn nhiều. Đây là hạn chế của mô hình: cần xem xét B/C theo quy mô." if 15 not in sel else ""
        if p15_note: note(p15_note)

    with tab2:
        BT=st.slider("Ngân sách tổng (tỷ VND)",80000,130000,100000,5000)
        sel2,tc2,Z2,_=solve(BT=BT)
        sel0,_,Z0,_=solve()
        added=set(sel2)-set(sel0); removed=set(sel0)-set(sel2)
        st.success(f"✅ Chọn {len(sel2)} dự án · Z* = {Z2:,.0f} tỷ (Δ{Z2-Z0:+,.0f} tỷ so với B=80.000)")
        if added: st.write("🟢 Thêm mới: "+", ".join(f"P{i} ({names[i]})" for i in sorted(added)))
        if removed: st.write("🔴 Loại bỏ: "+", ".join(f"P{i}" for i in sorted(removed)))
        st.dataframe(pd.DataFrame([{"P":f"P{i}","Tên":names[i],"Chi phí":C[i],"NPV":B[i]} for i in sel2]),
                    use_container_width=True,hide_index=True)

    with tabf:
        sel0,_,Z0,_=solve()
        sel3,tc3,Z3,stt=solve(force12=True)
        if stt=="Optimal":
            st.success(f"✅ Khả thi — Z* = {Z3:,.0f} tỷ (giảm {Z0-Z3:,.0f} tỷ do buộc cả P1+P2)")
            note(f"Bắt buộc P1 (TT DL Hòa Lạc, 12.000 tỷ) <b>và</b> P2 (TT DL phía Nam, 11.500 tỷ) cùng lúc vi phạm ràng buộc loại trừ gốc (y₁+y₂≤1). Khi nới lỏng, Z* giảm {Z0-Z3:,.0f} tỷ — chi phí redundancy địa lý.")
            st.dataframe(pd.DataFrame([{"Mã":f"P{i}","Tên":names[i],"Chi phí":C[i],"NPV":B[i]} for i in sel3]),
                        use_container_width=True,hide_index=True)
        else:
            st.error("⚠️ Bài toán KHÔNG khả thi khi bắt buộc cả P1 và P2.")

    with tab3:
        sel0,_,Z0,_=solve()
        sel4,tc4,Z4,_=solve(use_exp=True)
        c1,c2=st.columns(2)
        kpi(c1,"Deterministic Z*",f"{Z0:,.0f} tỷ")
        kpi(c2,"Expected E[Z] = Σpᵢ·Bᵢ·yᵢ",f"{Z4:,.0f} tỷ",color="#fbbf24")
        added4=set(sel4)-set(sel0); removed4=set(sel0)-set(sel4)
        if added4: st.write(f"🟢 Thêm khi tính rủi ro: "+", ".join(f"P{i} ({names[i]}, p={prob_p[fields[i]]})" for i in sorted(added4)))
        if removed4: st.write(f"🔴 Loại khi tính rủi ro: "+", ".join(f"P{i} ({names[i]}, p={prob_p[fields[i]]})" for i in sorted(removed4)))
        note("Dự án AI (P8, p=0.65) và Bán dẫn (P13, p=0.65) bị điều chỉnh giảm NPV kỳ vọng khi tính xác suất hoàn thành. Đây là lý do cần <b>portfolio risk management</b> trong hoạch định chương trình CĐS quốc gia.",green=True)

# ============================================================
# BÀI 6
# ============================================================
def page_bai6():
    section("🏆 Bài 6 — TOPSIS xếp hạng 6 vùng theo ưu tiên đầu tư AI",
            "Chuẩn hóa vector · trọng số chuyên gia vs Entropy · AHP · QĐ 127/QĐ-TTg")
    X=REGIONS[TOPSIS_COLS].values.astype(float)
    rn=REGIONS["region_name_en"].values if "region_name_en" in REGIONS.columns else np.array(REGIONS_VI)
    rn_vi=np.array(REGIONS_VI)

    w_exp=np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
    C_exp=_topsis(X,w_exp,IS_BENEFIT)
    w_ent=_entropy_w(X)
    C_ent=_topsis(X,w_ent,IS_BENEFIT)

    # Xếp hạng theo expert
    rank_exp=np.argsort(-C_exp)
    c1,c2,c3=st.columns(3)
    kpi(c1,f"#1 — {rn_vi[rank_exp[0]]}",f"C* = {C_exp[rank_exp[0]]:.4f}")
    kpi(c2,f"#2 — {rn_vi[rank_exp[1]]}",f"C* = {C_exp[rank_exp[1]]:.4f}",color="#a855f7")
    kpi(c3,f"#3 — {rn_vi[rank_exp[2]]}",f"C* = {C_exp[rank_exp[2]]:.4f}",color="#fbbf24")
    st.markdown("<br>",unsafe_allow_html=True)

    tab1,tab2,tab3,tab4=st.tabs(["📌 6.4.1 Trọng số chuyên gia","🔀 6.4.2 Entropy vs Expert",
                                 "🌡️ 6.4.3 Độ nhạy w_AI","🧮 6.4.4 AHP vs TOPSIS"])
    with tab1:
        df_t=pd.DataFrame({"Vùng":rn_vi,"C* (Expert)":C_exp.round(4),
                          "S_star":np.array([np.sqrt((((_topsis_detail(X,w_exp,IS_BENEFIT))[0])[i])**2) for i in range(6)]).round(6) if False else ["—"]*6,
                          "AI Ready":REGIONS["ai_readiness_0_100"].values,
                          "Digital":REGIONS["digital_index_0_100"].values})\
            .sort_values("C* (Expert)",ascending=False).reset_index(drop=True)
        df_t.insert(0,"Hạng",range(1,7))
        c1,c2=st.columns([1,1.5])
        c1.dataframe(df_t[["Hạng","Vùng","C* (Expert)","AI Ready","Digital"]],
                    use_container_width=True,hide_index=True)
        with c2:
            fig=px.bar(df_t.sort_values("C* (Expert)"),x="C* (Expert)",y="Vùng",
                      orientation="h",template=PLOT_TMPL,
                      color="C* (Expert)",color_continuous_scale="Tealgrn",text_auto=".4f")
            fig.update_traces(textposition="outside",textfont_size=10)
            st.plotly_chart(plotly_cfg(fig,title="Xếp hạng TOPSIS — trọng số chuyên gia",h=320), use_container_width=True)
        note(f"<b>Trọng số chuyên gia:</b> GRDP(0.10) FDI(0.10) Digital(0.15) AI(0.20) LĐ(0.15) R&D(0.15) Internet(0.05) Gini(0.10). "
             f"ĐNB và ĐBSH dẫn đầu nhờ AI Ready cao và Digital Index tốt. Tây Nguyên và MN Bắc cuối do năng lực số thấp.",green=True)

    with tab2:
        comp=pd.DataFrame({"Vùng":rn_vi,"C* Expert":C_exp.round(4),"C* Entropy":C_ent.round(4),
                          "Δ rank":np.argsort(-C_ent)+1})
        st.dataframe(comp,use_container_width=True,hide_index=True)
        fig=go.Figure()
        fig.add_bar(y=rn_vi,x=C_exp,name="Expert (w₄_AI=0.20)",orientation="h",
                   marker_color="#f472b6",opacity=0.85)
        fig.add_bar(y=rn_vi,x=C_ent,name="Entropy (khách quan)",orientation="h",
                   marker_color="#f87171",opacity=0.75)
        fig.update_layout(barmode="group")
        st.plotly_chart(plotly_cfg(fig,title="TOPSIS C*: Trọng số chuyên gia vs Entropy",h=360), use_container_width=True)
        we=pd.DataFrame({"Tiêu chí":TOPSIS_LBL,"Entropy w":w_ent.round(4),"Expert w":w_exp.round(2)})
        st.dataframe(we,use_container_width=True,hide_index=True)
        note("Trọng số Entropy đặt rất cao FDI (0.415) vì FDI có biến động lớn nhất giữa các vùng (range: 0.8→20 tỷ USD). Internet (0.007) gần đồng đều → Entropy thấp → trọng số Entropy thấp. Kết quả xếp hạng thay đổi nhẹ nhưng top-2 ổn định.")

    with tab3:
        rng=np.arange(0.10,0.45,0.05)
        heat=[]
        for wai in rng:
            wg=0.10
            rem=1-wai-wg
            wb=np.array([0.10,0.10,0.15,0.15,0.15,0.05])
            wsc=wb*(rem/wb.sum())
            wfull=np.insert(wsc,3,wai)
            wfull=np.append(wfull,wg)
            heat.append(_topsis(X,wfull,IS_BENEFIT))
        heat=np.array(heat)
        fig=px.imshow(heat,x=[f"V{i+1}\n{n[:5]}" for i,n in enumerate(REGIONS_VI)],
                     y=[f"{w:.2f}" for w in rng],aspect="auto",text_auto=".3f",
                     color_continuous_scale="RdYlGn",template=PLOT_TMPL,
                     labels=dict(x="Vùng",y="w_AI (a₄)",color="C*"),
                     title="Heatmap C* theo trọng số AI Readiness (w₄: 0.10→0.40)")
        st.plotly_chart(plotly_cfg(fig,h=420), use_container_width=True)
        st.caption("V1=MN Bắc · V2=ĐBSH · V3=Trung Bộ · V4=Tây Nguyên · V5=ĐNB · V6=ĐBSCL")
        note("Top-2 (ĐNB + ĐBSH) <b>ổn định</b> qua mọi mức w_AI. Thứ 3 giữa Trung Bộ và ĐBSCL thay đổi khi w_AI>0.25. Tây Nguyên luôn cuối khi AI Ready=18.")

    with tab4:
        ahp=np.array([
            [1,1,1/3,1/5,1/3,1/3,3,3],[1,1,1/3,1/5,1/3,1/3,3,3],
            [3,3,1,1/2,1,1,5,5],[5,5,2,1,2,2,7,7],
            [3,3,1,1/2,1,1,5,5],[3,3,1,1/2,1,1,5,5],
            [1/3,1/3,1/5,1/7,1/5,1/5,1,1],[1/3,1/3,1/5,1/7,1/5,1/5,1,1]])
        n8=8
        gm=np.prod(ahp,axis=1)**(1/n8); w_ahp=gm/gm.sum()
        lam=np.mean((ahp@w_ahp)/w_ahp)
        CI=(lam-n8)/(n8-1); CR=CI/1.41
        C_ahp=_topsis(X,w_ahp,IS_BENEFIT)
        st.write(f"**Kiểm tra nhất quán AHP:** λ_max={lam:.3f}, CI={CI:.3f}, CR={CR:.3f} "
                 f"{'✅ Nhất quán (CR<0.10)' if CR<0.10 else '⚠️ Chưa nhất quán'}")
        cmp=pd.DataFrame({"Vùng":rn_vi,"C* Expert":C_exp.round(4),"C* Entropy":C_ent.round(4),"C* AHP":C_ahp.round(4)})
        st.dataframe(cmp,use_container_width=True,hide_index=True)
        wd=pd.DataFrame({"Tiêu chí":TOPSIS_LBL,"Expert w":w_exp.round(3),"Entropy w":w_ent.round(3),"AHP w":w_ahp.round(3)})
        st.dataframe(wd,use_container_width=True,hide_index=True)
        note(f"Ba phương pháp trọng số đồng thuận <b>ĐNB và ĐBSH</b> là top-2. Theo QĐ 127/QĐ-TTg (3 trung tâm AI quốc gia): <b>{REGIONS_VI[rank_exp[0]]} + {REGIONS_VI[rank_exp[1]]} + {REGIONS_VI[rank_exp[2]]}</b>. Cần cân nhắc thêm yếu tố địa-chính trị (cân bằng Bắc-Nam).",green=True)

# ============================================================
# BÀI 7
# ============================================================
BETA_MAT7=np.array([[1.15,.85,.55,1.30],[.95,1.25,1.40,1.05],[1.05,.95,.85,1.15],
                    [1.20,.75,.45,1.35],[.90,1.30,1.55,1.00],[1.10,.85,.65,1.25]])
D0_7=np.array([38,78,55,32,82,48]); E_7=np.array([.42,.55,.48,.32,.62,.38])
RHO_7=np.array([.18,.45,.28,.12,.52,.22]); SIG_7=np.array([.32,.28,.30,.35,.25,.30])

@st.cache_data(show_spinner=False)
def _run_nsga(pop=80,gen=120):
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize as moo_min
        from pymoo.termination import get_termination
        class VP(ElementwiseProblem):
            def __init__(s):
                super().__init__(n_var=24,n_obj=4,n_ieq_constr=20,xl=np.zeros(24),xu=np.ones(24)*12000)
            def _evaluate(s,x,out,*a,**k):
                X=x.reshape(6,4)
                f1=-(BETA_MAT7*X).sum()
                su=X.sum(1); f2=np.abs(su-su.mean()).mean()
                f3=(E_7*(X[:,0]+X[:,2])).sum()
                f4=(RHO_7*X[:,2]).sum()-(SIG_7*X[:,3]).sum()
                out['F']=[f1,f2,f3,f4]
                g=[X.sum()-50000]
                for r in range(6): g.append(5000-X[r].sum())
                for r in range(6): g.append(X[r].sum()-12000)
                g.append(12000-X[:,3].sum())
                Dn=D0_7+0.002*X[:,1]; Dm=Dn.max()
                for r in range(6): g.append(0.6*Dm-Dn[r])
                out['G']=np.array(g)
        res=moo_min(VP(),NSGA2(pop_size=pop),get_termination("n_gen",gen),seed=42,verbose=False,save_history=False)
        if res.F is not None and len(np.atleast_2d(res.F))>0:
            F=np.atleast_2d(res.F); X=np.atleast_2d(res.X)
        else:
            pop_obj=res.pop; Xall=pop_obj.get("X"); Fall=pop_obj.get("F"); Gall=pop_obj.get("G")
            cv=np.maximum(0,Gall).sum(axis=1); keep=np.argsort(cv)[:max(20,pop//2)]
            F,X=Fall[keep],Xall[keep]
        return F,X
    except ImportError:
        return None,None

def page_bai7():
    section("🌐 Bài 7 — Tối ưu đa mục tiêu Pareto NSGA-II",
            "4 mục tiêu: tăng trưởng · bao trùm · môi trường · an ninh dữ liệu")
    with st.spinner("Đang chạy NSGA-II (pop=80, gen=120)..."):
        F,X=_run_nsga()
    if F is None:
        st.error("⚠️ Cần cài pymoo: `pip install pymoo`"); return

    w_pol=np.array([0.40,0.25,0.20,0.15])
    fmin,fmax=F.min(0),F.max(0)
    fr=np.where(fmax-fmin>1e-12,fmax-fmin,1.0)
    R=(F-fmin)/fr; V=R*w_pol
    S_s=np.sqrt((V**2).sum(1)); S_n=np.sqrt(((V-w_pol)**2).sum(1))
    denom=S_s+S_n; denom[denom==0]=1e-12
    Cs=S_n/denom; best=int(np.argmax(Cs)); mg=int(np.argmin(F[:,0]))
    bX=X[best].reshape(6,4)

    c1,c2,c3=st.columns(3)
    kpi(c1,"Số nghiệm Pareto",f"{len(F)}")
    kpi(c2,"GDP gain (thỏa hiệp)",f"{-F[best,0]:,.0f} tỷ")
    kpi(c3,"GDP gain (max tăng trưởng)",f"{-F[mg,0]:,.0f} tỷ",color="#fbbf24")
    st.markdown("<br>",unsafe_allow_html=True)

    tab1,tabp,tab2=st.tabs(["🌐 7.4.2 Biên Pareto 3D","📊 Parallel Coordinates","🎯 7.4.3-4 Nghiệm thỏa hiệp"])
    with tab1:
        fig=go.Figure(go.Scatter3d(x=-F[:,0],y=F[:,1],z=F[:,2],mode="markers",
            marker=dict(size=3,color=F[:,3],colorscale="Plasma",colorbar=dict(title="Rủi ro",len=0.7))))
        fig.add_trace(go.Scatter3d(x=[-F[best,0]],y=[F[best,1]],z=[F[best,2]],
            mode="markers",marker=dict(size=12,color="red",symbol="diamond"),name="✅ Thỏa hiệp"))
        fig.add_trace(go.Scatter3d(x=[-F[mg,0]],y=[F[mg,1]],z=[F[mg,2]],
            mode="markers",marker=dict(size=12,color="#fbbf24",symbol="diamond"),name="📈 Max GDP"))
        fig.update_layout(template=PLOT_TMPL,height=560,
            scene=dict(xaxis_title="GDP gain (tỷ)",yaxis_title="Gini/MAD (bao trùm)",zaxis_title="Phát thải CO₂"),
            title="Tập Pareto 3D — màu theo rủi ro an ninh dữ liệu")
        st.plotly_chart(fig,use_container_width=True)
    with tabp:
        fn=np.copy(F).astype(float)
        for i in range(4):
            lo,hi=F[:,i].min(),F[:,i].max()
            fn[:,i]=(F[:,i]-lo)/(hi-lo) if hi>lo else 0.5
        figp=go.Figure(go.Parcoords(
            line=dict(color=fn[:,0],colorscale="Viridis"),
            dimensions=[dict(label="GDP gain ↓min",values=fn[:,0]),
                       dict(label="Gini/MAD ↓min",values=fn[:,1]),
                       dict(label="Phát thải ↓min",values=fn[:,2]),
                       dict(label="Rủi ro ↓min",values=fn[:,3])]))
        figp.update_layout(template=PLOT_TMPL,height=460,
                          title="Parallel coordinates — 4 mục tiêu chuẩn hóa [0,1]")
        st.plotly_chart(figp,use_container_width=True)
    with tab2:
        dfm=pd.DataFrame(bX,index=REGIONS_VI,columns=ITEMS).round(0)
        dfm["Tổng"]=dfm.sum(1)
        st.write("**Phân bổ nghiệm thỏa hiệp TOPSIS (w=0.40/0.25/0.20/0.15):**")
        st.dataframe(dfm,use_container_width=True)
        oc=pd.DataFrame({"Mục tiêu":["GDP gain (tỷ)","Gini/MAD (bao trùm)","Phát thải CO₂","Rủi ro an ninh"],
                        "Thỏa hiệp":[-F[best,0],F[best,1],F[best,2],F[best,3]],
                        "Max tăng trưởng":[-F[mg,0],F[mg,1],F[mg,2],F[mg,3]]}).round(1)
        oc["Chênh lệch %"]=(np.abs(oc["Thỏa hiệp"]-oc["Max tăng trưởng"])/np.abs(oc["Max tăng trưởng"])*100).round(1)
        st.write("**Chi phí cơ hội — thỏa hiệp vs tăng trưởng cao nhất:**")
        st.dataframe(oc,use_container_width=True,hide_index=True)
        note(f"<b>Chi phí cơ hội:</b> Chọn nghiệm thỏa hiệp hi sinh {(((-F[mg,0])-(-F[best,0]))/(-F[mg,0])*100):.1f}% GDP gain nhưng đổi lấy cải thiện về bao trùm, môi trường và an ninh dữ liệu. Đây là lý do NSGA-II phù hợp hoạch định chính sách công hơn LP đơn mục tiêu.")

# ============================================================
# BÀI 8
# ============================================================
@st.cache_data(show_spinner=False)
def _run_dynamic():
    from scipy.optimize import minimize
    a,b,g,d,th=0.33,0.42,0.10,0.08,0.07
    dK,dD,dAI=0.05,0.12,0.15; thH,mu=0.8,0.02
    p1,p2,p3=0.003,0.002,0.004; rho,gcr,T=0.97,1.5,10
    K0,L0,D0,AI0,H0,Y0=27500.,53.9,20.3,86.,30.,12847.6
    A0=Y0/(K0**a*L0**b*D0**g*AI0**d*H0**th)
    Lv=np.array([L0*1.009**t for t in range(T+1)])
    def traj(u,sy=None,sp=0.0):
        IK,ID,IAI,IH=u[0::4],u[1::4],u[2::4],u[3::4]
        K=np.zeros(T+1);Dv=np.zeros(T+1);AI=np.zeros(T+1);H=np.zeros(T+1)
        A=np.zeros(T+1);Yv=np.zeros(T+1);C=np.zeros(T)
        K[0],Dv[0],AI[0],H[0],A[0]=K0,D0,AI0,H0,A0
        for t in range(T):
            At=A[t]*(1-sp) if sy is not None and t==sy else A[t]
            Yv[t]=At*K[t]**a*Lv[t]**b*Dv[t]**g*AI[t]**d*H[t]**th
            C[t]=Yv[t]-IK[t]-ID[t]-IAI[t]-IH[t]
            if C[t]<=0: return None
            K[t+1]=(1-dK)*K[t]+IK[t]; Dv[t+1]=(1-dD)*Dv[t]+ID[t]
            AI[t+1]=(1-dAI)*AI[t]+IAI[t]; H[t+1]=H[t]+thH*IH[t]-mu*H[t]
            A[t+1]=A[t]*(1+p1*Dv[t]/100+p2*AI[t]/100+p3*H[t]/100)
        Yv[T]=A[T]*K[T]**a*Lv[T]**b*Dv[T]**g*AI[T]**d*H[T]**th
        return K,Dv,AI,H,Yv,C,A
    def welf(u,sy=None,sp=0.0):
        r=traj(u,sy,sp)
        if r is None or np.any(r[5]<=0): return 1e15
        C=r[5]; return -sum(rho**t*(C[t]**(1-gcr)-1)/(1-gcr) for t in range(T))
    ti=14000*0.15; u0=np.tile([ti*.4,ti*.25,ti*.2,ti*.15],T)
    cons=[{"type":"ineq","fun":lambda u:(lambda r:1e10 if r is None else min(r[5])-1)(traj(u))}]
    res=minimize(welf,u0,method="SLSQP",bounds=[(0,None)]*(T*4),
                constraints=cons,options={"maxiter":600,"ftol":1e-8})
    W_base=-welf(res.x); W_plan=-welf(res.x,2,0.08)
    res_sh=minimize(lambda u:welf(u,2,.08),res.x,method="SLSQP",bounds=[(0,None)]*(T*4),
                   constraints=[{"type":"ineq","fun":lambda u:(lambda r:1e10 if r is None else min(r[5])-1)(traj(u,2,.08))}],
                   options={"maxiter":600,"ftol":1e-8})
    W_reopt=-res_sh.fun
    u_front=np.zeros(T*4)
    for t in range(T):
        f=1.5 if t<3 else 0.7
        u_front[t*4:(t+1)*4]=np.array([ti*.4,ti*.25,ti*.2,ti*.15])*f
    return {"opt":traj(res.x),"W":-res.fun,
            "shock":dict(W_base=W_base,W_plan=W_plan,W_reopt=W_reopt,
                        Y_base=traj(res.x)[4],Y_shock=traj(res.x,2,.08)[4],Y_reopt=traj(res_sh.x,2,.08)[4]),
            "strat":dict(W_opt=-res.fun,W_even=-welf(u0),W_front=-welf(u_front),
                        Y_opt=traj(res.x)[4],Y_even=traj(u0)[4],Y_front=traj(u_front)[4])}

def page_bai8():
    section("⏳ Bài 8 — Tối ưu động liên thời gian 2026-2035",
            "Quỹ đạo tối ưu K, D, AI, H, Y, C · CRRA utility · SLSQP · cú sốc TFP")
    with st.spinner("Đang tối ưu quỹ đạo 10 năm (SLSQP, ~30s)..."):
        out=_run_dynamic()
    (K,D,AI,H,Y,C,A)=out["opt"]; W=out["W"]
    years=list(range(2026,2037))
    c1,c2,c3=st.columns(3)
    kpi(c1,"Phúc lợi W* (CRRA)",f"{W:.3f}")
    kpi(c2,"GDP 2035 dự báo",f"{Y[-1]:,.0f} ng.tỷ")
    kpi(c3,"Tăng trưởng bình quân",f"{((Y[-1]/Y[0])**(1/10)-1)*100:.2f}%/năm",color="#a855f7")
    st.markdown("<br>",unsafe_allow_html=True)

    tab1,tab3,tab4=st.tabs(["📈 8.3.1-2 Quỹ đạo tối ưu","⚡ 8.3.3 Cú sốc TFP 2028","🆚 8.3.4 So sánh chiến lược"])
    with tab1:
        fig=make_subplots(rows=2,cols=3,subplot_titles=("K — Vốn vật chất (ng.tỷ)","D — Hạ tầng số (%)","AI — Năng lực AI (K DN)","H — Nhân lực số (%)","Y (GDP) & C (tiêu dùng)","A — TFP"))
        for s,r,c,col,nm in [(K,1,1,"#38bdf8","K"),(D,1,2,"#f472b6","D"),(AI,1,3,"#a78bfa","AI"),(H,2,1,"#fb923c","H"),(A,2,3,"#4ade80","A")]:
            fig.add_trace(go.Scatter(x=years,y=s,mode="lines+markers",name=nm,line_color=col,showlegend=True,marker_size=5),row=r,col=c)
        fig.add_trace(go.Scatter(x=years,y=Y,name="Y GDP",line_color="#f1f5f9",marker_size=4),row=2,col=2)
        fig.add_trace(go.Scatter(x=years[:10],y=C,name="C tiêu dùng",line_color="#22d3ee",marker_size=4),row=2,col=2)
        fig.update_layout(template=PLOT_TMPL,height=620,title="Quỹ đạo tối ưu 2026-2035 — SLSQP+CRRA")
        st.plotly_chart(fig,use_container_width=True)
        note("<b>Front-load AI & H:</b> Mô hình ưu tiên đầu tư D và AI mạnh giai đoạn đầu (2026-2029) để tích lũy năng lực hấp thụ. H (nhân lực số) nên đầu tư <b>đồng thời hoặc trước AI</b> để tránh bottleneck kỹ năng. Đây là lý do hệ số φ₃ (H→TFP) lớn nhất.",green=True)

    with tab3:
        sh=out["shock"]
        cc=st.columns(3)
        kpi(cc[0],"W không sốc",f"{sh['W_base']:.3f}")
        kpi(cc[1],"W giữ kế hoạch",f"{sh['W_plan']:.3f}",color="#fbbf24")
        kpi(cc[2],"W tái tối ưu",f"{sh['W_reopt']:.3f}",color="#f472b6")
        fig=go.Figure()
        fig.add_scatter(x=years,y=sh["Y_base"],name="Không sốc",line=dict(color="#4ade80",width=2.5))
        fig.add_scatter(x=years,y=sh["Y_shock"],name="Có sốc — giữ KH",line=dict(color="#f87171",width=2.5))
        fig.add_scatter(x=years,y=sh["Y_reopt"],name="Có sốc — tái tối ưu",line=dict(color="#fbbf24",width=2,dash="dot"))
        fig.add_vline(x=2028,line_dash="dash",line_color="#4b6070",annotation_text="Sốc 2028 (bão Yagi)",annotation_font_color="#94a3b8")
        st.plotly_chart(plotly_cfg(fig,title="GDP: Cú sốc TFP -8% năm 2028",xtitle="Năm",ytitle="GDP (ng.tỷ)"), use_container_width=True)
        note(f"<b>Tái tối ưu phục hồi {(sh['W_reopt']-sh['W_plan'])/(abs(sh['W_base'])-abs(sh['W_plan']))*100:.1f}% phúc lợi</b> bị mất do sốc. Sốc TFP persistent (lan qua A[t+1]) nặng hơn sốc tạm thời → cần quỹ dự phòng H như 'hàng hóa bảo hiểm'.")

    with tab4:
        s=out["strat"]
        df_s=pd.DataFrame({"Chiến lược":["Tối ưu SLSQP","Đầu tư đều","Front-load (×1.5 năm 1-3)"],
                          "Phúc lợi W":[round(s["W_opt"],3),round(s["W_even"],3),round(s["W_front"],3)],
                          "GDP 2035 (ng.tỷ)":[round(s["Y_opt"][-1]),round(s["Y_even"][-1]),round(s["Y_front"][-1])]})
        c1,c2=st.columns([1,1.5])
        c1.dataframe(df_s,use_container_width=True,hide_index=True)
        with c2:
            fig=go.Figure()
            fig.add_scatter(x=years,y=s["Y_opt"],name="Tối ưu",line=dict(color="#f472b6",width=2.5))
            fig.add_scatter(x=years,y=s["Y_even"],name="Đầu tư đều",line=dict(color="#38bdf8",width=2))
            fig.add_scatter(x=years,y=s["Y_front"],name="Front-load",line=dict(color="#fb923c",width=2,dash="dot"))
            st.plotly_chart(plotly_cfg(fig,title="GDP theo chiến lược đầu tư",xtitle="Năm",ytitle="GDP"), use_container_width=True)

# ============================================================
# BÀI 9
# ============================================================
def page_bai9():
    from scipy.optimize import linprog
    section("👷 Bài 9 — AI & thị trường lao động Việt Nam",
            "Tối đa hóa NetJob ròng 8 ngành · phân tích ngưỡng đào tạo · Sankey")
    N=8
    sec=['Nông-LT','CN chế biến','Xây dựng','Bán buôn-bán lẻ','Tài chính-NH','Logistics','CNTT-TT','Giáo dục']
    Lv=np.array([13.20,11.50,4.80,7.80,0.55,1.95,0.62,2.15])
    risk=np.array([18,42,25,38,52,35,28,22])/100
    a1=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
    b1=np.array([45,28,35,32,22,30,20,55])
    c1v=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
    d1=np.array([50,32,42,38,26,36,24,62])
    coeff=a1-c1v*risk

    def solve(cap5=False):
        c_obj=np.concatenate([-coeff,-b1])
        A1=np.concatenate([np.ones(N),np.ones(N)]).reshape(1,-1)
        A1b=np.concatenate([-np.ones(N),np.zeros(N)]).reshape(1,-1)
        A2=np.zeros((N,2*N)); A3=np.zeros((N,2*N))
        for i in range(N):
            A2[i,i]=-coeff[i]; A2[i,N+i]=-b1[i]
            A3[i,i]=c1v[i]*risk[i]; A3[i,N+i]=-d1[i]
        A_ub=np.vstack([A1,A1b,A2,A3])
        b_ub=np.concatenate([[30000],[-9000],np.zeros(N),np.zeros(N)])
        if cap5:
            A4=np.zeros((N,2*N))
            for i in range(N): A4[i,i]=c1v[i]*risk[i]
            A_ub=np.vstack([A_ub,A4]); b_ub=np.concatenate([b_ub,0.05*Lv*1e6])
        return linprog(c_obj,A_ub=A_ub,b_ub=b_ub,bounds=[(0,None)]*(2*N),method="highs")

    res=solve(); xA,xH=res.x[:N],res.x[N:]
    NetJob=coeff*xA+b1*xH; Displaced=c1v*risk*xA
    RetrainCap=d1*xH

    c1,c2,c3=st.columns(3)
    kpi(c1,"Tổng NetJob ròng",f"{-res.fun:,.0f} việc")
    kpi(c2,"Tổng x_AI",f"{xA.sum():,.0f} tỷ")
    kpi(c3,"Tổng x_H (đào tạo)",f"{xH.sum():,.0f} tỷ")
    st.markdown("<br>",unsafe_allow_html=True)

    tab1,tab2s,tab3s,tab2=st.tabs(["📌 9.4.1 Phân bổ & NetJob","📏 9.4.2 Ngưỡng đào tạo","🌊 9.4.3 Sankey lao động","🔒 9.4.4 Ràng buộc 5%L"])
    with tab1:
        df=pd.DataFrame({"Ngành":sec,"x_AI (tỷ)":xA.round(0),"x_H (tỷ)":xH.round(0),
                        "Displaced (việc)":Displaced.round(0),"RetrainCap":RetrainCap.round(0),"NetJob":NetJob.round(0)})
        df["NetJob≥0"]=df["NetJob"]>=0
        c1,c2=st.columns([1.3,1.2])
        c1.dataframe(df,use_container_width=True,hide_index=True)
        with c2:
            fig=px.bar(df,x="Ngành",y="NetJob",template=PLOT_TMPL,
                      color="NetJob",color_continuous_scale="RdYlGn",
                      title="NetJob ròng theo ngành")
            fig.add_hline(y=0,line_dash="dash",line_color="#64748b")
            st.plotly_chart(plotly_cfg(fig,h=320), use_container_width=True)

    with tab2s:
        i=1
        net=a1[i]-c1v[i]*risk[i]; rr=c1v[i]*risk[i]/d1[i]
        st.write(f"**CN chế biến chế tạo:** Hệ số net AI = {net:.1f} việc/tỷ, "
                 f"Tỷ lệ retrain cần thiết = c₁·risk/d₁ = **{rr:.4f}**")
        st.write(f"→ Mỗi 1 tỷ đầu tư AI cần ≥ **{rr:.4f} tỷ** đầu tư đào tạo.")
        xr=np.linspace(0,25000,200); xh_re=rr*xr; xh_nj=np.maximum(0,-net/b1[i]*xr)
        fig=go.Figure()
        fig.add_scatter(x=xr,y=xh_re,name="Ngưỡng retrain (Displaced≤Retrain)",line=dict(color="#f87171",width=2))
        fig.add_scatter(x=xr,y=xh_nj,name="Ngưỡng NetJob≥0",line=dict(color="#38bdf8",width=2))
        fig.add_scatter(x=xr,y=np.maximum(xh_re,xh_nj),name="Vùng khả thi (x_H ≥)",
                       fill="tonexty",line=dict(color="#4ade80",width=1.5),fillcolor="rgba(74,222,128,0.1)")
        st.plotly_chart(plotly_cfg(fig,title="Ngưỡng đào tạo tối thiểu — CN chế biến chế tạo",
                                  xtitle="x_AI (tỷ VND)",ytitle="x_H tối thiểu (tỷ VND)"), use_container_width=True)

    with tab3s:
        vuln=[0,2,3]; labels=list(np.array(sec)[vuln])+["Giữ việc","Đào tạo lại","Mất việc"]
        src,tgt,val,colr=[],[],[],[]
        for idx,k in enumerate(vuln):
            disp=max(0,Displaced[k]); retr=min(disp,RetrainCap[k]); lost=max(0,disp-retr)
            kept=Lv[k]*1e6-disp
            for tn,v,col in [(len(vuln),kept,"#4ade80"),(len(vuln)+1,retr,"#fbbf24"),(len(vuln)+2,lost,"#f87171")]:
                if v>100: src.append(idx);tgt.append(tn);val.append(v);colr.append(col)
        fig=go.Figure(go.Sankey(
            node=dict(label=labels,pad=20,thickness=20,
                     color=["#60a5fa"]*len(vuln)+["#4ade80","#fbbf24","#f87171"]),
            link=dict(source=src,target=tgt,value=val,color=colr)))
        fig.update_layout(template=PLOT_TMPL,height=420,title="Luồng dịch chuyển lao động dễ tổn thương (Nông-LT · Xây dựng · Bán buôn)")
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        res4=solve(cap5=True)
        if res4.success:
            gain_loss=(-res.fun)-(-res4.fun)
            st.success(f"✅ Khả thi — NetJob = {-res4.fun:,.0f} (giảm {gain_loss:,.0f} = {gain_loss/(-res.fun)*100:.1f}%)")
            note("<b>Ràng buộc cốt lõi:</b> Displaced_i ≤ 5%·L_i (không ngành nào mất >5% lao động). "
                "Tức: c₁·risk·x_AI ≤ 0.05·L_i·1e6. Ngành Tài chính-NH (risk=52%, c₁=72.5) bị giới hạn mạnh nhất.",green=True)
        else:
            st.error("⚠️ KHÔNG khả thi với ràng buộc displaced ≤ 5%L — cần tăng ngân sách đào tạo.")

# ============================================================
# BÀI 10 — QUY HOẠCH NGẪU NHIÊN 2 GIAI ĐOẠN
# ============================================================
J10 = ['I', 'D', 'AI', 'H']
S10 = ['s1', 's2', 's3', 's4']
P_S = {'s1': 0.30, 's2': 0.45, 's3': 0.20, 's4': 0.05}
BETA_BASE = {'I': 1.00, 'D': 1.10, 'AI': 1.25, 'H': 0.95}
BETA_S = {
    ('s1', 'I'): 1.25, ('s1', 'D'): 1.35, ('s1', 'AI'): 1.55, ('s1', 'H'): 1.05,
    ('s2', 'I'): 1.00, ('s2', 'D'): 1.10, ('s2', 'AI'): 1.25, ('s2', 'H'): 0.95,
    ('s3', 'I'): 0.75, ('s3', 'D'): 0.85, ('s3', 'AI'): 0.90, ('s3', 'H'): 1.00,
    ('s4', 'I'): 0.40, ('s4', 'D'): 0.50, ('s4', 'AI'): 0.55, ('s4', 'H'): 1.10
}

def _solve_sp():
    from scipy.optimize import linprog
    n = 4 + 16
    c = np.zeros(n)
    for k, j in enumerate(J10): c[k] = -BETA_BASE[j]
    for si, s in enumerate(S10):
        for k, j in enumerate(J10): c[4 + si * 4 + k] = -P_S[s] * BETA_S[(s, j)]
    A_ub = []; b_ub = []
    A_ub.append([1, 1, 1, 1] + [0] * 16); b_ub.append(65000)
    for si in range(4):
        row = [0] * n
        for k in range(4): row[4 + si * 4 + k] = 1
        A_ub.append(row); b_ub.append(15000)
    for si in range(4):
        row = [0] * n; row[3] = -0.5; row[4 + si * 4 + 2] = 1
        A_ub.append(row); b_ub.append(0)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0, None)] * n, method="highs")
    x_sp = res.x[:4]; Z_sp = -res.fun
    y_sp = {s: res.x[4 + i * 4:4 + i * 4 + 4] for i, s in enumerate(S10)}
    
    det = {}
    for s in S10:
        cs = np.zeros(8)
        for k, j in enumerate(J10): cs[k] = -BETA_BASE[j]; cs[4 + k] = -BETA_S[(s, j)]
        r = linprog(cs, A_ub=np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, -0.5, 0, 0, 1, 0]]),
                    b_ub=[65000, 15000, 0], bounds=[(0, None)] * 8, method="highs")
        det[s] = {"Z": -r.fun, "x": r.x[:4]}
    Z_ws = sum(P_S[s] * det[s]["Z"] for s in S10)
    
    beta_avg = {j: sum(P_S[s] * BETA_S[(s, j)] for s in S10) for j in J10}
    rev = linprog([-beta_avg[j] for j in J10], A_ub=[[1, 1, 1, 1]], b_ub=[65000], bounds=[(0, None)] * 4, method="highs")
    x_ev = rev.x
    Z_ev = sum(BETA_BASE[j] * x_ev[k] for k, j in enumerate(J10))
    for s in S10:
        r = linprog([-BETA_S[(s, j)] for j in J10], A_ub=[[1, 1, 1, 1], [0, 0, 1, 0]],
                    b_ub=[15000, 0.5 * x_ev[3]], bounds=[(0, None)] * 4, method="highs")
        Z_ev += P_S[s] * (-r.fun)
    
    nr = 21; cr = np.zeros(nr); cr[20] = 1.0
    Ar = []; br = []
    row0 = [0] * nr
    for k in range(4): row0[k] = 1
    Ar.append(row0); br.append(65000)
    for si in range(4):
        row = [0] * nr
        for k in range(4): row[4 + si * 4 + k] = 1
        Ar.append(row); br.append(15000)
    for si in range(4):
        row = [0] * nr; row[3] = -0.5; row[4 + si * 4 + 2] = 1
        Ar.append(row); br.append(0)
    for si, s in enumerate(S10):
        row = [0] * nr
        for k, j in enumerate(J10): row[k] = -BETA_BASE[j]; row[4 + si * 4 + k] = -BETA_S[(s, j)]
        row[20] = -1; Ar.append(row); br.append(-det[s]["Z"])
    rr = linprog(cr, A_ub=np.array(Ar), b_ub=np.array(br), bounds=[(0, None)] * 20 + [(None, None)], method="highs")
    
    return x_sp, y_sp, Z_sp, Z_ev, Z_ws, det, rr.x[:4], rr.x[20], x_ev

def page_bai10():
    section("🎲 Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn",
            "First-stage here-and-now · recourse · VSS · EVPI · robust minimax regret")
    
    x_sp, y_sp, Z_sp, Z_ev, Z_ws, det, x_rob, w_rob, x_ev = _solve_sp()
    VSS = Z_sp - Z_ev; EVPI = Z_ws - Z_sp
    
    c = st.columns(4)
    kpi(c[0], "Z* Stochastic SP", f"{Z_sp:,.0f}")
    kpi(c[1], "Z* EV solution", f"{Z_ev:,.0f}", color="#fbbf24")
    kpi(c[2], "VSS = Z*_SP − Z*_EV", f"{VSS:,.0f}", color="#f472b6")
    kpi(c[3], "EVPI = Z*_WS − Z*_SP", f"{EVPI:,.0f}", color="#a855f7")
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tabr = st.tabs(["📌 10.5.1 First & Second stage", "📊 10.5.2-3 Kịch bản · VSS · EVPI", "🛡️ 10.5.4 Robust minimax regret"])
    with tab1:
        c1, c2 = st.columns(2)
        c1.write("**First-stage x* (here-and-now):**")
        c1.dataframe(pd.DataFrame({"Hạng mục": J10, "x* (tỷ)": x_sp.round(1), "% ngân sách": (x_sp / 65000 * 100).round(1)}), use_container_width=True, hide_index=True)
        ydf = pd.DataFrame({s: y_sp[s].round(1) for s in S10}, index=J10).T
        ydf.index = [f"{s} (p={P_S[s]})" for s in S10]
        c2.write("**Second-stage recourse y_s:**")
        c2.dataframe(ydf, use_container_width=True)
        note(f"SP đầu tư H = {x_sp[3]:.0f} tỷ nhiều hơn EV ({x_ev[3]:.0f} tỷ). <b>H là hàng hóa bảo hiểm</b>: ràng buộc y_AI≤0.5·x_H buộc first-stage phải tích lũy H đủ để mở rộng AI linh hoạt trong tương lai.", green=True)

    with tab2:
        scn = pd.DataFrame({"Kịch bản": ["Lạc quan (s1)", "Cơ sở (s2)", "Bi quan (s3)", "Khủng hoảng (s4)"],
                            "Xác suất": [0.30, 0.45, 0.20, 0.05], "Z*[s] (tỷ)": [round(det[s]["Z"], 0) for s in S10]})
        c1, c2 = st.columns([1, 1.5])
        c1.dataframe(scn, use_container_width=True, hide_index=True)
        with c2:
            fig = px.bar(scn, x="Kịch bản", y="Z*[s] (tỷ)", template=PLOT_TMPL, color="Kịch bản")
            fig.add_hline(y=Z_sp, line_dash="dash", line_color="#f472b6", annotation_text=f"Z*_SP={Z_sp:,.0f}")
            st.plotly_chart(fig, use_container_width=True)
        note(f"<b>VSS = {VSS:,.0f}:</b> giá trị của tư duy xác suất. <b>EVPI = {EVPI:,.0f}:</b> giá trị thông tin hoàn hảo.", green=True)

    with tabr:
        kpi(st, "Minimax regret (worst-case)", f"{w_rob:,.0f} tỷ")
        cmp = pd.DataFrame({"Hạng mục": J10, "x* SP (tỷ)": x_sp.round(0), "x* Robust (tỷ)": x_rob.round(0)})
        st.dataframe(cmp, use_container_width=True, hide_index=True)
# ============================================================
# BÀI 11
# ============================================================
class VietnamEconomyEnv:
    def __init__(self):
        self.allocation={0:np.array([.70,.10,.10,.10]),1:np.array([.40,.25,.15,.20]),
                        2:np.array([.25,.45,.15,.15]),3:np.array([.20,.20,.45,.15]),4:np.array([.30,.20,.10,.40])}
        self.action_names=['Truyền thống','Cân bằng','Số hóa nhanh','AI dẫn dắt','Bao trùm']
        self.w=np.array([.40,.25,.20,.15]); self.T=10
        self.rng=np.random.default_rng(0)
    def reset(self,state=None):
        self.state=np.array(state) if state is not None else self.rng.integers(0,3,4)
        self.t=0; self.K,self.D,self.AI,self.H=27500.,20.3,86.,30.; self.Y_prev=12847.6
        return self.state.copy()
    def step(self,action):
        a=self.allocation[action]; budget=2100.
        self.K=.95*self.K+a[0]*budget; self.D=.88*self.D+a[1]*budget*.01
        self.AI=.85*self.AI+a[2]*budget*.05; self.H=self.H+.8*a[3]*budget*.01-.02*self.H
        A=33.70*(1+.003*self.D/100+.002*self.AI/100+.004*self.H/100)**self.t
        L=53.9*1.009**self.t
        Y=A*self.K**.33*L**.42*self.D**.10*self.AI**.08*self.H**.07
        dg=(Y-self.Y_prev)/self.Y_prev
        du=max(0,-dg*.5); cy=self.AI/(self.H+1)*.01; em=(self.K+self.AI)*.0001
        r=self.w[0]*dg*100-self.w[1]*du*100-self.w[2]*cy-self.w[3]*em
        self.Y_prev=Y; self.t+=1
        gl=0 if dg<.03 else(1 if dg<.06 else 2)
        dl=0 if self.D<25 else(1 if self.D<35 else 2)
        al=0 if self.AI<100 else(1 if self.AI<200 else 2)
        hl=0 if self.H<35 else(1 if self.H<50 else 2)
        self.state=np.array([gl,dl,al,hl])
        return self.state.copy(),r,self.t>=self.T

@st.cache_data(show_spinner=False)
def _train_q(n_ep=8000):
    env=VietnamEconomyEnv(); Q=np.zeros((3,3,3,3,5)); hist=[]
    for ep in range(n_ep):
        s=env.reset(); tot=0; eps=max(.05,1-ep/4000)
        while True:
            a=env.rng.integers(5) if env.rng.random()<eps else int(np.argmax(Q[tuple(s)]))
            s2,r,done=env.step(a)
            Q[tuple(s)+(a,)]+=.1*(r+.95*Q[tuple(s2)].max()*(1-done)-Q[tuple(s)+(a,)])
            tot+=r; s=s2
            if done: break
        hist.append(tot)
    return Q,hist

def _eval_policy(Q,pol,n=300):
    env=VietnamEconomyEnv(); rs=[]
    for _ in range(n):
        s=env.reset(); tot=0
        while True:
            a=pol(s,Q,env); s,r,done=env.step(a); tot+=r
            if done: break
        rs.append(tot)
    return np.mean(rs),np.std(rs)

def page_bai11():
    section("🤖 Bài 11 — Q-learning chính sách kinh tế thích nghi",
            "MDP 3⁴=81 trạng thái · 5 hành động · ε-greedy · so sánh rule-based")
    with st.spinner("Đang huấn luyện Q-learning (8.000 episodes)..."):
        Q,hist=_train_q()
    env=VietnamEconomyEnv()

    tab01,tab02,tab1,tab2=st.tabs(["🧩 11.3.1 Môi trường MDP","📚 11.3.2 Huấn luyện Q","🎯 11.3.3 Chính sách π*","📊 11.3.4 So sánh & Learning curve"])
    with tab01:
        c1,c2=st.columns(2)
        c1.markdown("**Không gian trạng thái (3⁴ = 81):**")
        c1.dataframe(pd.DataFrame({
            "Yếu tố":["GDP growth","Digital D","AI capacity","Nhân lực H"],
            "Low (0)":["<3%","<25%","<100K DN","<35%"],
            "Med (1)":["3-6%","25-35%","100-200K","35-50%"],
            "High (2)":[">6%",">35%",">200K DN",">50%"]}),hide_index=True,use_container_width=True)
        c2.markdown("**Không gian hành động (5 chiến lược K/D/AI/H):**")
        c2.dataframe(pd.DataFrame({
            "Hành động":env.action_names,
            "K%":[70,40,25,20,30],"D%":[10,25,45,20,20],
            "AI%":[10,15,15,45,10],"H%":[10,20,15,15,40]}),hide_index=True,use_container_width=True)
        note("<b>Phần thưởng:</b> R = 0.40·ΔGDP − 0.25·ΔUnemploy − 0.20·CyberRisk − 0.15·Emission. "
             "Môi trường mô phỏng bằng Cobb-Douglas đã calibrate. Mỗi episode = 10 năm (T=10). "
             "MDP này là <b>đơn giản hóa minh họa</b> — không phản ánh toàn bộ độ phức tạp kinh tế.",green=True)

    with tab02:
        c=st.columns(4)
        kpi(c[0],"Learning rate α","0.10")
        kpi(c[1],"Discount γ","0.95")
        kpi(c[2],"Episodes","8.000")
        kpi(c[3],"ε: 1.0 → 0.05","greedy",color="#a855f7")
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]")
        w=200; sm=np.convolve(hist,np.ones(w)/w,mode="valid")
        fig=go.Figure()
        fig.add_scatter(x=list(range(len(sm))),y=sm,mode="lines",name="Phần thưởng (TB200 ep)",
                       line=dict(color="#f472b6",width=2))
        st.plotly_chart(plotly_cfg(fig,title="Hội tụ Q-learning — phần thưởng TB trượt 200 episodes",
                                  xtitle="Episode",ytitle="Tổng phần thưởng"), use_container_width=True)
        st.caption(f"Q-table: {Q.shape} | Q_max = {Q.max():.3f} | Hội tụ rõ từ episode ~4000")

    with tab1:
        tests=[([1,1,0,1],"VN 2026 thực tế — GDP_med, D_med, AI_low, H_med"),
               ([0,0,0,2],"Kịch bản tệ — GDP_low, D_low, AI_low, H_high"),
               ([2,2,2,2],"Kịch bản tốt — GDP_high, D_high, AI_high, H_high"),
               ([0,1,0,0],"Sau khủng hoảng — GDP_low, D_med, AI_low, H_low"),
               ([1,0,2,1],"AI mạnh D yếu — GDP_med, D_low, AI_high, H_med")]
        rows=[{"Trạng thái":desc,"π* chọn":env.action_names[int(np.argmax(Q[tuple(s)]))],
              "Q values":str(Q[tuple(s)].round(2))} for s,desc in tests]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        note("VN 2026 (trạng thái [1,1,0,1]): π* chọn <b>Cân bằng (a1)</b> — phù hợp giai đoạn build momentum. "
             "Kịch bản tệ: π* ưu tiên <b>Bao trùm (a4)</b> hoặc Số hóa nhanh để thoát bẫy thu nhập thấp. "
             "Kịch bản tốt: <b>AI dẫn dắt (a3)</b> khi năng lực đủ cao.",green=True)

    with tab2:
        pols=[("π* Q-learning",lambda s,Q,e:int(np.argmax(Q[tuple(s)]))),
             ("Luôn Cân bằng (a1)",lambda s,Q,e:1),
             ("Luôn AI dẫn dắt (a3)",lambda s,Q,e:3),
             ("Random",lambda s,Q,e:e.rng.integers(5))]
        results={n:_eval_policy(Q,p) for n,p in pols}
        dfp=pd.DataFrame({"Chính sách":list(results.keys()),
                         "Phúc lợi BQ":[round(v[0],3) for v in results.values()],
                         "Std":[round(v[1],3) for v in results.values()]})
        c1,c2=st.columns(2)
        c1.dataframe(dfp,use_container_width=True,hide_index=True)
        c1.write(f"**π* cải thiện {((results['π* Q-learning'][0]-results['Luôn Cân bằng (a1)'][0])/abs(results['Luôn Cân bằng (a1)'][0])*100):.1f}%** so với Cân bằng cố định")
        with c2:
            fig=px.bar(dfp,x="Chính sách",y="Phúc lợi BQ",template=PLOT_TMPL,
                      color="Chính sách",color_discrete_sequence=PALETTE,error_y="Std")
            fig.update_layout(showlegend=False)
            st.plotly_chart(plotly_cfg(fig,title="So sánh phúc lợi BQ các chính sách"), use_container_width=True)
        note("<b>Lưu ý quan trọng:</b> AI hỗ trợ ra quyết định, <b>không thay thế</b> trách nhiệm chính trị-xã hội. "
             "π* minh họa kỹ thuật học chính sách thích nghi theo trạng thái — cần xử lý thêm tính diễn giải được (explainability) và giám sát của con người trước khi ứng dụng thực tế.")

# ============================================================
# BÀI 12
# ============================================================
def _m1_forecast(alloc,T=4,budget=3000):
    a,b,g,d,th=0.33,0.42,0.10,0.08,0.07
    K,Dv,AI,H,A,L0=27500.,20.3,86.,30.,33.70,53.9
    traj=[A*K**a*L0**b*Dv**g*AI**d*H**th]
    for t in range(T):
        K=.95*K+alloc['K']*budget; Dv=.88*Dv+alloc['D']*budget*.01
        AI=.85*AI+alloc['AI']*budget*.05; H=H+.8*alloc['H']*budget*.01-.02*H
        A=A*(1+.003*Dv/100+.002*AI/100+.004*H/100); L=L0*1.009**(t+1)
        traj.append(A*K**a*L**b*Dv**g*AI**d*H**th)
    return traj

M12_SCEN={'S1 Truyền thống':{'K':.70,'D':.10,'AI':.10,'H':.10},
          'S2 Số hóa nhanh':{'K':.25,'D':.45,'AI':.15,'H':.15},
          'S3 AI dẫn dắt':  {'K':.20,'D':.20,'AI':.45,'H':.15},
          'S4 Bao trùm số': {'K':.30,'D':.20,'AI':.10,'H':.40},
          'S5 Tối ưu cân bằng':{'K':.25,'D':.25,'AI':.30,'H':.20}}

def page_bai12():
    section("🇻🇳 Bài 12 — Nguyên mẫu AIDEOM-VN tích hợp",
            "6 module M1→M6 · 5 kịch bản chính sách · dashboard 4 tab")
    years=list(range(2026,2031))
    gdp_fc={n:_m1_forecast(al) for n,al in M12_SCEN.items()}

    a,b,g,d,th=0.33,0.42,0.10,0.08,0.07
    Y,K,L,Dv,AI,H=Y_HIST,K_HIST,L_HIST,D_HIST,AI_HIST,H_HIST
    A_=Y/(K**a*L**b*Dv**g*AI**d*H**th)
    Y_hat=A_.mean()*(K**a*L**b*Dv**g*AI**d*H**th)
    mape_=np.mean(np.abs((Y-Y_hat)/Y))*100
    Y2030=gdp_fc['S5 Tối ưu cân bằng'][-1]

    c1,c2,c3,c4=st.columns(4)
    kpi(c1,"MAPE Cobb-Douglas",f"{mape_:.2f}%")
    kpi(c2,"TFP Ā (2020-25)",f"{A_.mean():.4f}")
    kpi(c3,"GDP 2030 (S5 cân bằng)",f"{Y2030:,.0f} ng.tỷ")
    kpi(c4,"6 module tích hợp","M1→M6 ✅",color="#f472b6")
    st.markdown("<br>",unsafe_allow_html=True)

    t1,t2,t3,t4=st.tabs(["📊 Tổng quan M1-M2","💰 Phân bổ M3","🎬 5 Kịch bản M6","⚠️ Rủi ro M4-M5"])
    with t1:
        st.markdown("#### M1 — Dự báo kinh tế (Cobb-Douglas)")
        n=5; gY=(np.log(Y[-1])-np.log(Y[0]))/n
        gs={"TFP":(np.log(A_[-1])-np.log(A_[0]))/n,
            "Vốn K":a*(np.log(K[-1])-np.log(K[0]))/n,
            "Lao động L":b*(np.log(L[-1])-np.log(L[0]))/n,
            "Số hóa D":g*(np.log(Dv[-1])-np.log(Dv[0]))/n,
            "AI":d*(np.log(AI[-1])-np.log(AI[0]))/n,
            "Nhân lực H":th*(np.log(H[-1])-np.log(H[0]))/n}
        dec=pd.DataFrame({"Yếu tố":list(gs.keys()),"Đóng góp %":[v/gY*100 for v in gs.values()]})
        fig=px.bar(dec,x="Yếu tố",y="Đóng góp %",template=PLOT_TMPL,color="Yếu tố",
                  color_discrete_sequence=PALETTE,title=f"M1: Phân rã tăng trưởng 2020-2025 (GDP BQ {gY*100:.2f}%/năm)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(plotly_cfg(fig,h=340), use_container_width=True)

        st.markdown("#### M2 — Đánh giá sẵn sàng số (TOPSIS)")
        X_t=REGIONS[TOPSIS_COLS].values.astype(float)
        w_e=np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
        C_e=_topsis(X_t,w_e,IS_BENEFIT)
        C_en=_topsis(X_t,_entropy_w(X_t),IS_BENEFIT)
        m2=pd.DataFrame({"Vùng":REGIONS_VI,"C* Expert":C_e.round(4),"C* Entropy":C_en.round(4)})
        fig=go.Figure()
        fig.add_bar(y=m2["Vùng"],x=m2["C* Expert"],name="Expert",orientation="h",marker_color="#60a5fa",opacity=0.85)
        fig.add_bar(y=m2["Vùng"],x=m2["C* Entropy"],name="Entropy",orientation="h",marker_color="#fb923c",opacity=0.75)
        fig.update_layout(barmode="group")
        st.plotly_chart(plotly_cfg(fig,title="M2: TOPSIS Expert vs Entropy — Sẵn sàng AI",h=320), use_container_width=True)

    with t2:
        st.markdown("#### M3 — Tối ưu phân bổ LP (có ràng buộc công bằng C5)")
        with st.spinner("Đang giải LP..."):
            x_opt,Z_lp=_solve_lp4(True)
        kpi(st,"LP Z* (GDP gain)",f"{Z_lp:,.0f} tỷ VND")
        fig=px.imshow(x_opt,x=ITEMS,y=REGIONS_VI,aspect="auto",text_auto=".0f",
                     color_continuous_scale="YlOrRd",template=PLOT_TMPL,
                     title=f"M3: Phân bổ tối ưu 6 vùng × 4 hạng mục (Z*={Z_lp:,.0f})")
        st.plotly_chart(plotly_cfg(fig,h=360), use_container_width=True)
        tot=x_opt.sum()
        note(f"<b>M3 summary:</b> D={x_opt[:,1].sum()/tot*100:.1f}% · AI={x_opt[:,2].sum()/tot*100:.1f}% · H={x_opt[:,3].sum()/tot*100:.1f}%. "
             "ĐNB+ĐBSH nhận AI nhiều nhất; MN Bắc+Tây Nguyên ưu tiên H và I.",green=True)

    with t3:
        st.markdown("#### M6 — So sánh 5 kịch bản chính sách 2026-2030")
        fig=go.Figure()
        for (n,traj),col in zip(gdp_fc.items(),PALETTE):
            fig.add_trace(go.Scatter(x=years,y=traj,mode="lines+markers",name=n,
                                   line=dict(color=col,width=2.5),marker=dict(size=7)))
        st.plotly_chart(plotly_cfg(fig,title="M6: Quỹ đạo GDP theo 5 kịch bản chính sách 2026-2030",
                                  xtitle="Năm",ytitle="GDP (ng.tỷ VND)"), use_container_width=True)
        tbl=pd.DataFrame({"Kịch bản":list(gdp_fc.keys()),
                         "GDP 2026":[round(t[0],0) for t in gdp_fc.values()],
                         "GDP 2030":[round(t[-1],0) for t in gdp_fc.values()],
                         "TB %/năm":[round(((t[-1]/t[0])**(1/4)-1)*100,2) for t in gdp_fc.values()]})
        st.dataframe(tbl,use_container_width=True,hide_index=True)
        best_scen=tbl.loc[tbl["GDP 2030"].idxmax(),"Kịch bản"]
        note(f"<b>Kịch bản tốt nhất GDP 2030: {best_scen}.</b> S3 AI dẫn dắt cho GDP cao nhất nhưng đòi hỏi AI Readiness cao. S5 Cân bằng bền vững nhất. S1 Truyền thống thấp nhất → cần chuyển dịch cơ cấu đầu tư.",green=True)

    with t4:
        from scipy.optimize import linprog
        st.markdown("#### M4 — Mô phỏng lao động (NetJob)")
        N=8
        sec_=['Nông-LT','CN chế biến','Xây dựng','Bán buôn','Tài chính','Logistics','CNTT','Giáo dục']
        a1_=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
        b1_=np.array([45,28,35,32,22,30,20,55])
        c1v_=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
        d1_=np.array([50,32,42,38,26,36,24,62])
        risk_=np.array([18,42,25,38,52,35,28,22])/100
        coeff_=a1_-c1v_*risk_
        c_obj=np.concatenate([-coeff_,-b1_])
        A1r=np.concatenate([np.ones(N),np.ones(N)]).reshape(1,-1)
        A1b=np.concatenate([-np.ones(N),np.zeros(N)]).reshape(1,-1)
        A2r=np.zeros((N,2*N)); A3r=np.zeros((N,2*N))
        for i in range(N):
            A2r[i,i]=-coeff_[i]; A2r[i,N+i]=-b1_[i]
            A3r[i,i]=c1v_[i]*risk_[i]; A3r[i,N+i]=-d1_[i]
        r_=linprog(c_obj,A_ub=np.vstack([A1r,A1b,A2r,A3r]),
                  b_ub=np.concatenate([[30000],[-9000],np.zeros(N),np.zeros(N)]),
                  bounds=[(0,None)]*(2*N),method="highs")
        NJ_=coeff_*r_.x[:N]+b1_*r_.x[N:]
        kpi(st,"Tổng NetJob M4",f"{-r_.fun:,.0f} việc")
        fig=px.bar(x=sec_,y=NJ_.round(0),template=PLOT_TMPL,color=NJ_,
                  color_continuous_scale="RdYlGn",labels={"x":"Ngành","y":"NetJob"},
                  title="M4: NetJob ròng theo ngành")
        fig.add_hline(y=0,line_dash="dash",line_color="#64748b")
        st.plotly_chart(plotly_cfg(fig,h=320), use_container_width=True)

        st.markdown("#### M5 — Đánh giá rủi ro (Stochastic SP)")
        # Thêm một dấu gạch dưới nữa để khớp với giá trị x_ev vừa thêm vào
        _,_,Z_sp_,Z_ev_,Z_ws_,det_,_,_,_ = _solve_sp()
        c1,c2,c3=st.columns(3)
        kpi(c1,"Z* SP",f"{Z_sp_:,.0f}")
        kpi(c2,"VSS",f"{Z_sp_-Z_ev_:,.0f}",color="#f472b6")
        kpi(c3,"EVPI",f"{Z_ws_-Z_sp_:,.0f}",color="#a855f7")
        scn_=pd.DataFrame({"Kịch bản":["Lạc quan","Cơ sở","Bi quan","Khủng hoảng"],
                           "p":[0.30,0.45,0.20,0.05],
                           "Z*[s]":[round(det_[s]["Z"],0) for s in S10]})
        fig=px.bar(scn_,x="Kịch bản",y="Z*[s]",template=PLOT_TMPL,color="Kịch bản",
                  color_discrete_sequence=['#4ade80','#60a5fa','#fb923c','#f87171'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(plotly_cfg(fig,title=f"M5: Z* theo kịch bản (SP={Z_sp_:,.0f})",h=300), use_container_width=True)
        note("<b>Khuyến nghị tích hợp AIDEOM-VN:</b> Kịch bản <b>S5 Cân bằng</b> tối ưu tổng thể: GDP 2030 cao, NetJob dương, VSS>0 (dùng SP). "
             "Ưu tiên đầu tư H trước AI (ràng buộc y_AI≤0.5·x_H). "
             "AIDEOM-VN là công cụ hỗ trợ — quyết định cuối thuộc quy trình chính sách.",green=True)

# ============================================================
# ROUTER
# ============================================================
ROUTES={
    PAGES[0]:page_home, PAGES[1]:page_bai1, PAGES[2]:page_bai2,
    PAGES[3]:page_bai3, PAGES[4]:page_bai4, PAGES[5]:page_bai5,
    PAGES[6]:page_bai6, PAGES[7]:page_bai7, PAGES[8]:page_bai8,
    PAGES[9]:page_bai9, PAGES[10]:page_bai10, PAGES[11]:page_bai11,
    PAGES[12]:page_bai12,
}
ROUTES[page]()
