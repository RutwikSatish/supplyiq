"""
SupplyIQ — Supply Chain Risk Intelligence Platform
====================================================
Built by Rutwik Satish | MS Engineering Management, Northeastern University

WHY THIS EXISTS:
  Manufacturing companies lose an average of $260,000 per hour during
  unplanned production stoppages (Aberdeen Research). Most supplier failures
  are detectable 2–3 weeks in advance — but only if you're monitoring the
  right signals. Most teams aren't. They track suppliers manually in
  spreadsheets and react after the line has already stopped.

WHAT IT DOES:
  SupplyIQ monitors 4 KPIs that together predict supplier failure before it
  happens: on-time delivery trend, defect rate, inventory coverage, and
  lead time variance. It scores each supplier's failure probability,
  classifies them as Stable / At Risk / Critical, and uses an AI model
  to generate the same risk brief a senior SC analyst would write manually —
  in under 30 seconds instead of 3 hours.

HOW IT SOLVES THE PROBLEM:
  1. Risk scoring model surfaces at-risk suppliers before a planner notices
  2. Single-source penalty (+10 pts) flags catastrophic exposure
  3. AI brief gives root cause + 3 actions + 30-day plan per supplier
  4. Portfolio briefing lets VP Operations see the whole picture in one view

STACK: Python · Streamlit · Plotly · Pandas · Groq (Llama 3, free)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import json

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SupplyIQ | Risk Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DESIGN SYSTEM ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    background-color: #0a0e17 !important;
    color: #c9d1d9 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stSidebar"] { background-color: #0d1117 !important; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

h1,h2,h3,h4,h5,h6 { font-family: 'IBM Plex Sans', sans-serif !important; color: #f0f6fc !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"]  { color: #f0f6fc !important; font-weight: 600 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricDelta"]  svg { display: none; }

/* Tabs */
[data-testid="stTabs"] button { color: #8b949e !important; background: transparent !important; font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.85rem !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

/* Dataframe */
[data-testid="stDataFrame"] { background: #0d1117 !important; border: 1px solid #21262d !important; border-radius: 8px !important; }
.stDataFrame th { background: #161b22 !important; color: #8b949e !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
.stDataFrame td { color: #c9d1d9 !important; background: #0d1117 !important; font-size: 0.85rem !important; }

/* Buttons */
[data-testid="stButton"] button {
    background: #238636 !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 500 !important;
}
[data-testid="stButton"] button:hover { background: #2ea043 !important; }

/* Selectbox, inputs */
[data-testid="stSelectbox"] > div { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 6px !important; color: #c9d1d9 !important; }
[data-testid="stSelectbox"] label { color: #8b949e !important; font-size: 0.8rem !important; }

/* Expander */
[data-testid="stExpander"] { background: #0d1117 !important; border: 1px solid #21262d !important; border-radius: 8px !important; }
[data-testid="stExpander"] summary { color: #58a6ff !important; font-weight: 500 !important; }
[data-testid="stExpander"] * { color: #c9d1d9 !important; }

/* Alert/info boxes */
[data-testid="stAlert"] { background: #0d1117 !important; border-radius: 8px !important; }
hr { border-color: #21262d !important; }

/* Custom components */
.hero-stat {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 600;
    color: #f85149;
    line-height: 1;
}
.hero-label {
    font-size: 0.78rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
.problem-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid #f85149;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.solution-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid #3fb950;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.kpi-row {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; padding: 5px 0;
    border-bottom: 1px solid #21262d;
    color: #c9d1d9;
}
.kpi-row:last-child { border-bottom: none; }
.kpi-key  { color: #8b949e; }
.kpi-val  { color: #f0f6fc; font-weight: 500; font-family: 'IBM Plex Mono', monospace; }
.section-label {
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.12em; color: #58a6ff;
    font-weight: 600; margin-bottom: 8px;
}
.ai-block {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid #3fb950;
    border-radius: 8px;
    padding: 18px 22px;
    font-size: 0.88rem;
    line-height: 1.8;
    color: #c9d1d9;
    white-space: pre-wrap;
    font-family: 'IBM Plex Sans', sans-serif;
}
.badge-critical { background:#3d1f1f; color:#f85149; padding:2px 10px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-risk     { background:#3d2f0f; color:#d29922; padding:2px 10px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-stable   { background:#0f2d1f; color:#3fb950; padding:2px 10px; border-radius:4px; font-size:0.75rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY DARK TEMPLATE ──────────────────────────────────────────────────────
DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0e17", plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="IBM Plex Sans"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(color="#8b949e")),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(color="#8b949e")),
    margin=dict(t=36, b=44, l=12, r=12),
)
COLOR_MAP = {"Critical": "#f85149", "At Risk": "#d29922", "Stable": "#3fb950"}

# ── GROQ CLIENT ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    key = st.secrets.get("GROQ_API_KEY", "")
    if key:
        return Groq(api_key=key)
    return None

# ── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.DataFrame([
        {"Supplier":"Apex Castings",       "Component":"Engine Block",          "Category":"Powertrain",   "Lead Time (days)":18,"Lead Time Variance (%)":32,"On-Time Delivery (%)":71,"Defect Rate (%)":3.8,"Inventory Coverage (days)":6, "Open POs":4, "Last SCAR":"45 days ago",      "Single Source":True,  "Country":"USA"},
        {"Supplier":"PrecisionFab Co.",    "Component":"Fuel Injector Housing",  "Category":"Fuel Systems", "Lead Time (days)":12,"Lead Time Variance (%)":8, "On-Time Delivery (%)":96,"Defect Rate (%)":0.4,"Inventory Coverage (days)":22,"Open POs":7, "Last SCAR":"None (12 months)", "Single Source":False, "Country":"USA"},
        {"Supplier":"KoreMetals Ltd.",     "Component":"Crankshaft",             "Category":"Powertrain",   "Lead Time (days)":35,"Lead Time Variance (%)":41,"On-Time Delivery (%)":64,"Defect Rate (%)":2.1,"Inventory Coverage (days)":4, "Open POs":3, "Last SCAR":"12 days ago",      "Single Source":True,  "Country":"South Korea"},
        {"Supplier":"ValveTech Industries","Component":"Exhaust Valve",          "Category":"Engine",       "Lead Time (days)":9, "Lead Time Variance (%)":12,"On-Time Delivery (%)":91,"Defect Rate (%)":0.9,"Inventory Coverage (days)":18,"Open POs":9, "Last SCAR":"None (8 months)",  "Single Source":False, "Country":"USA"},
        {"Supplier":"HydroSeal GmbH",      "Component":"Hydraulic Seal Kit",     "Category":"Hydraulics",   "Lead Time (days)":28,"Lead Time Variance (%)":55,"On-Time Delivery (%)":58,"Defect Rate (%)":5.2,"Inventory Coverage (days)":3, "Open POs":2, "Last SCAR":"7 days ago",       "Single Source":True,  "Country":"Germany"},
        {"Supplier":"NorthStar Bearings",  "Component":"Main Bearing Set",       "Category":"Powertrain",   "Lead Time (days)":14,"Lead Time Variance (%)":18,"On-Time Delivery (%)":88,"Defect Rate (%)":1.2,"Inventory Coverage (days)":14,"Open POs":6, "Last SCAR":"None (5 months)",  "Single Source":False, "Country":"Canada"},
        {"Supplier":"FastenerWorld",       "Component":"Structural Fasteners",   "Category":"Hardware",     "Lead Time (days)":5, "Lead Time Variance (%)":6, "On-Time Delivery (%)":98,"Defect Rate (%)":0.2,"Inventory Coverage (days)":45,"Open POs":12,"Last SCAR":"None (18 months)", "Single Source":False, "Country":"USA"},
        {"Supplier":"ElectroParts MX",     "Component":"ECM Wiring Harness",     "Category":"Electronics",  "Lead Time (days)":22,"Lead Time Variance (%)":29,"On-Time Delivery (%)":79,"Defect Rate (%)":1.8,"Inventory Coverage (days)":8, "Open POs":5, "Last SCAR":"30 days ago",      "Single Source":True,  "Country":"Mexico"},
        {"Supplier":"AlloyCraft Inc.",     "Component":"Turbo Housing",          "Category":"Engine",       "Lead Time (days)":20,"Lead Time Variance (%)":22,"On-Time Delivery (%)":85,"Defect Rate (%)":1.4,"Inventory Coverage (days)":11,"Open POs":4, "Last SCAR":"None (4 months)",  "Single Source":False, "Country":"USA"},
        {"Supplier":"GlobalGasket Co.",    "Component":"Head Gasket Assembly",   "Category":"Engine",       "Lead Time (days)":16,"Lead Time Variance (%)":48,"On-Time Delivery (%)":67,"Defect Rate (%)":4.1,"Inventory Coverage (days)":5, "Open POs":3, "Last SCAR":"21 days ago",      "Single Source":False, "Country":"China"},
        {"Supplier":"ThermoShield Corp.",  "Component":"Cooling System Module",  "Category":"Thermal",      "Lead Time (days)":11,"Lead Time Variance (%)":9, "On-Time Delivery (%)":94,"Defect Rate (%)":0.6,"Inventory Coverage (days)":20,"Open POs":8, "Last SCAR":"None (10 months)", "Single Source":False, "Country":"USA"},
        {"Supplier":"SteelForge Ltd.",     "Component":"Connecting Rod",         "Category":"Powertrain",   "Lead Time (days)":30,"Lead Time Variance (%)":37,"On-Time Delivery (%)":72,"Defect Rate (%)":2.6,"Inventory Coverage (days)":7, "Open POs":3, "Last SCAR":"18 days ago",      "Single Source":True,  "Country":"India"},
    ])

# ── RISK SCORING ──────────────────────────────────────────────────────────────
def compute_risk(row):
    s = 0
    otd = row["On-Time Delivery (%)"]
    dr  = row["Defect Rate (%)"]
    ic  = row["Inventory Coverage (days)"]
    ltv = row["Lead Time Variance (%)"]
    if   otd < 70: s += 35
    elif otd < 85: s += 20
    elif otd < 92: s += 8
    if   dr > 4:  s += 30
    elif dr > 2:  s += 18
    elif dr > 1:  s += 8
    if   ic < 5:  s += 25
    elif ic < 10: s += 14
    elif ic < 14: s += 6
    if   ltv > 40: s += 10
    elif ltv > 20: s += 5
    if row["Single Source"]: s += 10
    return min(s, 100)

def risk_label(score):
    if score >= 60: return "Critical", "#f85149"
    if score >= 35: return "At Risk",  "#d29922"
    return "Stable", "#3fb950"

def style_risk(v):
    m = {"Critical":"background:#3d1f1f;color:#f85149;font-weight:600",
         "At Risk":  "background:#3d2f0f;color:#d29922;font-weight:600",
         "Stable":   "background:#0f2d1f;color:#3fb950;font-weight:600"}
    return m.get(v, "")

def style_score(v):
    if v >= 60: return "color:#f85149;font-weight:600"
    if v >= 35: return "color:#d29922;font-weight:600"
    return "color:#3fb950;font-weight:600"

df = load_data()
df["Risk Score"] = df.apply(compute_risk, axis=1)
df["Risk Level"] = df["Risk Score"].apply(lambda s: risk_label(s)[0])

# ── AI ────────────────────────────────────────────────────────────────────────
def get_ai_insight(supplier_data: dict) -> str:
    client = get_client()
    if not client:
        return "⚠️ Add GROQ_API_KEY to Streamlit secrets to enable AI briefs."
    fleet = {
        "avg_otd":    round(df["On-Time Delivery (%)"].mean(), 1),
        "avg_defect": round(df["Defect Rate (%)"].mean(), 2),
        "avg_coverage": round(df["Inventory Coverage (days)"].mean(), 1),
        "critical_n": int((df["Risk Level"] == "Critical").sum()),
    }
    prompt = f"""You are a senior supply chain analyst at a Tier 1 automotive manufacturer.
Analyze this supplier KPI data and write a concise, actionable risk brief for a supply chain planning manager.

SUPPLIER DATA:
{json.dumps(supplier_data, indent=2)}

FLEET BENCHMARKS:
- Portfolio avg on-time delivery: {fleet['avg_otd']}%
- Portfolio avg defect rate: {fleet['avg_defect']}%
- Portfolio avg inventory coverage: {fleet['avg_coverage']} days
- Critical suppliers in portfolio: {fleet['critical_n']}

Use exactly this structure:

RISK SUMMARY — 2 sentences: primary risk and production impact
ROOT CAUSE HYPOTHESIS — 1–2 sentences: most likely operational cause
IMMEDIATE ACTIONS (3 bullet points with specific steps for this week)
30-DAY MITIGATION PLAN (2 bullet points of tactical improvements)
WATCH METRIC — 1 sentence: single KPI to monitor and escalation threshold

Be specific. Use supply chain terminology. Plain text only."""
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=600,
    )
    return r.choices[0].message.content

def get_portfolio_insight() -> str:
    client = get_client()
    if not client:
        return "⚠️ Add GROQ_API_KEY to Streamlit secrets to enable portfolio briefs."
    crit = df[df["Risk Level"] == "Critical"][
        ["Supplier","Component","On-Time Delivery (%)","Defect Rate (%)","Inventory Coverage (days)","Single Source"]
    ].to_dict(orient="records")
    risk = df[df["Risk Level"] == "At Risk"][["Supplier","Component","Risk Score"]].to_dict(orient="records")
    prompt = f"""You are a supply chain director preparing a weekly risk briefing for senior leadership.

CRITICAL SUPPLIERS ({len(crit)}):
{json.dumps(crit, indent=2)}

AT-RISK SUPPLIERS ({len(risk)}):
{json.dumps(risk, indent=2)}

PORTFOLIO STATS:
- Total suppliers: {len(df)}
- Single-source critical: {int(df[(df['Single Source']==True)&(df['Risk Level']=='Critical')].shape[0])}
- Avg portfolio OTD: {round(df['On-Time Delivery (%)'].mean(),1)}%
- Suppliers below 7 days coverage: {int((df['Inventory Coverage (days)']<7).sum())}

Structure:
PORTFOLIO RISK ASSESSMENT — 3 sentences on overall health, critical exposure, production risk
TOP 3 PRIORITY ACTIONS — ranked by urgency, one sentence each naming the supplier
SINGLE-SOURCE EXPOSURE — 2 sentences on risk concentration
RECOMMENDED PLANNING ADJUSTMENTS — 2 bullet points of specific parameter changes

Write for VP-level audience. Be direct. Use numbers."""
    r = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=700,
    )
    return r.choices[0].message.content

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:28px 0 8px">
  <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.18em;color:#58a6ff;font-weight:600;margin-bottom:6px">SUPPLY CHAIN INTELLIGENCE</div>
  <h1 style="font-size:2rem;font-weight:600;margin:0;color:#f0f6fc;letter-spacing:-0.02em">SupplyIQ</h1>
  <p style="color:#8b949e;font-size:0.9rem;margin-top:4px;max-width:620px">
    AI-powered supplier risk scoring and failure prediction. Surfaces at-risk suppliers 2–3 weeks before disruption — the same analysis a senior SC analyst performs manually, in under 30 seconds.
  </p>
</div>
""", unsafe_allow_html=True)

# ── PROBLEM / SOLUTION FRAMING ────────────────────────────────────────────────
with st.expander("📋 The problem this solves — and how", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div class="problem-box">
<div class="section-label">THE PROBLEM</div>
<p style="font-size:0.88rem;line-height:1.7;margin:0">
Manufacturing companies lose <strong style="color:#f85149">$260,000 per hour</strong> during unplanned production stoppages (Aberdeen Research). Supply chain planners at mid-size manufacturers monitor 20–80 suppliers simultaneously — manually, in spreadsheets. When on-time delivery drops or defect rates spike, the planner must identify the problem, understand why, and act. By the time they see it in a spreadsheet, the line has already stopped.
<br><br>
Most teams are <strong style="color:#f85149">reactive, not predictive</strong>. They respond to failures rather than preventing them.
</p>
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="solution-box">
<div class="section-label">HOW SUPPLYIQ SOLVES IT</div>
<p style="font-size:0.88rem;line-height:1.7;margin:0">
SupplyIQ scores each supplier across <strong style="color:#3fb950">4 leading indicators</strong> that together predict failure 2–3 weeks before it becomes a production event:
<br><br>
• <strong>On-time delivery trend</strong> — weighted most heavily; primary leading indicator<br>
• <strong>Defect rate</strong> — triggers SCAR workflow escalation when breached<br>
• <strong>Inventory coverage</strong> — below 7 days = immediate production risk<br>
• <strong>Lead time variance</strong> — high variance signals instability even when averages look OK
<br><br>
Single-source suppliers receive a <strong style="color:#d29922">+10 risk premium</strong>. The AI layer generates the root-cause brief a planner would write manually — in seconds.
</p>
</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px 20px;font-size:0.82rem;color:#8b949e">
<strong style="color:#c9d1d9">Demo data</strong> — 12 simulated suppliers from an automotive powertrain assembly program. Supplier names, KPIs, and SCAR dates are realistic but fictional. Replace with a CSV upload (coming soon) to run against your real supplier base.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── KPI STRIP ─────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Critical Suppliers",    int((df["Risk Level"]=="Critical").sum()))
c2.metric("At-Risk Suppliers",     int((df["Risk Level"]=="At Risk").sum()))
c3.metric("Single-Source Critical",int(df[(df["Single Source"]==True)&(df["Risk Level"]=="Critical")].shape[0]))
c4.metric("Low Coverage (<7 days)",int((df["Inventory Coverage (days)"]<7).sum()))
c5.metric("Avg On-Time Delivery",  f"{round(df['On-Time Delivery (%)'].mean(),1)}%")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️  Risk Dashboard",
    "🔍  Supplier Deep Dive + AI Brief",
    "📊  Portfolio AI Briefing",
    "📈  KPI Explorer",
    "📂  Data & Methodology",
])

# ── TAB 1: RISK DASHBOARD ─────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-label">SUPPLIER RISK OVERVIEW</div>', unsafe_allow_html=True)

    fig_scatter = px.scatter(
        df, x="On-Time Delivery (%)", y="Defect Rate (%)",
        size="Lead Time (days)", color="Risk Level",
        color_discrete_map=COLOR_MAP,
        hover_name="Supplier",
        hover_data={"Component":True,"Inventory Coverage (days)":True,
                    "Single Source":True,"Risk Score":True},
        size_max=32,
    )
    fig_scatter.add_vline(x=85, line_dash="dash", line_color="#30363d",
                          annotation_text="OTD threshold 85%",
                          annotation_font_color="#8b949e")
    fig_scatter.add_hline(y=2.0, line_dash="dash", line_color="#30363d",
                          annotation_text="Defect threshold 2%",
                          annotation_font_color="#8b949e")
    fig_scatter.update_layout(**DARK, height=400,
                              legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                         font=dict(color="#c9d1d9", size=12)))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('<div class="section-label" style="margin-top:8px">RANKED RISK REGISTER</div>', unsafe_allow_html=True)
    reg_cols = ["Supplier","Component","Category","Risk Level","Risk Score",
                "On-Time Delivery (%)","Defect Rate (%)","Inventory Coverage (days)",
                "Single Source","Last SCAR"]
    ranked = df[reg_cols].sort_values("Risk Score", ascending=False).reset_index(drop=True)
    st.dataframe(
        ranked.style.map(style_risk, subset=["Risk Level"]).map(style_score, subset=["Risk Score"]),
        use_container_width=True, hide_index=True, height=440,
    )

# ── TAB 2: SUPPLIER DEEP DIVE ─────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-label">SELECT SUPPLIER FOR AI RISK ANALYSIS</div>', unsafe_allow_html=True)
    selected = st.selectbox(
        "Supplier",
        df.sort_values("Risk Score", ascending=False)["Supplier"].tolist(),
        label_visibility="collapsed",
    )
    row = df[df["Supplier"] == selected].iloc[0]
    lbl, col = risk_label(row["Risk Score"])

    ca, cb = st.columns([1, 2])
    with ca:
        st.markdown(
            f"<p style='color:#f0f6fc;font-weight:600;font-size:0.95rem;margin-bottom:6px'>"
            f"{row['Supplier']} — {row['Component']}</p>"
            f"<span style='background:{col};color:#0a0e17;padding:3px 12px;"
            f"border-radius:4px;font-size:0.75rem;font-weight:700'>"
            f"{lbl} — {int(row['Risk Score'])}/100</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        kpis = {
            "On-time delivery":   f"{row['On-Time Delivery (%)']}%",
            "Defect rate":        f"{row['Defect Rate (%)']}%",
            "Inventory coverage": f"{int(row['Inventory Coverage (days)'])} days",
            "Lead time":          f"{int(row['Lead Time (days)'])} days",
            "Lead time variance": f"{row['Lead Time Variance (%)']}%",
            "Open POs":           str(int(row["Open POs"])),
            "Single source":      "Yes" if row["Single Source"] else "No",
            "Country":            row["Country"],
            "Last SCAR":          row["Last SCAR"],
        }
        for k, v in kpis.items():
            st.markdown(
                f"<div class='kpi-row'>"
                f"<span class='kpi-key'>{k}</span>"
                f"<span class='kpi-val'>{v}</span></div>",
                unsafe_allow_html=True,
            )

    with cb:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=int(row["Risk Score"]),
            title={"text":"Risk Score / 100", "font":{"size":12,"color":"#8b949e"}},
            number={"font":{"color":col,"size":44}},
            gauge={
                "axis":{"range":[0,100],"tickwidth":1,"tickfont":{"color":"#8b949e"}},
                "bar":{"color":col},
                "bgcolor":"#0d1117","bordercolor":"#21262d",
                "steps":[
                    {"range":[0, 35],"color":"#0f2d1f"},
                    {"range":[35,60],"color":"#3d2f0f"},
                    {"range":[60,100],"color":"#3d1f1f"},
                ],
                "threshold":{"line":{"color":"#f0f6fc","width":2},
                             "thickness":0.75,"value":int(row["Risk Score"])}
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
            font=dict(color="#c9d1d9"), height=240,
            margin=dict(t=30,b=10,l=20,r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # KPI bar chart for this supplier vs portfolio avg
        metrics = ["On-Time Delivery (%)","Defect Rate (%)","Lead Time Variance (%)"]
        labels  = ["On-Time Delivery","Defect Rate","Lead Time Variance"]
        sup_vals = [row[m] for m in metrics]
        avg_vals = [df[m].mean() for m in metrics]
        fig_kpi = go.Figure()
        fig_kpi.add_trace(go.Bar(name="This supplier", x=labels, y=sup_vals,
                                  marker_color=col, opacity=0.9))
        fig_kpi.add_trace(go.Bar(name="Portfolio avg", x=labels, y=avg_vals,
                                  marker_color="#30363d"))
        fig_kpi.update_layout(**DARK, height=200, barmode="group",
                               legend=dict(orientation="h",y=1.1,font=dict(size=11)),
                               margin=dict(t=20,b=30,l=0,r=0))
        st.plotly_chart(fig_kpi, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label">AI RISK BRIEF</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;font-size:0.8rem;margin-top:0'>Powered by Groq · Llama 3.1 · Free API</p>", unsafe_allow_html=True)

    if st.button("Generate AI Risk Brief", type="primary"):
        with st.spinner("Analysing KPI patterns and generating brief..."):
            d = row.to_dict()
            d = {k: (bool(v) if isinstance(v, bool) else (int(v) if hasattr(v,"item") else v))
                 for k, v in d.items()}
            brief = get_ai_insight(d)
        st.markdown(f'<div class="ai-block">{brief}</div>', unsafe_allow_html=True)

# ── TAB 3: PORTFOLIO AI BRIEFING ──────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-label">PORTFOLIO-LEVEL EXECUTIVE BRIEFING</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;font-size:0.82rem;margin-top:0'>VP-level summary across all critical and at-risk suppliers. Same analysis your SC director would prepare for Monday morning's review — generated in seconds.</p>", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        fig_bar = px.bar(
            df.sort_values("Risk Score", ascending=True),
            x="Risk Score", y="Supplier", orientation="h",
            color="Risk Level", color_discrete_map=COLOR_MAP,
        )
        fig_bar.update_layout(**DARK, height=420, showlegend=False,
                               xaxis_title="Risk Score", yaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)
    with cb:
        cat_risk = df.groupby("Category")["Risk Score"].mean().reset_index()
        fig_cat = px.bar(
            cat_risk.sort_values("Risk Score", ascending=False),
            x="Category", y="Risk Score",
            color="Risk Score",
            color_continuous_scale=["#3fb950","#d29922","#f85149"],
        )
        fig_cat.update_layout(**DARK, height=420, coloraxis_showscale=False,
                               xaxis_title="", yaxis_title="Avg Risk Score")
        st.plotly_chart(fig_cat, use_container_width=True)

    if st.button("Generate Portfolio AI Briefing", type="primary"):
        with st.spinner("Building executive risk summary..."):
            brief = get_portfolio_insight()
        st.markdown(f'<div class="ai-block">{brief}</div>', unsafe_allow_html=True)

# ── TAB 4: KPI EXPLORER ───────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-label">KPI DISTRIBUTION EXPLORER</div>', unsafe_allow_html=True)
    kpi_opts = ["On-Time Delivery (%)","Defect Rate (%)","Inventory Coverage (days)","Lead Time Variance (%)"]
    c1, c2 = st.columns(2)
    x_kpi = c1.selectbox("X axis", kpi_opts, index=0)
    y_kpi = c2.selectbox("Y axis", kpi_opts, index=1)
    fig_exp = px.scatter(
        df, x=x_kpi, y=y_kpi, color="Risk Level", color_discrete_map=COLOR_MAP,
        hover_name="Supplier",
        hover_data={"Component":True,"Category":True,"Risk Score":True},
        size=[14]*len(df), size_max=14,
    )
    fig_exp.update_layout(**DARK, height=380,
                           legend=dict(orientation="h",y=1.02,font=dict(size=12,color="#c9d1d9")))
    st.plotly_chart(fig_exp, use_container_width=True)
    st.markdown('<div class="section-label" style="margin-top:8px">SUMMARY STATISTICS BY RISK LEVEL</div>', unsafe_allow_html=True)
    st.dataframe(df.groupby("Risk Level")[kpi_opts].mean().round(2), use_container_width=True)

# ── TAB 5: DATA & METHODOLOGY ─────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-label">RISK SCORING METHODOLOGY</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:16px 20px;font-size:0.85rem;line-height:1.8;color:#c9d1d9;margin-bottom:16px">
Each supplier is scored 0–100 across four KPI dimensions. Scores above 60 are <span style="color:#f85149;font-weight:600">Critical</span>, 35–59 are <span style="color:#d29922;font-weight:600">At Risk</span>, and below 35 are <span style="color:#3fb950;font-weight:600">Stable</span>.
Single-source suppliers receive a +10 risk premium regardless of KPI performance — because even a stable single-source supplier represents a catastrophic concentration risk.
</div>
""", unsafe_allow_html=True)

    scoring_df = pd.DataFrame([
        ["On-time delivery", "Below 70%",    "+35 pts", "Primary leading indicator of disruption"],
        ["On-time delivery", "70–84%",       "+20 pts", ""],
        ["On-time delivery", "85–91%",       "+8 pts",  ""],
        ["Defect rate",      "Above 4%",     "+30 pts", "Triggers SCAR workflow at threshold"],
        ["Defect rate",      "2–4%",         "+18 pts", ""],
        ["Defect rate",      "1–2%",         "+8 pts",  ""],
        ["Inventory coverage","Below 5 days","+25 pts", "Immediate production stoppage risk"],
        ["Inventory coverage","5–9 days",    "+14 pts", ""],
        ["Inventory coverage","10–13 days",  "+6 pts",  ""],
        ["Lead time variance","Above 40%",   "+10 pts", "High variance = process instability"],
        ["Lead time variance","20–40%",      "+5 pts",  ""],
        ["Single source",    "Yes",          "+10 pts", "Concentration risk premium"],
    ], columns=["Dimension","Threshold","Points","Why it matters"])
    st.dataframe(scoring_df, use_container_width=True, hide_index=True, height=440)

    st.markdown('<div class="section-label" style="margin-top:16px">RAW SUPPLIER DATA</div>', unsafe_allow_html=True)
    disp_cols = ["Supplier","Component","Category","Country",
                 "On-Time Delivery (%)","Defect Rate (%)","Inventory Coverage (days)",
                 "Lead Time (days)","Lead Time Variance (%)","Open POs",
                 "Single Source","Last SCAR","Risk Level","Risk Score"]
    st.dataframe(
        df[disp_cols].sort_values("Risk Score", ascending=False).reset_index(drop=True)
           .style.map(style_risk, subset=["Risk Level"]).map(style_score, subset=["Risk Score"]),
        use_container_width=True, hide_index=True, height=440,
    )
    csv = df[disp_cols].to_csv(index=False).encode()
    st.download_button("Download Dataset (CSV)", csv, "supplyiq_data.csv", "text/csv")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='font-size:0.75rem;color:#484f58;text-align:center'>"
    "SupplyIQ · Supply Chain Risk Intelligence · "
    "AI by Groq (Llama 3.1, free) · "
    "Built by Rutwik Satish · MS Engineering Management, Northeastern University"
    "</p>",
    unsafe_allow_html=True,
)
