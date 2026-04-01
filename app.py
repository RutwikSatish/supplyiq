import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import json

st.set_page_config(
    page_title="SupplyIQ: Supply Chain Risk Intelligence",
    page_icon="🏭",
    layout="wide"
)

# ── FORCE DARK THEME ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global background and text */
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stMain"], [data-testid="block-container"] {
    background-color: #0f1117 !important;
    color: #e8e8e8 !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #161b22 !important;
  }
  [data-testid="stSidebar"] * { color: #e8e8e8 !important; }

  /* All text elements */
  h1, h2, h3, h4, h5, h6, p, span, div, label,
  .stMarkdown, .stText, .stCaption { color: #e8e8e8 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background-color: #1c2333 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 12px !important;
  }
  [data-testid="metric-container"] * { color: #e8e8e8 !important; }
  [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 600 !important; }
  [data-testid="stMetricLabel"] { color: #8b949e !important; }

  /* Tabs */
  [data-testid="stTabs"] button {
    color: #8b949e !important;
    background-color: transparent !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
  }

  /* Dataframe */
  [data-testid="stDataFrame"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
  }
  .stDataFrame th {
    background-color: #21262d !important;
    color: #c9d1d9 !important;
    font-weight: 600 !important;
  }
  .stDataFrame td { color: #e8e8e8 !important; background-color: #161b22 !important; }

  /* Select boxes and inputs */
  [data-testid="stSelectbox"] > div,
  [data-testid="stMultiSelect"] > div {
    background-color: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e8e8e8 !important;
    border-radius: 6px !important;
  }
  [data-testid="stSelectbox"] label,
  [data-testid="stMultiSelect"] label { color: #8b949e !important; font-size: 13px !important; }

  /* Slider */
  [data-testid="stSlider"] label { color: #8b949e !important; }
  [data-testid="stSlider"] * { color: #e8e8e8 !important; }

  /* Buttons */
  [data-testid="stButton"] button {
    background-color: #238636 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
  }
  [data-testid="stButton"] button:hover {
    background-color: #2ea043 !important;
  }

  /* Download button */
  [data-testid="stDownloadButton"] button {
    background-color: #21262d !important;
    color: #58a6ff !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
  }

  /* Expander */
  [data-testid="stExpander"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
  }
  [data-testid="stExpander"] * { color: #e8e8e8 !important; }
  [data-testid="stExpander"] summary { color: #58a6ff !important; font-weight: 500 !important; }

  /* Spinner */
  [data-testid="stSpinner"] * { color: #58a6ff !important; }

  /* Horizontal rule */
  hr { border-color: #30363d !important; }

  /* Caption text */
  .stCaption, [data-testid="stCaptionContainer"] { color: #8b949e !important; }

  /* Error and info boxes */
  [data-testid="stAlert"] { background-color: #21262d !important; border-radius: 8px !important; }

  /* Table in markdown */
  table { background-color: #161b22 !important; color: #e8e8e8 !important; border-collapse: collapse !important; width: 100% !important; }
  th { background-color: #21262d !important; color: #c9d1d9 !important; padding: 8px 12px !important; border: 1px solid #30363d !important; }
  td { color: #e8e8e8 !important; padding: 8px 12px !important; border: 1px solid #30363d !important; }
  tr:nth-child(even) td { background-color: #1c2333 !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #161b22; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY DARK TEMPLATE ──────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0f1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e8e8e8", family="sans-serif"),
    xaxis=dict(gridcolor="#30363d", linecolor="#30363d", tickfont=dict(color="#8b949e")),
    yaxis=dict(gridcolor="#30363d", linecolor="#30363d", tickfont=dict(color="#8b949e")),
    margin=dict(t=30, b=40, l=10, r=10),
)

# ── GROQ CLIENT ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_client()

# ── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    companies = [
        {"Supplier": "Apex Castings",       "Component": "Engine Block",          "Category": "Powertrain",
         "Lead Time (days)": 18, "Lead Time Variance (%)": 32, "On-Time Delivery (%)": 71,
         "Defect Rate (%)": 3.8, "Inventory Coverage (days)": 6,  "Open POs": 4,
         "Last SCAR": "45 days ago",       "Single Source": True,  "Country": "USA"},
        {"Supplier": "PrecisionFab Co.",    "Component": "Fuel Injector Housing",  "Category": "Fuel Systems",
         "Lead Time (days)": 12, "Lead Time Variance (%)":  8, "On-Time Delivery (%)": 96,
         "Defect Rate (%)": 0.4, "Inventory Coverage (days)": 22, "Open POs": 7,
         "Last SCAR": "None (12 months)",  "Single Source": False, "Country": "USA"},
        {"Supplier": "KoreMetals Ltd.",     "Component": "Crankshaft",             "Category": "Powertrain",
         "Lead Time (days)": 35, "Lead Time Variance (%)": 41, "On-Time Delivery (%)": 64,
         "Defect Rate (%)": 2.1, "Inventory Coverage (days)":  4, "Open POs": 3,
         "Last SCAR": "12 days ago",       "Single Source": True,  "Country": "South Korea"},
        {"Supplier": "ValveTech Industries","Component": "Exhaust Valve",          "Category": "Engine",
         "Lead Time (days)":  9, "Lead Time Variance (%)": 12, "On-Time Delivery (%)": 91,
         "Defect Rate (%)": 0.9, "Inventory Coverage (days)": 18, "Open POs": 9,
         "Last SCAR": "None (8 months)",   "Single Source": False, "Country": "USA"},
        {"Supplier": "HydroSeal GmbH",      "Component": "Hydraulic Seal Kit",     "Category": "Hydraulics",
         "Lead Time (days)": 28, "Lead Time Variance (%)": 55, "On-Time Delivery (%)": 58,
         "Defect Rate (%)": 5.2, "Inventory Coverage (days)":  3, "Open POs": 2,
         "Last SCAR": "7 days ago",        "Single Source": True,  "Country": "Germany"},
        {"Supplier": "NorthStar Bearings",  "Component": "Main Bearing Set",       "Category": "Powertrain",
         "Lead Time (days)": 14, "Lead Time Variance (%)": 18, "On-Time Delivery (%)": 88,
         "Defect Rate (%)": 1.2, "Inventory Coverage (days)": 14, "Open POs": 6,
         "Last SCAR": "None (5 months)",   "Single Source": False, "Country": "Canada"},
        {"Supplier": "FastenerWorld",       "Component": "Structural Fasteners",   "Category": "Hardware",
         "Lead Time (days)":  5, "Lead Time Variance (%)":  6, "On-Time Delivery (%)": 98,
         "Defect Rate (%)": 0.2, "Inventory Coverage (days)": 45, "Open POs": 12,
         "Last SCAR": "None (18 months)",  "Single Source": False, "Country": "USA"},
        {"Supplier": "ElectroParts MX",     "Component": "ECM Wiring Harness",     "Category": "Electronics",
         "Lead Time (days)": 22, "Lead Time Variance (%)": 29, "On-Time Delivery (%)": 79,
         "Defect Rate (%)": 1.8, "Inventory Coverage (days)":  8, "Open POs": 5,
         "Last SCAR": "30 days ago",       "Single Source": True,  "Country": "Mexico"},
        {"Supplier": "AlloyCraft Inc.",     "Component": "Turbo Housing",          "Category": "Engine",
         "Lead Time (days)": 20, "Lead Time Variance (%)": 22, "On-Time Delivery (%)": 85,
         "Defect Rate (%)": 1.4, "Inventory Coverage (days)": 11, "Open POs": 4,
         "Last SCAR": "None (4 months)",   "Single Source": False, "Country": "USA"},
        {"Supplier": "GlobalGasket Co.",    "Component": "Head Gasket Assembly",   "Category": "Engine",
         "Lead Time (days)": 16, "Lead Time Variance (%)": 48, "On-Time Delivery (%)": 67,
         "Defect Rate (%)": 4.1, "Inventory Coverage (days)":  5, "Open POs": 3,
         "Last SCAR": "21 days ago",       "Single Source": False, "Country": "China"},
        {"Supplier": "ThermoShield Corp.",  "Component": "Cooling System Module",  "Category": "Thermal",
         "Lead Time (days)": 11, "Lead Time Variance (%)":  9, "On-Time Delivery (%)": 94,
         "Defect Rate (%)": 0.6, "Inventory Coverage (days)": 20, "Open POs": 8,
         "Last SCAR": "None (10 months)",  "Single Source": False, "Country": "USA"},
        {"Supplier": "SteelForge Ltd.",     "Component": "Connecting Rod",         "Category": "Powertrain",
         "Lead Time (days)": 30, "Lead Time Variance (%)": 37, "On-Time Delivery (%)": 72,
         "Defect Rate (%)": 2.6, "Inventory Coverage (days)":  7, "Open POs": 3,
         "Last SCAR": "18 days ago",       "Single Source": True,  "Country": "India"},
    ]
    return pd.DataFrame(companies)

df = load_data()

# ── RISK SCORING ──────────────────────────────────────────────────────────────
def compute_risk(row):
    s = 0
    if   row["On-Time Delivery (%)"] < 70: s += 35
    elif row["On-Time Delivery (%)"] < 85: s += 20
    elif row["On-Time Delivery (%)"] < 92: s += 8
    if   row["Defect Rate (%)"] > 4:  s += 30
    elif row["Defect Rate (%)"] > 2:  s += 18
    elif row["Defect Rate (%)"] > 1:  s += 8
    if   row["Inventory Coverage (days)"] < 5:  s += 25
    elif row["Inventory Coverage (days)"] < 10: s += 14
    elif row["Inventory Coverage (days)"] < 14: s += 6
    if   row["Lead Time Variance (%)"] > 40: s += 10
    elif row["Lead Time Variance (%)"] > 20: s += 5
    if row["Single Source"]: s += 10
    return min(s, 100)

def risk_label(score):
    if score >= 60: return "Critical", "#f85149"
    if score >= 35: return "At Risk",  "#d29922"
    return "Stable", "#3fb950"

df["Risk Score"] = df.apply(compute_risk, axis=1)
df["Risk Level"] = df["Risk Score"].apply(lambda s: risk_label(s)[0])

COLOR_MAP = {"Critical": "#f85149", "At Risk": "#d29922", "Stable": "#3fb950"}

# ── CELL STYLERS (dark-safe) ──────────────────────────────────────────────────
def style_risk(v):
    m = {"Critical": "background-color:#3d1f1f;color:#f85149;font-weight:600",
         "At Risk":  "background-color:#3d2f0f;color:#d29922;font-weight:600",
         "Stable":   "background-color:#0f2d1f;color:#3fb950;font-weight:600"}
    return m.get(v, "")

def style_score(v):
    if v >= 60: return "color:#f85149;font-weight:600"
    if v >= 35: return "color:#d29922;font-weight:600"
    return "color:#3fb950;font-weight:600"

# ── AI ────────────────────────────────────────────────────────────────────────
def get_ai_insight(supplier_data: dict) -> str:
    fleet = {
        "avg_otd":      round(df["On-Time Delivery (%)"].mean(), 1),
        "avg_defect":   round(df["Defect Rate (%)"].mean(), 2),
        "avg_coverage": round(df["Inventory Coverage (days)"].mean(), 1),
        "critical_n":   int((df["Risk Level"] == "Critical").sum()),
    }
    prompt = f"""You are a senior supply chain analyst at a manufacturing company similar to Cummins.
Analyze this supplier KPI data and write a concise, actionable risk brief for a supply chain planning manager.

SUPPLIER DATA:
{json.dumps(supplier_data, indent=2)}

FLEET BENCHMARKS:
- Portfolio avg on-time delivery: {fleet['avg_otd']}%
- Portfolio avg defect rate: {fleet['avg_defect']}%
- Portfolio avg inventory coverage: {fleet['avg_coverage']} days
- Critical suppliers in portfolio: {fleet['critical_n']}

Use exactly this structure:

**Risk Summary** (2 sentences: primary risk and production impact)
**Root Cause Hypothesis** (1-2 sentences: most likely operational reason)
**Immediate Actions** (3 bullet points: specific steps for this week)
**30-Day Mitigation Plan** (2 bullet points: tactical improvements)
**Watch Metric** (1 sentence: single KPI to monitor and escalation threshold)

Be specific. Use supply chain terminology. No generic advice."""
    r = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=600,
    )
    return r.choices[0].message.content

def get_portfolio_insight() -> str:
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
- Suppliers with coverage below 7 days: {int((df['Inventory Coverage (days)']<7).sum())}

Use exactly this structure:

**Portfolio Risk Assessment** (3 sentences: overall health, critical exposure, immediate production risk)
**Top 3 Priority Actions** (ranked by urgency, one sentence each naming the supplier)
**Single-Source Exposure** (2 sentences: risk concentration assessment)
**Recommended Planning System Adjustments** (2 bullet points: specific parameter changes)

Write for a VP-level audience. Be direct. Use numbers."""
    r = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=700,
    )
    return r.choices[0].message.content

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='color:#e8e8e8;margin-bottom:2px'>SupplyIQ — Supply Chain Risk Intelligence</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color:#8b949e;font-size:13px;margin-top:0'>AI-powered KPI monitoring and failure prediction for integrated supply chain planning</p>",
    unsafe_allow_html=True
)

with st.expander("What does this app do?", expanded=False):
    st.markdown("""
**SupplyIQ** monitors supplier KPIs, scores each supplier's failure probability, and uses a
large language model (Llama 3 via Groq) to generate actionable risk briefs — replicating the
weekly planning review a senior SC analyst performs manually, in under 30 seconds.

**The problem it solves**

Supply chain planners at manufacturing companies monitor dozens of suppliers simultaneously.
When on-time delivery drops, defect rates spike, or inventory coverage falls below safe levels,
the planner must (1) identify at-risk suppliers, (2) understand why, and (3) act before a
production line stops. Most teams do this manually in spreadsheets — slow, inconsistent, reactive.

**How the risk score works**

Each supplier is scored across four KPI dimensions:
- **On-time delivery** — weighted most heavily; primary leading indicator of supply disruption
- **Defect rate** — triggers SCAR workflows when breached
- **Inventory coverage** — below 7 days signals immediate production risk
- **Lead time variance** — high variance signals supplier instability even when averages look acceptable

Single-source suppliers receive an additional +10 risk premium. Scores are capped at 100 and
classified as **Stable (0–34)**, **At Risk (35–59)**, or **Critical (60+)**.

**The AI layer**

Clicking "Generate AI risk brief" sends the supplier's KPI data plus portfolio benchmarks to
Llama 3 via Groq. The model returns: root cause hypothesis, three immediate actions,
a 30-day mitigation plan, and the single watch metric. The portfolio briefing generates a
VP-level executive summary across all critical and at-risk suppliers simultaneously.
    """)

st.markdown("---")

# ── METRICS ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Critical suppliers",     int((df["Risk Level"]=="Critical").sum()))
c2.metric("At-risk suppliers",      int((df["Risk Level"]=="At Risk").sum()))
c3.metric("Single-source critical", int(df[(df["Single Source"]==True)&(df["Risk Level"]=="Critical")].shape[0]))
c4.metric("Low coverage (<7 days)", int((df["Inventory Coverage (days)"]<7).sum()))
c5.metric("Avg on-time delivery",   f"{round(df['On-Time Delivery (%)'].mean(),1)}%")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Risk dashboard", "Supplier deep dive + AI brief",
    "Portfolio AI briefing", "KPI explorer", "Data preview"
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<h4 style='color:#e8e8e8'>Supplier risk overview</h4>", unsafe_allow_html=True)

    fig = px.scatter(
        df, x="On-Time Delivery (%)", y="Defect Rate (%)",
        size="Lead Time (days)", color="Risk Level",
        color_discrete_map=COLOR_MAP,
        hover_name="Supplier",
        hover_data={"Component": True, "Inventory Coverage (days)": True,
                    "Single Source": True, "Risk Score": True},
        size_max=30,
    )
    fig.add_vline(x=85, line_dash="dash", line_color="#8b949e",
                  annotation_text="OTD threshold 85%",
                  annotation_font_color="#8b949e")
    fig.add_hline(y=2.0, line_dash="dash", line_color="#8b949e",
                  annotation_text="Defect threshold 2%",
                  annotation_font_color="#8b949e")
    fig.update_layout(**DARK_LAYOUT, height=400,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  font=dict(color="#e8e8e8")))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h4 style='color:#e8e8e8'>Ranked risk register</h4>", unsafe_allow_html=True)
    reg_cols = ["Supplier","Component","Category","Risk Level","Risk Score",
                "On-Time Delivery (%)","Defect Rate (%)","Inventory Coverage (days)",
                "Single Source","Last SCAR"]
    ranked = df[reg_cols].sort_values("Risk Score", ascending=False).reset_index(drop=True)
    st.dataframe(
        ranked.style.map(style_risk, subset=["Risk Level"]).map(style_score, subset=["Risk Score"]),
        use_container_width=True, hide_index=True, height=420
    )

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<h4 style='color:#e8e8e8'>Select a supplier for AI risk analysis</h4>",
                unsafe_allow_html=True)
    selected = st.selectbox(
        "Supplier",
        df.sort_values("Risk Score", ascending=False)["Supplier"].tolist()
    )
    row  = df[df["Supplier"] == selected].iloc[0]
    lbl, col = risk_label(row["Risk Score"])

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown(f"<p style='color:#e8e8e8;font-weight:600;font-size:15px'>{row['Supplier']} — {row['Component']}</p>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<span style='background:{col};color:#0f1117;padding:3px 12px;"
            f"border-radius:6px;font-size:12px;font-weight:700'>"
            f"{lbl} — {int(row['Risk Score'])}/100</span>",
            unsafe_allow_html=True
        )
        st.markdown("")
        kpis = {
            "On-time delivery":     f"{row['On-Time Delivery (%)']}%",
            "Defect rate":          f"{row['Defect Rate (%)']}%",
            "Inventory coverage":   f"{int(row['Inventory Coverage (days)'])} days",
            "Lead time":            f"{int(row['Lead Time (days)'])} days",
            "Lead time variance":   f"{row['Lead Time Variance (%)']}%",
            "Open POs":             str(int(row["Open POs"])),
            "Single source":        "Yes" if row["Single Source"] else "No",
            "Country":              row["Country"],
            "Last SCAR":            row["Last SCAR"],
        }
        for k, v in kpis.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:13px;padding:4px 0;border-bottom:1px solid #30363d'>"
                f"<span style='color:#8b949e'>{k}</span>"
                f"<span style='color:#e8e8e8;font-weight:500'>{v}</span></div>",
                unsafe_allow_html=True
            )
    with col_b:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=int(row["Risk Score"]),
            title={"text": "Risk score", "font": {"size": 13, "color": "#8b949e"}},
            number={"font": {"color": col, "size": 40}},
            gauge={
                "axis": {"range": [0,100], "tickwidth": 1,
                         "tickfont": {"color": "#8b949e"}},
                "bar": {"color": col},
                "bgcolor": "#161b22",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0,  35], "color": "#0f2d1f"},
                    {"range": [35, 60], "color": "#3d2f0f"},
                    {"range": [60,100], "color": "#3d1f1f"},
                ],
                "threshold": {"line": {"color": "#e8e8e8","width": 2},
                              "thickness": 0.75, "value": int(row["Risk Score"])}
            }
        ))
        fig_g.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                            font=dict(color="#e8e8e8"), height=240,
                            margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("---")
    st.markdown("<h4 style='color:#e8e8e8'>AI risk brief</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;font-size:12px'>Powered by Groq (Llama 3 8B)</p>",
                unsafe_allow_html=True)
    if st.button("Generate AI risk brief", type="primary"):
        with st.spinner("Analyzing KPI patterns..."):
            d = row.to_dict()
            d = {k: (bool(v) if isinstance(v, bool)
                     else (int(v) if hasattr(v, "item") else v))
                 for k, v in d.items()}
            try:
                st.markdown(
                    f"<div style='background:#161b22;border:1px solid #30363d;"
                    f"border-radius:8px;padding:16px 20px;color:#e8e8e8;font-size:14px'>"
                    f"{get_ai_insight(d).replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Groq API error: {e}")

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<h4 style='color:#e8e8e8'>Portfolio-level AI executive briefing</h4>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;font-size:12px'>Powered by Groq (Llama 3 8B)</p>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(
            df.sort_values("Risk Score", ascending=True),
            x="Risk Score", y="Supplier", orientation="h",
            color="Risk Level", color_discrete_map=COLOR_MAP,
        )
        fig_bar.update_layout(**DARK_LAYOUT, height=420, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        cat_risk = df.groupby("Category")["Risk Score"].mean().reset_index()
        fig_cat = px.bar(
            cat_risk.sort_values("Risk Score", ascending=False),
            x="Category", y="Risk Score",
            color="Risk Score",
            color_continuous_scale=["#3fb950","#d29922","#f85149"],
        )
        fig_cat.update_layout(**DARK_LAYOUT, height=420, coloraxis_showscale=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    if st.button("Generate portfolio AI briefing", type="primary"):
        with st.spinner("Preparing executive risk briefing..."):
            try:
                briefing = get_portfolio_insight()
                st.markdown(
                    f"<div style='background:#161b22;border:1px solid #30363d;"
                    f"border-radius:8px;padding:16px 20px;color:#e8e8e8;font-size:14px'>"
                    f"{briefing.replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Groq API error: {e}")

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<h4 style='color:#e8e8e8'>KPI distribution explorer</h4>",
                unsafe_allow_html=True)
    kpi_opts = ["On-Time Delivery (%)","Defect Rate (%)","Inventory Coverage (days)","Lead Time Variance (%)"]
    c1, c2 = st.columns(2)
    x_kpi = c1.selectbox("X axis", kpi_opts, index=0)
    y_kpi = c2.selectbox("Y axis", kpi_opts, index=1)

    fig_exp = px.scatter(
        df, x=x_kpi, y=y_kpi,
        color="Risk Level", color_discrete_map=COLOR_MAP,
        hover_name="Supplier",
        hover_data={"Component": True, "Category": True, "Risk Score": True},
        size=[14]*len(df), size_max=14,
    )
    fig_exp.update_layout(**DARK_LAYOUT, height=380,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      font=dict(color="#e8e8e8")))
    st.plotly_chart(fig_exp, use_container_width=True)

    st.markdown("<h4 style='color:#e8e8e8'>Summary statistics by risk level</h4>",
                unsafe_allow_html=True)
    st.dataframe(df.groupby("Risk Level")[kpi_opts].mean().round(2),
                 use_container_width=True)

# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("<h4 style='color:#e8e8e8'>Raw supplier dataset</h4>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;font-size:13px'>Underlying data powering the risk scoring model.</p>",
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    f_cat  = c1.selectbox("Category",    ["All"] + sorted(df["Category"].unique().tolist()))
    f_risk = c2.selectbox("Risk level",  ["All","Critical","At Risk","Stable"])
    f_ss   = c3.selectbox("Single source", ["All","Yes","No"])

    prev = df.copy()
    if f_cat  != "All": prev = prev[prev["Category"]   == f_cat]
    if f_risk != "All": prev = prev[prev["Risk Level"] == f_risk]
    if f_ss   != "All": prev = prev[prev["Single Source"] == (f_ss == "Yes")]

    st.markdown(f"<p style='color:#8b949e;font-size:13px'>Showing <b style='color:#e8e8e8'>{len(prev)}</b> of <b style='color:#e8e8e8'>{len(df)}</b> suppliers</p>",
                unsafe_allow_html=True)

    prev_cols = ["Supplier","Component","Category","Country",
                 "On-Time Delivery (%)","Defect Rate (%)","Inventory Coverage (days)",
                 "Lead Time (days)","Lead Time Variance (%)","Open POs",
                 "Single Source","Last SCAR","Risk Level","Risk Score"]
    st.dataframe(
        prev[prev_cols].sort_values("Risk Score", ascending=False)
                       .reset_index(drop=True)
                       .style.map(style_risk, subset=["Risk Level"])
                             .map(style_score, subset=["Risk Score"]),
        use_container_width=True, hide_index=True, height=440
    )

    st.markdown("<h4 style='color:#e8e8e8'>KPI distribution by category</h4>",
                unsafe_allow_html=True)
    dist_kpi = st.selectbox("KPI", kpi_opts, key="dist_kpi")
    fig_box = px.box(
        prev, x="Category", y=dist_kpi,
        color="Risk Level", color_discrete_map=COLOR_MAP,
        points="all", hover_name="Supplier",
    )
    fig_box.update_layout(**DARK_LAYOUT, height=340,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      font=dict(color="#e8e8e8")))
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("<h4 style='color:#e8e8e8'>Risk scoring methodology</h4>",
                unsafe_allow_html=True)
    st.markdown("""
| Dimension | Threshold | Points |
|-----------|-----------|--------|
| On-time delivery | Below 70% | +35 |
| On-time delivery | 70–84% | +20 |
| On-time delivery | 85–91% | +8 |
| Defect rate | Above 4% | +30 |
| Defect rate | 2–4% | +18 |
| Defect rate | 1–2% | +8 |
| Inventory coverage | Below 5 days | +25 |
| Inventory coverage | 5–9 days | +14 |
| Inventory coverage | 10–13 days | +6 |
| Lead time variance | Above 40% | +10 |
| Lead time variance | 20–40% | +5 |
| Single source | Yes | +10 |

Score capped at 100. Stable = 0–34, At Risk = 35–59, Critical = 60+.
    """)

    csv = prev[prev_cols].to_csv(index=False).encode()
    st.download_button("Download filtered dataset as CSV", csv,
                       "supplyiq_data.csv", "text/csv")
