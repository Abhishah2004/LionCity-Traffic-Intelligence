import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import json
import base64

st.set_page_config(page_title="LionCity Traffic Intelligence", page_icon="🚦", layout="wide")

with open("singapore.geojson", "r", encoding="utf-8") as f:
    sg_geojson = json.load(f)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf            = joblib.load('rf_model.pkl')
    le            = joblib.load('le.pkl')
    oe            = joblib.load('oe.pkl')
    speed_ranges  = joblib.load('speed_ranges.pkl')
    day_options   = sorted(joblib.load('day_options.pkl'))
    event_options = ['None'] + sorted(joblib.load('event_options.pkl'))
    shap_background = joblib.load('shap_background.pkl')
    explainer = shap.TreeExplainer(rf, shap_background)
    return rf, le, oe, speed_ranges, day_options, event_options, explainer

rf, le, oe, speed_ranges, day_options, event_options, explainer = load_models()
CAT_COLS = ['day_of_week', 'special_event_type']

# ── Festival/Event Location Mapping (static, manually curated) ─────────────
# Each event maps to a LIST of affected zones.

EVENT_LOCATIONS = {
    'none': [],

    'chingay_parade': [
        {'name': 'Marina Bay (F1 Pit Building)', 'lat': 1.2914, 'lon': 103.8607, 'radius_km': 1.5},
        {'name': 'Promenade / City Hall (spillover)', 'lat': 1.2930, 'lon': 103.8550, 'radius_km': 1.0},
    ],

    'cny_eve': [
        {'name': 'Chinatown', 'lat': 1.2820, 'lon': 103.8440, 'radius_km': 1.2},
        {'name': 'Orchard Road (shopping crowds)', 'lat': 1.3048, 'lon': 103.8318, 'radius_km': 1.2},
    ],
    'cny_day1': [
        {'name': 'Chinatown', 'lat': 1.2820, 'lon': 103.8440, 'radius_km': 1.2},
    ],
    'cny_day2': [
        {'name': 'Chinatown', 'lat': 1.2820, 'lon': 103.8440, 'radius_km': 1.2},
    ],

    'christmas_festive': [
        {'name': 'Orchard Road', 'lat': 1.3048, 'lon': 103.8318, 'radius_km': 1.5},
        {'name': 'Marina Bay (light show crowds)', 'lat': 1.2834, 'lon': 103.8607, 'radius_km': 1.2},
    ],
    'christmas_eve_leadup': [
        {'name': 'Orchard Road', 'lat': 1.3048, 'lon': 103.8318, 'radius_km': 1.5},
    ],
    'christmas_eve': [
        {'name': 'Orchard Road', 'lat': 1.3048, 'lon': 103.8318, 'radius_km': 1.5},
        {'name': 'Marina Bay', 'lat': 1.2834, 'lon': 103.8607, 'radius_km': 1.2},
    ],
    'christmas_day': [
        {'name': 'Orchard Road', 'lat': 1.3048, 'lon': 103.8318, 'radius_km': 1.5},
    ],
    'christmas_holiday_week': [
        {'name': 'Orchard Road', 'lat': 1.3048, 'lon': 103.8318, 'radius_km': 1.5},
    ],
    'christmas_week': [
        {'name': 'Orchard Road', 'lat': 1.3048, 'lon': 103.8318, 'radius_km': 1.5},
    ],
    'festive_buildup': [
        {'name': 'Orchard Road', 'lat': 1.3048, 'lon': 103.8318, 'radius_km': 1.5},
    ],

    'nye_buildup': [
        {'name': 'Marina Bay', 'lat': 1.2834, 'lon': 103.8607, 'radius_km': 1.8},
    ],
    'nye_countdown': [
        {'name': 'Marina Bay', 'lat': 1.2834, 'lon': 103.8607, 'radius_km': 1.8},
        {'name': 'Sentosa', 'lat': 1.2494, 'lon': 103.8160, 'radius_km': 1.2},
    ],
    'new_year': [
        {'name': 'Marina Bay', 'lat': 1.2834, 'lon': 103.8607, 'radius_km': 1.8},
    ],

    'world_aquatics': [
        {'name': 'OCBC Aquatic Centre, Kallang', 'lat': 1.3033, 'lon': 103.8748, 'radius_km': 1.0},
    ],
    'world_aquatics_food_fest': [
        {'name': 'OCBC Aquatic Centre, Kallang', 'lat': 1.3033, 'lon': 103.8748, 'radius_km': 1.0},
    ],
    'aquatics_buildup': [
        {'name': 'OCBC Aquatic Centre, Kallang', 'lat': 1.3033, 'lon': 103.8748, 'radius_km': 1.0},
    ],
    'aquatics_tail': [
        {'name': 'OCBC Aquatic Centre, Kallang', 'lat': 1.3033, 'lon': 103.8748, 'radius_km': 1.0},
    ],
    'aquatics_end': [
        {'name': 'OCBC Aquatic Centre, Kallang', 'lat': 1.3033, 'lon': 103.8748, 'radius_km': 1.0},
    ],
    'racial_harmony_world_aquatics': [
        {'name': 'OCBC Aquatic Centre, Kallang', 'lat': 1.3033, 'lon': 103.8748, 'radius_km': 1.0},
    ],

    'motorshow': [
        {'name': 'Suntec Singapore', 'lat': 1.2936, 'lon': 103.8573, 'radius_km': 0.8},
    ],

    'fiba_3x3_qualifying': [
        {'name': 'Padang / Marina Bay', 'lat': 1.2907, 'lon': 103.8520, 'radius_km': 1.0},
    ],
    'fiba_3x3_main_draw': [
        {'name': 'Padang / Marina Bay', 'lat': 1.2907, 'lon': 103.8520, 'radius_km': 1.0},
    ],

    'hari_raya_puasa': [
        {'name': 'Kampong Glam / Geylang Serai', 'lat': 1.3171, 'lon': 103.8985, 'radius_km': 1.3},
        {'name': 'Sultan Mosque area', 'lat': 1.3023, 'lon': 103.8590, 'radius_km': 0.8},
    ],
    'hari_raya_eve': [
        {'name': 'Geylang Serai Bazaar', 'lat': 1.3171, 'lon': 103.8985, 'radius_km': 1.3},
    ],
    'hari_raya_haji': [
        {'name': 'Kampong Glam / Geylang Serai', 'lat': 1.3171, 'lon': 103.8985, 'radius_km': 1.3},
        {'name': 'Sultan Mosque area', 'lat': 1.3023, 'lon': 103.8590, 'radius_km': 0.8},
    ],

    'singapore_sevens': [
        {'name': 'National Stadium, Kallang', 'lat': 1.3034, 'lon': 103.8745, 'radius_km': 1.0},
    ],

    'qingming_festival': [
        {'name': 'Choa Chu Kang Cemetery', 'lat': 1.3850, 'lon': 103.7200, 'radius_km': 2.0},
        {'name': 'Mandai Columbarium', 'lat': 1.4090, 'lon': 103.7890, 'radius_km': 1.5},
    ],

    'good_friday_weekend': [
        {'name': 'Church of St Mary of the Angels', 'lat': 1.3185, 'lon': 103.7475, 'radius_km': 1.0},
    ],
    'good_friday': [
        {'name': 'Church of St Mary of the Angels', 'lat': 1.3185, 'lon': 103.7475, 'radius_km': 1.0},
    ],

    'labour_day': [],  # public holiday, no specific zone — citywide light traffic

    'national_day_buildup': [
        {'name': 'Padang / Marina Bay', 'lat': 1.2907, 'lon': 103.8520, 'radius_km': 1.5},
    ],
    'national_day_parade': [
        {'name': 'Padang / Marina Bay', 'lat': 1.2907, 'lon': 103.8520, 'radius_km': 1.5},
        {'name': 'City Hall / Suntec (spillover)', 'lat': 1.2936, 'lon': 103.8573, 'radius_km': 1.0},
    ],
    'ndp_buildup': [
        {'name': 'Padang / Marina Bay', 'lat': 1.2907, 'lon': 103.8520, 'radius_km': 1.5},
    ],

    'night_festival': [
        {'name': 'Bras Basah (SAM, NMS)', 'lat': 1.2967, 'lon': 103.8500, 'radius_km': 1.0},
    ],
    'hungry_ghost_night_festival': [
        {'name': 'Chinatown', 'lat': 1.2820, 'lon': 103.8440, 'radius_km': 1.5},
        {'name': 'Lorong Koo Chye Sheng Hong Temple', 'lat': 1.3231, 'lon': 103.8838, 'radius_km': 1.0},
    ],

    'painting_with_light': [
        {'name': 'Civic District / Riverside', 'lat': 1.2900, 'lon': 103.8500, 'radius_km': 1.0},
    ],
    'river_festival_design_week_painting_with_light': [
        {'name': 'Civic District / Riverside', 'lat': 1.2900, 'lon': 103.8500, 'radius_km': 1.0},
    ],
    'river_festival_design_week': [
        {'name': 'Civic District / Riverside', 'lat': 1.2900, 'lon': 103.8500, 'radius_km': 1.0},
    ],
    'design_week_painting_with_light': [
        {'name': 'Civic District / Riverside', 'lat': 1.2900, 'lon': 103.8500, 'radius_km': 1.0},
    ],

    'mid_autumn': [
        {'name': 'Chinatown', 'lat': 1.2820, 'lon': 103.8440, 'radius_km': 1.2},
        {'name': 'Gardens by the Bay (lantern displays)', 'lat': 1.2816, 'lon': 103.8636, 'radius_km': 1.0},
    ],

    'f1_grand_prix': [
        {'name': 'Marina Bay Street Circuit', 'lat': 1.2914, 'lon': 103.8642, 'radius_km': 2.0},
        {'name': 'City Hall / Esplanade (road closures)', 'lat': 1.2930, 'lon': 103.8550, 'radius_km': 1.2},
    ],
    'post_f1': [
        {'name': 'Marina Bay Street Circuit', 'lat': 1.2914, 'lon': 103.8642, 'radius_km': 2.0},
    ],

    'deepavali': [
        {'name': 'Little India', 'lat': 1.3066, 'lon': 103.8517, 'radius_km': 1.0},
        {'name': 'Mustafa Centre area', 'lat': 1.3110, 'lon': 103.8553, 'radius_km': 0.6},
    ],
    'pre_deepavali': [
        {'name': 'Little India', 'lat': 1.3066, 'lon': 103.8517, 'radius_km': 1.0},
    ],
    'post_deepavali': [
        {'name': 'Little India', 'lat': 1.3066, 'lon': 103.8517, 'radius_km': 1.0},
    ],

    'singapore_writers_festival': [
        {'name': 'Civic District / Bras Basah', 'lat': 1.2967, 'lon': 103.8500, 'radius_km': 1.0},
    ],
    'orchard_lightup_swf': [
        {'name': 'Orchard Road', 'lat': 1.3048, 'lon': 103.8318, 'radius_km': 1.5},
    ],
    'swf_orchard': [
        {'name': 'Orchard Road', 'lat': 1.3048, 'lon': 103.8318, 'radius_km': 1.5},
    ],
    'apex_expo_swf': [
        {'name': 'Singapore Expo', 'lat': 1.3346, 'lon': 103.9613, 'radius_km': 1.0},
    ],
    'swf': [
        {'name': 'Civic District / Bras Basah', 'lat': 1.2967, 'lon': 103.8500, 'radius_km': 1.0},
    ],
    'swf_closing': [
        {'name': 'Civic District / Bras Basah', 'lat': 1.2967, 'lon': 103.8500, 'radius_km': 1.0},
    ],

    'symphonic_gala': [
        {'name': 'Esplanade', 'lat': 1.2899, 'lon': 103.8559, 'radius_km': 0.8},
    ],

    'valentines_day': [
        {'name': 'Orchard Road', 'lat': 1.3048, 'lon': 103.8318, 'radius_km': 1.5},
        {'name': 'Marina Bay (dinner crowds)', 'lat': 1.2834, 'lon': 103.8607, 'radius_km': 1.0},
    ],

    'pink_dot_sg': [
        {'name': 'Hong Lim Park', 'lat': 1.2854, 'lon': 103.8455, 'radius_km': 0.7},
    ],

    'expo_residual': [
        {'name': 'Singapore Expo', 'lat': 1.3346, 'lon': 103.9613, 'radius_km': 1.0},
    ],
    'expo_events': [
        {'name': 'Singapore Expo', 'lat': 1.3346, 'lon': 103.9613, 'radius_km': 1.0},
    ],

    'armed_forces_day': [
        {'name': 'Padang / Marina Bay', 'lat': 1.2907, 'lon': 103.8520, 'radius_km': 1.2},
    ],

    'halloween': [
        {'name': 'Sentosa', 'lat': 1.2494, 'lon': 103.8160, 'radius_km': 1.2},
        {'name': 'Clarke Quay', 'lat': 1.2884, 'lon': 103.8467, 'radius_km': 0.8},
    ],

    'zoukout': [
        {'name': 'Siloso Beach, Sentosa', 'lat': 1.2494, 'lon': 103.8160, 'radius_km': 1.0},
    ],

    'concert': [
        {'name': 'National Stadium, Kallang', 'lat': 1.3034, 'lon': 103.8745, 'radius_km': 1.0},
    ],

    'vesak_day': [
        {'name': 'Buddha Tooth Relic Temple, Chinatown', 'lat': 1.2815, 'lon': 103.8443, 'radius_km': 1.0},
        {'name': 'Kong Meng San Phor Kark See Monastery', 'lat': 1.3625, 'lon': 103.8344, 'radius_km': 1.0},
    ],
}



# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;
}

.main { background-color: #080c14; }

.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 1400px;
}
/* ── Control Column (replaces sidebar) ── */
.control-panel {
    background: linear-gradient(180deg, #0d1117 0%, #0a0f1a 100%);
    border: 1px solid #1e2d40;
    border-radius: 16px;
    padding: 1.2rem 1.2rem 1.5rem 1.2rem;
}
.control-panel .stSlider label,
.control-panel .stSelectbox label {
    color: #94a3b8 !important;
    font-size: 0.78em !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Slider track */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #06b6d4) !important;
}

/* ── Section headers in control panel ── */
.sid-head {
    font-family: 'Syne', sans-serif;
    font-size: 0.7em;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3b82f6 !important;
    padding: 0.9rem 0 0.3rem 0;
    border-bottom: 1px solid #1e2d40;
    margin-bottom: 0.3rem;
}

/* ── Main header ── */
.app-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.app-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8em;
    font-weight: 800;
    background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.02em;
}
.app-header p {
    color: #64748b;
    font-size: 0.95em;
    margin: 0.4rem 0 0 0;
    font-weight: 300;
}
.app-header .badge {
    display: inline-block;
    background: #0f2044;
    border: 1px solid #1e3a6e;
    color: #60a5fa !important;
    font-size: 0.7em;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}

/* ── Result card ── */
.result-wrap {
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.result-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    opacity: 0.07;
    background: white;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 3.2em;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0.2rem 0;
    line-height: 1;
}
.result-sub {
    font-size: 0.9em;
    font-weight: 400;
    margin: 0.5rem 0 0 0;
    opacity: 0.85;
}
.confidence-pill {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 30px;
    font-size: 0.82em;
    font-weight: 600;
    margin-top: 0.8rem;
    letter-spacing: 0.03em;
}

/* ── Speed range box ── */
.speed-box {
    margin-top: 1.2rem;
    padding: 0.8rem 1.4rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid #1e2d40;
    border-radius: 12px;
    display: inline-block;
}
.speed-box .speed-label {
    font-size: 0.7em;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 4px;
}
.speed-box .speed-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5em;
    font-weight: 800;
}

/* HIGH */
.high-card {
    background: linear-gradient(135deg, #1a0505 0%, #2d0a0a 100%);
    border: 1.5px solid #7f1d1d;
    box-shadow: 0 0 40px rgba(239,68,68,0.15), 0 8px 32px rgba(0,0,0,0.4);
}
.high-label { color: #fca5a5; }
.high-pill  { background: rgba(239,68,68,0.2); color: #fca5a5; border: 1px solid #ef4444; }
.high-sub   { color: #f87171; }

/* MEDIUM */
.med-card {
    background: linear-gradient(135deg, #1a1000 0%, #2d1f00 100%);
    border: 1.5px solid #78350f;
    box-shadow: 0 0 40px rgba(245,158,11,0.15), 0 8px 32px rgba(0,0,0,0.4);
}
.med-label { color: #fcd34d; }
.med-pill  { background: rgba(245,158,11,0.2); color: #fcd34d; border: 1px solid #f59e0b; }
.med-sub   { color: #fbbf24; }

/* LOW */
.low-card {
    background: linear-gradient(135deg, #011a0a 0%, #022d12 100%);
    border: 1.5px solid #14532d;
    box-shadow: 0 0 40px rgba(34,197,94,0.15), 0 8px 32px rgba(0,0,0,0.4);
}
.low-label { color: #86efac; }
.low-pill  { background: rgba(34,197,94,0.2); color: #86efac; border: 1px solid #22c55e; }
.low-sub   { color: #4ade80; }

/* ── Prob bars ── */
.prob-container {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin-top: 0.5rem;
}
.metric-card {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.metric-card .m-label {
    font-size: 0.7em;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
    margin-bottom: 4px;
}
.metric-card .m-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.3em;
    font-weight: 700;
    color: #e2e8f0;
}

/* ── Section title ── */
.sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75em;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin: 1.8rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2d40;
}

/* ── Divider ── */
hr { border-color: #1e2d40 !important; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

with open("merlion.jpg", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode()

st.markdown(f"""
<style>
.hero-banner {{
    height: 320px;
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 20px;

    background-image:
        linear-gradient(
            rgba(0,0,0,0.25),
            rgba(0,0,0,0.45)
        ),
        url("data:image/jpeg;base64,{img_base64}");

    background-size: cover;
    background-position: center 25%;
    display: flex;
    align-items: center;
    justify-content: center;
}}

.hero-content {{
    text-align: center;
}}

.hero-content h1 {{
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    color: white;
    margin-bottom: 10px;
}}

.hero-content p {{
    color: #e2e8f0;
    font-size: 1.1rem;
}}
</style>

<div class="hero-banner">
    <div class="hero-content">
        <h1> LionCity Traffic Intelligence</h1>
        <p>Singapore Traffic Congestion Forecasting & Urban Mobility Analytics</p>
        <span class="badge">⚡ Powered by Random Forest · 85.5% Accuracy</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:1rem; padding:10px 14px; background:#0d1117;
            border:1px solid #1e2d40; border-left:3px solid #f59e0b;
            border-radius:8px;">
    <span style="font-size:0.8em; color:#94a3b8;">
        ⏳ <b>Note:</b> Predictions may take a few seconds to load due to real-time SHAP 
        explainability computation. Thanks for your patience!
    </span>
</div>
""", unsafe_allow_html=True)


# ── Layout: Control Column + Main Column (replaces sidebar) ───────────────────
control_col, main_col = st.columns([1, 2.6], gap="large")

with control_col:
    st.markdown("""
    <div class="control-panel">
    <div style='text-align:center; padding: 0 0 1rem 0;'>
        <div style='font-family:Syne,sans-serif; font-size:1.1em; font-weight:800;
                    color:#60a5fa !important; letter-spacing:0.05em;'>SCENARIO CONTROLS</div>
        <div style='font-size:0.72em; color:#475569 !important; margin-top:3px;'>
            Adjust inputs to simulate traffic conditions
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sid-head">🕐 Time</div>', unsafe_allow_html=True)
    hour  = st.slider("Hour of Day",  0, 23,  8)
    month = st.slider("Month",        1, 12,  6)

    st.markdown('<div class="sid-head">📅 Day</div>', unsafe_allow_html=True)
    day_of_week       = st.selectbox("Day of Week",        day_options)
    is_weekend        = st.selectbox("Is Weekend?",        ["No", "Yes"])
    is_public_holiday = st.selectbox("Is Public Holiday?", ["No", "Yes"])
    # ── Auto-derived: School Hour Logic ─────────────────────────────────────────
    # Schools run 7-9 AM, but not on weekends or public holidays
    school_hour_flag_value = "Yes" if (
        hour in [7, 8, 9] and is_weekend == "No" and is_public_holiday == "No"
    ) else "No"
    workday_flag      = st.selectbox("Workday?",           ["No", "Yes"])

    st.markdown('<div class="sid-head">🌤️ Weather</div>', unsafe_allow_html=True)
    temperature_celsius = st.slider("Temperature (°C)",      20,   40,   28)
    rainfall_mm_per_hr  = st.slider("Rainfall (mm/hr)",       0,   60,    0)
    humidity_percent    = st.slider("Humidity (%)",           65,  100,   80)
    wind_speed_kmh      = st.slider("Wind Speed (km/h)",       0,   25,   10)
    air_quality_index   = st.slider("Air Quality Index",      35,   70,   50)

    st.markdown('<div class="sid-head">🧳 Tourism</div>', unsafe_allow_html=True)
    tourist_arrival_idx = st.slider("Tourist Arrival Index",  0.0,  1.0,  0.5, step=0.01)

    st.markdown('<div class="sid-head">🎉 Event</div>', unsafe_allow_html=True)
    special_event_type = st.selectbox("Special Event Type", event_options)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Build Input ───────────────────────────────────────────────────────────────
def build_input():
    row = pd.DataFrame([{
        'hour':                  hour,
        'is_weekend':            1 if is_weekend == "Yes" else 0,
        'month':                 month,
        'is_public_holiday':     1 if is_public_holiday == "Yes" else 0,
        'temperature_celsius':   temperature_celsius,
        'rainfall_mm_per_hr':    rainfall_mm_per_hr,
        'humidity_percent':      humidity_percent,
        'wind_speed_kmh':        wind_speed_kmh,
        'air_quality_index_aqi': air_quality_index,
        'tourist_arrival_index': tourist_arrival_idx,
        'school_hour_flag':      1 if school_hour_flag_value == "Yes" else 0,
        'workday_flag':          1 if workday_flag == "Yes" else 0,
        'day_of_week':           day_of_week,
        'special_event_type':    special_event_type
    }])
    CAT_COLS = ['day_of_week', 'special_event_type']
    row[CAT_COLS] = oe.transform(row[CAT_COLS])
    return row

# ── Predict ───────────────────────────────────────────────────────────────────
input_row     = build_input()
prediction    = rf.predict(input_row)[0]
probabilities = rf.predict_proba(input_row)[0]
label_map     = {i: cls for i, cls in enumerate(le.classes_)}
label         = label_map[int(prediction)]
confidence    = max(probabilities) * 100
sp            = speed_ranges[label]

# ── Style Maps ────────────────────────────────────────────────────────────────
card_class  = {"High": "high-card",  "Medium": "med-card",  "Low": "low-card"}
label_class = {"High": "high-label", "Medium": "med-label", "Low": "low-label"}
pill_class  = {"High": "high-pill",  "Medium": "med-pill",  "Low": "low-pill"}
sub_class   = {"High": "high-sub",   "Medium": "med-sub",   "Low": "low-sub"}
emoji_map   = {"High": "🔴",          "Medium": "🟡",         "Low": "🟢"}
bar_color   = {"High": "linear-gradient(90deg,#ef4444,#dc2626)",
               "Medium": "linear-gradient(90deg,#f59e0b,#d97706)",
               "Low":  "linear-gradient(90deg,#22c55e,#16a34a)"}
name_color_map = {"High": "#fca5a5", "Medium": "#fcd34d", "Low": "#86efac"}
desc_map    = {
    "High":   "Heavy congestion detected — expect significant delays.",
    "Medium": "Moderate traffic — minor slowdowns on key routes.",
    "Low":    "Clear roads ahead — traffic flowing smoothly."
}

with main_col:
    # ── Layout ────────────────────────────────────────────────────────────────────
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        # Result card
        st.markdown(f"""
        <div class="result-wrap {card_class[label]}">
            <div style="font-size:0.75em; text-transform:uppercase; letter-spacing:0.12em;
                        color:#475569; font-weight:600; margin-bottom:0.5rem;">
                PREDICTED CONGESTION LEVEL
            </div>
            <div class="result-label {label_class[label]}">{emoji_map[label]} {label.upper()}</div>
            <div class="result-sub {sub_class[label]}">{desc_map[label]}</div>
            <div class="confidence-pill {pill_class[label]}">
                ⚡ {confidence:.1f}% confidence
            </div>
            <div class="speed-box">
                <div class="speed-label">🚗 Expected Vehicle Speed</div>
                <div class="speed-value" style="color:{name_color_map[label]};">
                    {sp['low']} – {sp['high']} km/h
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Scenario summary metrics (no speed)
        st.markdown('<div class="sec-title">📋 Scenario Snapshot</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="m-label">🕐 Hour</div>
                <div class="m-value">{hour:02d}:00</div>
            </div>
            <div class="metric-card">
                <div class="m-label">📅 Day</div>
                <div class="m-value">{day_of_week[:3]}</div>
            </div>
            <div class="metric-card">
                <div class="m-label">🌡️ Temp</div>
                <div class="m-value">{temperature_celsius}°C</div>
            </div>
            <div class="metric-card">
                <div class="m-label">🌧️ Rain</div>
                <div class="m-value">{rainfall_mm_per_hr} mm</div>
            </div>
            <div class="metric-card">
                <div class="m-label">💧 Humidity</div>
                <div class="m-value">{humidity_percent}%</div>
            </div>
            <div class="metric-card">
                <div class="m-label">💨 Wind</div>
                <div class="m-value">{wind_speed_kmh} km/h</div>
            </div>
            <div class="metric-card">
                <div class="m-label">🌫️ AQI</div>
                <div class="m-value">{air_quality_index}</div>
            </div>
            <div class="metric-card">
                <div class="m-label">🧳 Tourism</div>
                <div class="m-value">{tourist_arrival_idx:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        # Probability breakdown
        st.markdown('<div class="sec-title">📊 Probability Breakdown</div>', unsafe_allow_html=True)

        prob_data = sorted(zip(le.classes_, probabilities), key=lambda x: x[1], reverse=True)

        for cls, prob in prob_data:
            pct      = prob * 100
            clr      = bar_color[cls]
            name_clr = name_color_map[cls]
            st.markdown(f"""
            <div style="margin-bottom:1.1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-family:'Syne',sans-serif; font-size:0.85em; font-weight:700;
                                 letter-spacing:0.05em; text-transform:uppercase; color:{name_clr};">
                        {emoji_map[cls]} {cls}
                    </span>
                    <span style="font-size:0.9em; font-weight:600; color:{name_clr};">{pct:.1f}%</span>
                </div>
                <div style="background:#1e2d40; border-radius:8px; height:8px; overflow:hidden;">
                    <div style="width:{pct}%; height:8px; border-radius:8px; background:{clr};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Key conditions panel
        st.markdown('<div class="sec-title">🔍 Key Conditions</div>', unsafe_allow_html=True)

        flags = []
        if hour in range(7, 10) or hour in range(17, 20):
            flags.append(("🚨", "Rush Hour Active", "#ef4444"))
        if is_weekend == "Yes":
            flags.append(("🏖️", "Weekend", "#22c55e"))
        if is_public_holiday == "Yes":
            flags.append(("🎉", "Public Holiday", "#f59e0b"))
        if rainfall_mm_per_hr > 20:
            flags.append(("🌧️", "Heavy Rainfall", "#60a5fa"))
        if school_hour_flag_value == "Yes":
            flags.append(("🏫", "School Hours", "#f59e0b"))
        if special_event_type.lower() not in ['none', 'no event', 'normal', '']:
            flags.append(("🎪", f"Event: {special_event_type}", "#a78bfa"))

        if not flags:
            flags.append(("✅", "Normal conditions", "#64748b"))

        flags_html = ""
        for icon, text, color in flags:
            flags_html += f"""
            <div style="display:flex; align-items:center; gap:10px; padding:8px 14px;
                        background:#0d1117; border:1px solid #1e2d40; border-radius:10px;
                        margin-bottom:8px;">
                <span style="font-size:1.1em;">{icon}</span>
                <span style="font-size:0.85em; font-weight:500; color:{color};">{text}</span>
            </div>
            """
        st.markdown(flags_html, unsafe_allow_html=True)

        # ──  DISCLAIMER BLOCK  ──────────────────────────────────
        st.markdown("""
        <div style="margin-top:10px; padding:10px 14px; background:#0d1117;
                    border:1px solid #1e2d40; border-left:3px solid #3b82f6;
                    border-radius:8px;">
            <span style="font-size:0.78em; color:#64748b;">
                ℹ️ <b>Note:</b> Hours 7–9 AM are treated as school hours, provided it is not a 
                weekend or public holiday. If the holiday or weekend flag is active, 
                school hours are not considered.
            </span>
        </div>
        """, unsafe_allow_html=True)


    # ── Raw features expander ─────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("🔬 View Raw Feature Vector Sent to Model"):
        st.dataframe(input_row, use_container_width=True)


    # ── SHAP Explanation ──────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🧠 Why This Prediction?</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b; font-size:0.85em; margin-top:-0.5rem; margin-bottom:1rem;'>"
        "How each factor pushed the prediction toward or away from the predicted class.</p>",
        unsafe_allow_html=True
    )

    shap_values = explainer.shap_values(input_row)

    # shap_values is a list of arrays (one per class) for multiclass RF
    class_idx = int(prediction)
    sv = shap_values[class_idx][0] if isinstance(shap_values, list) else shap_values[0, :, class_idx]

    feature_names = input_row.columns.tolist()
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': sv
    }).sort_values('shap_value', key=abs, ascending=False)

    fig_shap = go.Figure()
    colors_shap = ['#22c55e' if v > 0 else '#ef4444' for v in shap_df['shap_value']]

    fig_shap.add_trace(go.Bar(
        x=shap_df['shap_value'],
        y=shap_df['feature'],
        orientation='h',
        marker_color=colors_shap,
        hovertemplate="<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>"
    ))

    fig_shap.update_layout(
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(family='DM Sans', color='#94a3b8'),
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
        xaxis=dict(
            title=f"Impact on prediction: {label}",
            showgrid=True,
            gridcolor='#1e2d40',
            zerolinecolor='#475569',
            tickfont=dict(size=11, color='#475569')
        ),
        yaxis=dict(
            tickfont=dict(size=11, color='#cbd5e1'),
            autorange='reversed'
        )
    )

    st.plotly_chart(fig_shap, use_container_width=True)
    st.markdown(
        "<p style='color:#475569; font-size:0.78em; margin-top:-0.5rem;'>"
        "🟢 Green pushes toward this prediction · 🔴 Red pushes away from it</p>",
        unsafe_allow_html=True
    )

    # ── Affected Zones Map ───────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🗺️ Affected Zones Map</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b; font-size:0.85em; margin-top:-0.5rem; margin-bottom:1rem;'>"
        "Areas affected by school hours and selected events, colored by predicted congestion.</p>",
        unsafe_allow_html=True
    )

    map_zones = []

    # School zones (active only during 7-9 AM on school days)
    if school_hour_flag_value == "Yes":
        SCHOOL_ZONES = [
            {'name': 'Bukit Timah School Belt', 'lat': 1.3294, 'lon': 103.8021, 'radius_km': 1.2},
            {'name': 'Marine Parade School Cluster', 'lat': 1.3030, 'lon': 103.9070, 'radius_km': 1.0},
            {'name': 'Ang Mo Kio School Cluster', 'lat': 1.3691, 'lon': 103.8454, 'radius_km': 1.0},
            {'name': 'Clementi School Belt', 'lat': 1.3151, 'lon': 103.7654, 'radius_km': 1.0},
        ]
        for zone in SCHOOL_ZONES:
            map_zones.append({**zone, 'category': 'School Zone'})

    # Event zones (active when an event is selected)
    event_zones = EVENT_LOCATIONS.get(special_event_type, [])
    for zone in event_zones:
        map_zones.append({**zone, 'category': 'Event Zone'})

    if map_zones:
        congestion_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}[label]

        map_df = pd.DataFrame(map_zones)
        map_df['color'] = congestion_color
        map_df['size'] = map_df['radius_km'] * 18

        area_names = [
            feature["properties"]["PLN_AREA_N"]
            for feature in sg_geojson["features"]
        ]

        area_df = pd.DataFrame({
            "PLN_AREA_N": area_names,
            "value": 1
        })

        fig_map = go.Figure()

        fig_map.add_trace(
        go.Choroplethmapbox(
            geojson=sg_geojson,
            locations=area_df["PLN_AREA_N"],
            z=area_df["value"],
            text=area_df["PLN_AREA_N"],
            featureidkey="properties.PLN_AREA_N",
            hovertemplate="<b>%{text}</b><extra></extra>",
            showscale=False,
            colorscale=[[0, "#1e293b"], [1, "#1e293b"]],
            marker_opacity=0.25,
            marker_line_width=1
        )

        )

        fig_map.add_trace(go.Scattermapbox(
            lat=map_df['lat'],
            lon=map_df['lon'],
            mode='markers',
            marker=dict(
                size=map_df['size'],
                color=congestion_color,
                opacity=0.45,
            ),
            text=map_df['name'] + " (" + map_df['category'] + ")",
            hovertemplate="<b>%{text}</b><br>Predicted: " + label + "<extra></extra>",
            showlegend=False
        ))

        # Smaller solid center dot for clarity
        fig_map.add_trace(go.Scattermapbox(
            lat=map_df['lat'],
            lon=map_df['lon'],
            mode='markers',
            marker=dict(size=8, color=congestion_color),
            text=map_df['name'],
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False
        ))

        fig_map.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(
                lat=1.3521,
                lon=103.8198
            ),
            zoom=11.2
        ),
        uirevision="locked",
        paper_bgcolor='#0d1117',
        margin=dict(l=0, r=0, t=0, b=0),
        height=420
    )
    

        st.plotly_chart(fig_map, use_container_width=True)

        # Legend / zone list
        zone_list_html = ""
        for zone in map_zones:
            icon = "🏫" if zone['category'] == 'School Zone' else "🎉"
            zone_list_html += f"""
            <div style="display:flex; align-items:center; gap:10px; padding:6px 14px;
                        background:#0d1117; border:1px solid #1e2d40; border-radius:10px;
                        margin-bottom:6px;">
                <span>{icon}</span>
                <span style="font-size:0.82em; color:#cbd5e1;">{zone['name']}</span>
                <span style="font-size:0.75em; color:#64748b; margin-left:auto;">{zone['category']}</span>
            </div>
            """
        st.markdown(zone_list_html, unsafe_allow_html=True)
    else:
        st.markdown(
            "<p style='color:#475569; font-size:0.85em;'>No school hours or special events active for current settings.</p>",
            unsafe_allow_html=True
        )



