"""
Collective Demand Forecast — Smart batching for fast responses.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
import sys
import time
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.styling import get_custom_css
from app.utils.geo_utils import (
    get_all_area_names,
    get_hexagons_for_input,
)

st.set_page_config(page_title="Demand Forecast", page_icon="🔮", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.title("🔮 Collective Demand Forecast")
st.markdown(
    "Enter a street address, community area, or H3 index. "
    "We predict demand across hexagons in that region and aggregate the results."
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Max hexagons to predict for (prevents mega-batches)
MAX_HEXAGONS = 20
BATCH_CHUNK_SIZE = 50  # Send in chunks of this size


def safe_predict_batch(predictions_list, timeout=15):
    """Send batch prediction with chunking and timeout."""
    all_results = []

    # Split into chunks
    for i in range(0, len(predictions_list), BATCH_CHUNK_SIZE):
        chunk = predictions_list[i:i + BATCH_CHUNK_SIZE]
        try:
            r = requests.post(
                f"{API_URL}/predict/batch",
                json={"predictions": chunk},
                timeout=timeout,
            )
            if r.status_code == 200:
                all_results.extend(r.json()["predictions"])
            else:
                st.warning(f"Chunk {i//BATCH_CHUNK_SIZE + 1} failed: {r.status_code}")
        except requests.Timeout:
            st.warning(f"Chunk {i//BATCH_CHUNK_SIZE + 1} timed out")
        except Exception as e:
            st.warning(f"Chunk error: {e}")

    return all_results



# Background API Startup (For Streamlit Deployment environment)
@st.cache_resource
def ensure_api_running():
    import requests
    import subprocess
    import sys
    try:
        r = requests.get(f"{API_URL}/health", timeout=1)
        if r.status_code == 200:
            return True
    except:
        pass
    cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True

ensure_api_running()

api_online = False
try:
    r = requests.get(f"{API_URL}/health", timeout=3)
    if r.status_code == 200:
        api_online = True
        health = r.json()
        st.sidebar.success(f"✅ API Online — Deploy Mode Active")
except Exception:
    pass

if not api_online:
    st.sidebar.warning("⏳ API Initializing...")
    st.info("The forecasting engine is spinning up in the background (model weights: 7.5M+ parameters). Please wait up to 60 seconds and refresh the page.")
    st.stop()



st.markdown("### 📍 Location & Environment")
col_loc, col_time, col_weather = st.columns([2, 1, 1])

with col_loc:
    search_mode = st.radio(
        "Search Method",
        ["Street Address", "Community Area", "Manual H3"],
        horizontal=True,
    )

    if search_mode == "Street Address":
        address = st.text_input(
            "Street Address",
            value="",
            placeholder="e.g., 233 S Wacker Drive",
        )
        ring_size = st.slider(
            "Neighborhood Radius",
            min_value=1, max_value=3, value=1,
            help="1 = 7 hexagons, 2 = 19, 3 = 37",
        )
        area_name = None
        h3_manual = None

    elif search_mode == "Community Area":
        area_names = get_all_area_names()
        if area_names:
            default_idx = area_names.index("Loop") if "Loop" in area_names else 0
            area_name = st.selectbox("Community Area", area_names, index=default_idx)
        else:
            st.warning("No community area data found.")
            area_name = None
        address = None
        h3_manual = None
        ring_size = 0

    else:
        h3_manual = st.text_input("H3 Index", value="882a100d27fffff")
        ring_size = st.slider("Include Neighbors", 0, 2, 1)
        address = None
        area_name = None

with col_time:
    pred_date = st.date_input("Date", value=datetime.now().date())
    pred_hour = st.slider("Hour", 0, 23, datetime.now().hour)

with col_weather:
    temperature = st.number_input("Temp (°F)", value=55, min_value=-20, max_value=120)
    humidity = st.number_input("Humidity (%)", value=65, min_value=0, max_value=100)
    wind_speed = st.number_input("Wind (mph)", value=10, min_value=0, max_value=80)
    precipitation = st.number_input("Precip (in)", value=0.0, min_value=0.0,
                                     max_value=5.0, step=0.01)



hexagons, hex_description = get_hexagons_for_input(
    mode=search_mode,
    address=address,
    area_name=area_name,
    h3_index=h3_manual,
    ring_size=ring_size,
)

if not hexagons:
    st.warning("No hexagons found. Try a different location.")
    st.stop()

# Cap hexagons for performance
original_count = len(hexagons)
if len(hexagons) > MAX_HEXAGONS:
    # Sample representative hexagons
    hexagons_sampled = [hexagons[0]] + list(
        np.random.choice(hexagons[1:], size=MAX_HEXAGONS - 1, replace=False)
    )
    scale_factor = original_count / MAX_HEXAGONS
    st.info(f"{hex_description}\n\n"
            f"📊 Area has {original_count} hexagons. "
            f"Using {MAX_HEXAGONS} representative hexagons "
            f"(results scaled ×{scale_factor:.1f})")
else:
    hexagons_sampled = hexagons
    scale_factor = 1.0
    st.info(hex_description)



st.markdown("---")

if st.button("🚀 Generate Collective Report", use_container_width=True, type="primary"):

    pickup_dt = f"{pred_date}T{pred_hour:02d}:00:00"
    progress = st.progress(0, text="Preparing predictions...")

    progress.progress(10, text=f"Predicting {len(hexagons_sampled)} hexagons at {pred_hour:02d}:00...")

    current_payload = [
        {
            "h3_index": h3_idx,
            "pickup_datetime": pickup_dt,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "precipitation": precipitation,
        }
        for h3_idx in hexagons_sampled
    ]

    t0 = time.time()
    batch_results = safe_predict_batch(current_payload, timeout=15)
    phase1_time = time.time() - t0

    if not batch_results:
        st.error("Failed to get predictions. Check API logs.")
        st.stop()

    progress.progress(50, text="Processing results...")

    t1_values = []
    t2_values = []
    t3_values = []
    t4_values = []

    for pred in batch_results:
        preds = pred.get("predictions", {})
        if "T1" in preds:
            t1_values.append(preds["T1"]["value"])
        if "T2" in preds:
            t2_values.append(preds["T2"]["value"])
        if "T3" in preds:
            t3_values.append(preds["T3"]["value"])
        if "T4" in preds:
            t4_values.append(preds["T4"]["value"])

    # Scale T3 if we sampled
    area_total_t3 = sum(t3_values) * scale_factor
    area_avg_t3 = np.mean(t3_values) if t3_values else 0

    st.markdown(f"### 📊 Collective Report — {pred_date} at {pred_hour:02d}:00")
    st.caption(f"{len(hexagons_sampled)} hexagons analyzed in {phase1_time:.1f}s"
               + (f" (scaled from {original_count})" if scale_factor > 1 else ""))

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        citywide = int(t1_values[0]) if t1_values else 0
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2);
                    border-radius: 16px; padding: 24px; text-align: center;">
            <div style="color: rgba(255,255,255,0.8); font-size: 0.85rem;">🌆 Citywide Demand</div>
            <div style="color: white; font-size: 2.5rem; font-weight: 700;">{citywide:,}</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 0.75rem;">Total rides across Chicago</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #11998e, #38ef7d);
                    border-radius: 16px; padding: 24px; text-align: center;">
            <div style="color: rgba(255,255,255,0.8); font-size: 0.85rem;">📍 Area Demand</div>
            <div style="color: white; font-size: 2.5rem; font-weight: 700;">{int(area_total_t3):,}</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 0.75rem;">
                Avg {area_avg_t3:.1f}/hex × {original_count} hexagons
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        repo_count = sum(1 for v in t2_values if v == 1)
        repo_pct = (repo_count / len(t2_values) * 100) if t2_values else 0
        rec_text = "REPOSITION" if repo_pct > 50 else "STAY"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb, #f5576c);
                    border-radius: 16px; padding: 24px; text-align: center;">
            <div style="color: rgba(255,255,255,0.8); font-size: 0.85rem;">🚗 Driver Strategy</div>
            <div style="color: white; font-size: 2.2rem; font-weight: 700;">{rec_text}</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 0.75rem;">
                {repo_count}/{len(t2_values)} hexagons ({repo_pct:.0f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        surge_count = sum(1 for v in t4_values if v == 1)
        surge_pct = (surge_count / len(t4_values) * 100) if t4_values else 0
        surge_risk = "HIGH" if surge_pct > 30 else ("MODERATE" if surge_pct > 10 else "LOW")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe, #00f2fe);
                    border-radius: 16px; padding: 24px; text-align: center;">
            <div style="color: rgba(255,255,255,0.8); font-size: 0.85rem;">⚡ Surge Risk</div>
            <div style="color: white; font-size: 2.2rem; font-weight: 700;">{surge_risk}</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 0.75rem;">
                {surge_count}/{len(t4_values)} hexagons ({surge_pct:.0f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🔍 Per-Hexagon Breakdown")

    rows = []
    for pred in batch_results:
        preds = pred.get("predictions", {})
        rows.append({
            "H3 Index": pred.get("h3_index", ""),
            "Demand (T3)": preds.get("T3", {}).get("value", 0),
            "Reposition": "Yes" if preds.get("T2", {}).get("value", 0) == 1 else "No",
            "Surge": "⚡" if preds.get("T4", {}).get("value", 0) == 1 else "—",
        })

    hex_df = pd.DataFrame(rows).sort_values("Demand (T3)", ascending=False)

    # Bar chart
    fig_hex = go.Figure()
    fig_hex.add_trace(go.Bar(
        x=hex_df["H3 Index"].str[-7:],
        y=hex_df["Demand (T3)"],
        marker=dict(color=hex_df["Demand (T3)"], colorscale="Viridis", showscale=True),
        hovertext=hex_df["H3 Index"],
        hovertemplate="<b>%{hovertext}</b><br>Demand: %{y:.1f}<extra></extra>",
    ))
    fig_hex.update_layout(
        title="Demand per Hexagon",
        xaxis_title="Hexagon",
        yaxis_title="Predicted Rides",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
    )
    st.plotly_chart(fig_hex, use_container_width=True)
    st.dataframe(hex_df, use_container_width=True, hide_index=True)

    progress.progress(60, text="Generating 24-hour trend...")

    st.markdown("### 📈 24-Hour Demand Trend")
    st.caption("Using representative hexagon + citywide model for speed")

    # Pick the busiest hexagon as representative
    if hex_df is not None and len(hex_df) > 0:
        rep_hex = hex_df.iloc[0]["H3 Index"]
    else:
        rep_hex = hexagons_sampled[0]

    # 24 predictions for 1 hexagon = fast!
    trend_payload = [
        {
            "h3_index": rep_hex,
            "pickup_datetime": f"{pred_date}T{h:02d}:00:00",
            "temperature": temperature + (h - 12) * 0.5,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "precipitation": precipitation,
        }
        for h in range(24)
    ]

    progress.progress(70, text="Fetching 24-hour predictions...")
    trend_results = safe_predict_batch(trend_payload, timeout=15)

    if trend_results:
        progress.progress(90, text="Building charts...")

        hours = list(range(24))
        hour_labels = [f"{h:02d}:00" for h in hours]

        t3_hourly = []
        t1_hourly = []
        for pred in trend_results:
            preds = pred.get("predictions", {})
            # Scale the single hex demand by number of hexagons for area estimate
            t3_val = preds.get("T3", {}).get("value", 0) * original_count
            t1_val = preds.get("T1", {}).get("value", 0)
            t3_hourly.append(t3_val)
            t1_hourly.append(t1_val)

        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])

        fig_trend.add_trace(
            go.Scatter(
                x=hour_labels, y=t3_hourly,
                mode="lines+markers",
                name=f"Area Demand (est.)",
                line=dict(color="#38ef7d", width=3),
                fill="tozeroy",
                fillcolor="rgba(56, 239, 125, 0.1)",
            ),
            secondary_y=False,
        )

        fig_trend.add_trace(
            go.Scatter(
                x=hour_labels, y=t1_hourly,
                mode="lines",
                name="Citywide Demand",
                line=dict(color="#667eea", width=2, dash="dot"),
            ),
            secondary_y=True,
        )

        fig_trend.update_layout(
            title=f"24-Hour Forecast — {address or area_name or h3_manual}",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            legend=dict(x=0.01, y=0.99),
        )
        fig_trend.update_yaxes(title_text="Area Rides (est.)", secondary_y=False)
        fig_trend.update_yaxes(title_text="Citywide Rides", secondary_y=True)

        st.plotly_chart(fig_trend, use_container_width=True)

        sc1, sc2, sc3, sc4 = st.columns(4)
        peak_idx = np.argmax(t3_hourly)
        sc1.metric("Peak Hour", f"{peak_idx:02d}:00")
        sc2.metric("Peak Demand", f"{int(t3_hourly[peak_idx]):,}")
        sc3.metric("Daily Total", f"{int(sum(t3_hourly)):,}")
        sc4.metric("Avg/Hour", f"{np.mean(t3_hourly):,.1f}")

    progress.progress(100, text="Done!")
    time.sleep(0.5)
    progress.empty()

else:
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); border: 1px dashed rgba(255,255,255,0.2);
                border-radius: 12px; padding: 40px; text-align: center; margin: 20px 0;">
        <div style="font-size: 2rem;">🔮</div>
        <div style="color: rgba(255,255,255,0.6); margin-top: 10px;">
            Ready to analyze <b>{len(hexagons_sampled)} hexagons</b>
            {f'(sampled from {original_count})' if scale_factor > 1 else ''}<br>
            Click "Generate Collective Report"
        </div>
    </div>
    """, unsafe_allow_html=True)



