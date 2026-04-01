"""System & Performance Integrity Monitoring (Professional Overhaul)."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.styling import get_custom_css, status_badge, metric_card
from app.utils.data_loader import load_citywide_data, load_h3_features
from src.monitoring.drift import detect_drift_report

st.set_page_config(page_title="System Monitoring | ChiRide", layout="wide", page_icon=None)
st.markdown(get_custom_css(), unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/taxi.png", width=80)
    st.markdown("### SYSTEM NAVIGATION")
    st.markdown("---")
    st.page_link("streamlit_app.py", label="Executive Summary", icon=None)
    st.page_link("pages/live_demand_map.py", label="Spatial Demand Analysis", icon=None)
    st.page_link("pages/model_performance.py", label="Model Benchmarking", icon=None)
    st.page_link("pages/predict.py", label="Demand Forecasting", icon=None)
    st.page_link("pages/data_explorer.py", label="In-Depth Data Explorer", icon=None)
    st.page_link("pages/monitoring.py", label="System Monitoring", icon=None)
    st.page_link("pages/about.py", label="Project Documentation", icon=None)
    st.markdown("---")
    st.header("Drift Notification Settings")
    psi_threshold = st.slider("PSI Alert Threshold", 0.05, 0.5, 0.2, step=0.05)
    st.caption("Lower thresholds result in higher sensitivity but may increase false positives.")

st.title("System Integrity & Performance Monitoring")
st.markdown("Automated monitoring of data distribution stability (Population Stability Index) and forecasting reliability metrics.")

# Data Analysis for Drift
df_all = load_h3_features()
if df_all.empty:
    st.error("System Integrity Analysis: Unable to locate feature data artifact. Monitoring inactive.")
    st.stop()

# Reference vs Observation Split
split_at = int(len(df_all) * 0.75)
df_ref = df_all.iloc[:split_at]
df_cur = df_all.iloc[split_at:]

# Primary Feature List
numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
banned = ["h3_index", "latitude", "longitude"]
target_feats = [c for c in numeric_cols if c not in banned]

# Report Generation
report = detect_drift_report(df_ref, df_cur, target_feats, threshold=psi_threshold)

# Executive KPI Grid
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(metric_card(status_badge("healthy" if report["status"] == "NO_DRIFT" else "degraded"), "System Health Score"), unsafe_allow_html=True)
with m2:
    st.markdown(metric_card(f"{report['percent_drifted']}%", "Features Diverged"), unsafe_allow_html=True)
with m3:
    avg_psi = np.mean(list(report["scores"].values())) if report["scores"] else 0
    st.markdown(metric_card(f"{avg_psi:.3f}", "Avg. PSI Coefficient"), unsafe_allow_html=True)
with m4:
    st.markdown(metric_card(str(report["n_features"]), "Active Monitors"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Performance Audit Breakdown
col_t, col_p = st.columns([1, 1])

with col_t:
    st.subheader("Population Stability Index (PSI) Report")
    score_df = pd.DataFrame([{"Feature": f, "Stability": s, "Verdict": "DIVERGED" if s > psi_threshold else "STABLE"} for f, s in report["scores"].items()])
    st.dataframe(score_df, use_container_width=True, hide_index=True)
    st.caption("Divergence thresholds follow standard PSI guidelines (< 0.1: Stable, > 0.2: Significant Shift).")

with col_p:
    st.subheader("Distribution Variance Audit")
    f_select = st.selectbox("Audit Individual Feature", target_feats)
    if f_select:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_ref[f_select], name="Base (Training Set)", marker_color="#1E40AF", opacity=0.7, histnorm='probability'))
        fig.add_trace(go.Histogram(x=df_cur[f_select], name="Current (Recent Set)", marker_color="#00DC82", opacity=0.7, histnorm='probability'))
        fig.update_layout(barmode='overlay', template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Comparing Reference (Base) vs Observation (Current) distributions for **{f_select}**.")

# Reliability Summary
st.markdown("---")
st.subheader("Technical Definitions for Management")
st.markdown("""
| Metric | Definition | Importance |
|:---|:---|:---|
| **Population Stability Index (PSI)** | Statistical measure of how much a feature's distribution has shifted over time. | Critical for detecting changing user behaviors or data collection malfunctions. |
| **Drift Verdict** | Automated determination of whether the input data matches the model's training assumptions. | Justifies model retraining or investigation when 'DIVERGED'. |
| **System Health Score** | Aggregate indicator based on drift and API availability. | High-level status for non-technical stakeholders. |
""")


