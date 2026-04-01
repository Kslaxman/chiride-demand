"""Data Exploration & Statistical Profiling (Professional Overhaul)."""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Styling and loading
from app.utils.styling import get_custom_css
from app.utils.data_loader import load_citywide_data, load_h3_data

st.set_page_config(page_title="Data Explorer | ChiRide", layout="wide", page_icon=None)
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

st.title("Analytical Data Explorer")
st.markdown("Direct inspection of Chicago's processed ride-hailing datasets.")

st.info("""
**DATA SOURCES & METHODOLOGY REFERENCES:**
- **Raw Ride-hailing Data:** [Chicago Data Portal - Transportation Network Providers (TNC) Trips](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p)
- **Meteorological Features:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- **Spatial Resolution & Indexing:** [Uber H3 Hexagonal Hierarchical Spatial Index](https://h3geo.org/)
""")

st.markdown("---")

df_city = load_citywide_data()
if not df_city.empty:
    st.subheader("Citywide Aggregates Sample")
    st.markdown("First 5 rows of `chirde.citywide_hourly.parquet`")
    st.dataframe(df_city.head(), use_container_width=True, hide_index=True)
else:
    st.warning("Unable to locate citywide data artifact.")

st.markdown("<br>", unsafe_allow_html=True)

df_h3 = load_h3_data()
if not df_h3.empty:
    st.subheader("H3 Hexagonal Distribution Sample")
    st.markdown("First 5 rows of `chirde.h3_hourly_comp.parquet`")
    st.dataframe(df_h3.head(), use_container_width=True, hide_index=True)
else:
    st.warning("Unable to locate H3 hexagonal data artifact.")
