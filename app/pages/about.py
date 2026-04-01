"""Project Documentation & Technical Specifications (Professional Overhaul)."""

import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Styling
from app.utils.styling import get_custom_css

st.set_page_config(page_title="Project Documentation | ChiRide", layout="wide", page_icon=None)
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

st.title("Project Documentation & Technical Specifications")
st.markdown("Comprehensive overview of the system architecture, data lineage, and analytical methodology for the Chicago Ride Demand system.")

# Document Structure
tab_arch, tab_data, tab_model = st.tabs(["🏗️ System Architecture", "📊 Data Lineage & Engineering", "🤖 Model Selection Rationale"])

with tab_arch:
    st.subheader("High-Level Technical Architecture")
    st.markdown("""
    The Chicago Ride Demand Forecasting system is built on a modular, containerized architecture designed for high throughput and sub-millisecond inference.
    
    **KEY COMPONENTS**
    1. **Data Ingestion Layer**: Scalable ETL pipelines consuming raw TNC (Transportation Network Provider) feeds. Handles deduplication and UTC-06:00 normalization.
    2. **Spatial Processing Engine**: Leveraging Uber's H3 hierarchical indexing at Resolution 8 (~0.73km² cells) for optimized spatial binning.
    3. **Feature Registry**: Centralized repository of pre-computed temporal lags, rolling averages, and external weather embeddings.
    4. **Inference Gateway (FastAPI)**: Asynchronous REST API serving multiple model targets (T1-T4) simultaneously.
    5. **Executive Presentation Layer (Streamlit)**: High-fidelity reactive dashboard optimized for analytical deep-dives and executive reporting.
    """)
    
    st.info("System integration follows industry standards for ML-Ops, including automated drift detection and cross-algorithm benchmarking.")

with tab_data:
    st.subheader("Dataset Specifications & Engineering")
    st.markdown("""
    | Attribute | Specification |
    |:---|:---|
    | **Source Dataset** | Chicago Data Portal - TNC Trip Data (2022-2024) |
    | **Observation Volume** | 7.5M+ Individual Rides |
    | **Temporal Aggregation** | 60-Minute Non-Overlapping Windows |
    | **Spatial Resolution** | H3 Resolution 8 |
    | **Primary Features** | Trip Frequency, Average Fare, Lagged Demand (-1h, -24h, -1w) |
    | **Secondary Features** | Temperature (F), Precipitation (In), US Federal Holidays |
    """)
    
    st.markdown("---")
    st.subheader("Feature Engineering Methodology")
    st.markdown("""
    - **Cyclical Encoding**: Hour, Day, and Month features are transformed using Sine/Cosine components to preserve temporal continuity (e.g., matching 23:00 to 00:00).
    - **Lagging Strategy**: Features include $t-1$, $t-24$, and $t-168$ demand counts to capture short-term momentum and long-term weekly seasonality.
    - **Spatial Smoothing**: Cell-level demand is enriched with neighboring hexagon activity to reduce variance in low-density cells.
    """)

with tab_model:
    st.subheader("Analytical Model Selection Rationale")
    st.markdown("""
    The system utilizes a multi-model ensemble approach to serve different business objectives with the most suitable architecture.
    
    **1. Temporal Fusion Transformer (TFT)**
    - **Target**: T1 (Citywide Demand Regression)
    - **Rational**: State-of-the-art deep learning for time-series forecasting. Handles multi-horizon predictions and provides interpretable attention weights.
    
    **2. Gradient Boosting Machines (XGBoost / LightGBM)**
    - **Targets**: T2-T4 (Spatial Regression & Classification)
    - **Rational**: High-performance gradient boosting optimized for tabular data. Selected for its ability to handle sparse spatial features at the H3 cell level.
    
    **3. Ensemble Strategies**
    - The system compares the individual performance of these architectures to automatically select the optimal checkpoint during the weekly retraining pipeline.
    """)
    
    st.info("Continuous Model Evaluation: Precision-Recall curves and MAE metrics are recalculated every 24 hours to ensure system integrity.")
