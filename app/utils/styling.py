import streamlit as st

def get_custom_css():
    """Returns a highly professional, dark-themed CSS for the Chicago Ride Demand system."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main Container Style */
    .main {
        background-color: #0c1117;
    }

    /* Sidebar Professional Navigation */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }

    /* Hero Header Layout */
    .hero-container {
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        margin-bottom: 2.5rem;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.05rem;
        margin-bottom: 1rem;
    }

    .hero-title span {
        color: #00DC82; /* Signature Nuxt/Enterprise Green */
    }

    .hero-subtitle {
        font-size: 1.25rem;
        color: #8b949e;
        max-width: 800px;
        line-height: 1.75;
    }

    /* Professional Metric Cards */
    .metric-card {
        background-color: #161b22;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #30363d;
        transition: all 0.2s ease-in-out;
    }

    .metric-card:hover {
        border-color: #00DC82;
        background-color: #1c2128;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }

    /* Glass Effect Containers */
    .glass-card {
        background: rgba(22, 27, 34, 0.8);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #30363d;
    }

    /* Professional Status Badges */
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        display: inline-block;
    }

    .status-healthy { background: rgba(0, 220, 130, 0.1); color: #00DC82; border: 1px solid rgba(0, 220, 130, 0.2); }
    .status-degraded { background: rgba(255, 172, 51, 0.1); color: #ffac33; border: 1px solid rgba(255, 172, 51, 0.2); }
    .status-critical { background: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.2); }

    /* Footer Analysis Context */
    .footer {
        padding-top: 4rem;
        padding-bottom: 2rem;
        text-align: center;
        color: #484f58;
        font-size: 0.875rem;
        border-top: 1px solid #30363d;
        margin-top: 4rem;
    }

    /* Fix Streamlit Header Overlay */
    header { visibility: visible !important; }
    </style>
    """

def metric_card(value, label, delta=""):
    """Returns an HTML snippet for a professional, high-density metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def glass_card(content):
    """Wraps content in a formal glass-morphism container."""
    return f"""
    <div class="glass-card">
        {content}
    </div>
    """

def status_badge(status):
    """Returns a formal professional status pill."""
    status = status.lower()
    if status in ["healthy", "online", "stable"]:
        css_class = "status-healthy"
    elif status in ["degraded", "drifted", "warning"]:
        css_class = "status-degraded"
    else:
        css_class = "status-critical"
    
    return f'<span class="status-badge {css_class}">{status.upper()}</span>'