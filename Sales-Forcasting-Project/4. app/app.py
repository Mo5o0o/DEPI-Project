import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import time
from streamlit.components.v1 import html
import plotly.graph_objects as go

# Page config must be first
st.set_page_config(
    page_title="✨ Dynamic Sales Forecaster",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations and vibrant colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    :root {
        --primary: #6c5ce7;
        --secondary: #a29bfe;
        --accent: #fd79a8;
        --dark: #2d3436;
        --light: #f5f6fa;
        --success: #00b894;
        --warning: #fdcb6e;
        --danger: #d63031;
    }
    
    * {
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stApp {
        background: linear-gradient(to bottom right, #f5f7fa, #e4e8f0);
    }
    
    .stButton>button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 28px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.3);
        transition: all 0.4s ease;
        transform: translateY(0);
    }
    
    .stButton>button:hover {
        background: var(--primary);
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(108, 92, 231, 0.4);
    }
    
    .stSelectbox, .stSlider, .stNumberInput {
        border-radius: 12px !important;
    }
    
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        margin: 15px 0;
        border-left: 5px solid var(--primary);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.12);
    }
    
    .hover-zoom {
        transition: transform 0.3s ease;
    }
    
    .hover-zoom:hover {
        transform: scale(1.02);
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(108, 92, 231, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(108, 92, 231, 0); }
        100% { box-shadow: 0 0 0 0 rgba(108, 92, 231, 0); }
    }
    
    .glow {
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 5px rgba(253, 121, 168, 0.5); }
        to { box-shadow: 0 0 20px rgba(253, 121, 168, 0.8); }
    }
    
    .floating {
        animation: floating 6s ease-in-out infinite;
    }
    
    @keyframes floating {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .stTabs [role="tablist"] button {
        border-radius: 12px !important;
        padding: 8px 16px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Mouse follower animation
html("""
<script>
    document.addEventListener('DOMContentLoaded', function() {
        const follower = document.createElement('div');
        follower.style.width = '20px';
        follower.style.height = '20px';
        follower.style.borderRadius = '50%';
        follower.style.background = 'radial-gradient(circle, #fd79a8, #6c5ce7)';
        follower.style.position = 'fixed';
        follower.style.pointerEvents = 'none';
        follower.style.zIndex = '9999';
        follower.style.transform = 'translate(-50%, -50%)';
        follower.style.opacity = '0.7';
        follower.style.filter = 'blur(5px)';
        document.body.appendChild(follower);
        
        let mouseX = 0, mouseY = 0;
        let posX = 0, posY = 0;
        
        document.addEventListener('mousemove', function(e) {
            mouseX = e.clientX;
            mouseY = e.clientY;
        });
        
        function animate() {
            posX += (mouseX - posX) / 10;
            posY += (mouseY - posY) / 10;
            
            follower.style.left = posX + 'px';
            follower.style.top = posY + 'px';
            
            requestAnimationFrame(animate);
        }
        
        animate();
        
        // Add hover effects to interactive elements
        const interactiveElements = document.querySelectorAll('.stButton>button, .stSelectbox, .stSlider, .stNumberInput, .metric-card');
        interactiveElements.forEach(el => {
            el.addEventListener('mouseenter', () => {
                follower.style.transform = 'translate(-50%, -50%) scale(2)';
                follower.style.background = 'radial-gradient(circle, #00b894, #0984e3)';
            });
            el.addEventListener('mouseleave', () => {
                follower.style.transform = 'translate(-50%, -50%) scale(1)';
                follower.style.background = 'radial-gradient(circle, #fd79a8, #6c5ce7)';
            });
        });
    });
</script>
""")

# Header with floating animation
st.markdown("""
<div style="text-align: center; padding: 40px 0 20px;">
    <h1 style="color: #2d3436; font-size: 2.8rem; margin-bottom: 10px; 
        background: linear-gradient(45deg, #6c5ce7, #fd79a8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;" class="floating">
        ✨ Dynamic Sales Forecaster
    </h1>
    <p style="color: #636e72; font-size: 1.2rem; max-width: 700px; margin: 0 auto;">
        Predictive analytics with interactive visualizations and real-time insights
    </p>
</div>
""", unsafe_allow_html=True)

# Load sample data (replace with your actual data loading)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("C:/Users/bhbt/Desktop/Sales-Forcasting-Project/1. data/processed/Sales__cleaned.csv")
        df['Order Date'] = pd.to_datetime(df['Order Year'].astype(str) + '-' + df['Order Month'].astype(str) + '-01')  # ✅ القوس اتقفل هنا
        monthly_sales = df.groupby('Order Date')['Sales_log'].sum().reset_index()
        prophet_df = monthly_sales.rename(columns={'Order Date': 'ds', 'Sales_log': 'y'})
        return prophet_df
    except Exception as e:
        st.error(f"⚠️ Error loading data! {e}")
        st.stop()


prophet_df = load_data()

# Animated sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #2d3436; margin-bottom: 5px;">⚙️ Controls</h2>
        <div style="height: 3px; background: linear-gradient(90deg, #6c5ce7, #fd79a8); 
            border-radius: 3px; margin: 0 auto 20px; width: 50%;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📅 Time Settings", expanded=True):
        forecast_months = st.slider(
            "Forecast Horizon (months)",
            min_value=6, max_value=36, value=24, step=6,
            help="How far into the future to predict"
        )
        
    with st.expander("💰 Revenue Model", expanded=True):
        calc_mode = st.checkbox("Enable Revenue Projection", True)
        if calc_mode:
            price_per_unit = st.number_input(
                "Unit Price ($)",
                min_value=0.0, value=100.0, step=10.0
            )
            start_month = st.selectbox(
                "Start Month",
                pd.date_range(start=pd.Timestamp.now(), periods=12, freq='MS').strftime('%Y-%m').tolist()
            )

# Model training with progress animation
@st.cache_resource
def train_model(data):
    with st.spinner('Training model...'):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)  # Simulate training time
            progress_bar.progress(percent_complete + 1)
        model = Prophet()
        model.fit(data)
        progress_bar.empty()
        st.success('Model trained successfully!', icon="✅")
        return model

model = train_model(prophet_df)

# Forecasting
future = model.make_future_dataframe(periods=forecast_months, freq='MS')
forecast = model.predict(future)

# Main content area
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 📊 Interactive Forecast")
    
    # Create interactive Plotly chart
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=prophet_df['ds'], y=prophet_df['y'],
        mode='lines+markers',
        name='Actual Sales',
        line=dict(color='#6c5ce7', width=3),
        marker=dict(size=8, color='#a29bfe')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#00b894', width=3)
    ))
    
    # Add uncertainty interval
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        name='Uncertainty',
        line=dict(width=0),
        fillcolor='rgba(0, 184, 148, 0.2)'
    ))
    
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Sales (log scale)'),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🔍 Key Metrics")
    
    # Calculate metrics with animation
    latest_actual = prophet_df['y'].iloc[-1]
    latest_forecast = forecast['yhat'].iloc[-1]
    change_pct = ((latest_forecast - latest_actual) / latest_actual) * 100
    
    st.markdown(f"""
    <div class="metric-card pulse">
        <h3 style="color: #636e72; margin-bottom: 15px;">Current Period</h3>
        <div style="display: flex; flex-direction: column; gap: 15px;">
            <div>
                <p style="color: #2d3436; margin: 0; font-size: 0.9rem;">Actual</p>
                <h2 style="color: #6c5ce7; margin: 5px 0;">{latest_actual:,.2f}</h2>
            </div>
            <div>
                <p style="color: #2d3436; margin: 0; font-size: 0.9rem;">Forecast</p>
                <h2 style="color: #00b894; margin: 5px 0;">{latest_forecast:,.2f}</h2>
            </div>
            <div>
                <p style="color: #2d3436; margin: 0; font-size: 0.9rem;">Change</p>
                <h2 style="color: {'#d63031' if change_pct < 0 else '#00b894'}; margin: 5px 0;">
                    {change_pct:.1f}% {'▼' if change_pct < 0 else '▲'}
                </h2>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Revenue projection with glow effect
if calc_mode:
    st.markdown("---")
    st.markdown("## 💰 Revenue Projection")
    
    selected_range = forecast[forecast['ds'] >= pd.to_datetime(start_month)]
    total_units = selected_range['yhat'].sum()
    total_revenue = total_units * price_per_unit
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"""
        <div class="metric-card hover-zoom">
            <h3 style="color: #636e72; margin-bottom: 15px;">Units Forecast</h3>
            <h1 style="color: #6c5ce7; margin: 0; font-size: 2.5rem; text-align: center;">
                {total_units:,.0f}
            </h1>
            <p style="color: #636e72; text-align: center; margin-top: 5px;">units</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card glow">
            <h3 style="color: #636e72; margin-bottom: 15px;">Projected Revenue</h3>
            <h1 style="color: #00b894; margin: 0; font-size: 2.5rem; text-align: center;">
                ${total_revenue:,.2f}
            </h1>
            <p style="color: #636e72; text-align: center; margin-top: 5px;">at ${price_per_unit:,.2f}/unit</p>
        </div>
        """, unsafe_allow_html=True)

# Components analysis with tabs
st.markdown("---")
st.markdown("## 🔍 Forecast Components")

tab1, tab2 = st.tabs(["Trend Analysis", "Seasonality"])
with tab1:
    fig_trend = model.plot_components(forecast)
    st.pyplot(fig_trend)
with tab2:
    st.markdown("""
    <div style="background: white; border-radius: 16px; padding: 20px; box-shadow: 0 6px 18px rgba(0,0,0,0.08);">
        <h3 style="color: #2d3436; margin-top: 0;">Weekly & Yearly Patterns</h3>
        <p style="color: #636e72;">The charts below show how sales vary by day of week and month of year.</p>
    </div>
    """, unsafe_allow_html=True)
    fig_season = model.plot_components(forecast)
    st.pyplot(fig_season)

# Footer with current time
html("""
<div style="text-align: center; padding: 20px; margin-top: 40px; color: #636e72; font-size: 0.9rem;">
    <p>✨ Dynamic Sales Forecaster • Updated: <span id="live-time"></span></p>
</div>
<script>
    function updateTime() {
        const now = new Date();
        document.getElementById('live-time').textContent = now.toLocaleString();
    }
    updateTime();
    setInterval(updateTime, 1000);
</script>
""")