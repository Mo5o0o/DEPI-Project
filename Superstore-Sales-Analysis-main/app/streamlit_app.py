import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit.components.v1 import html
import json
from calendar import monthrange

# Custom CSS and JavaScript with enhanced effects
def inject_custom_style():
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.8)), 
                        ('https://th.bing.com/th/id/OIP.hxjqE438G87V4J32XMF-BwHaE_?w=1200&h=800&rs=1&pid=ImgDetMain');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        
        /* Enhanced card with glow effect */
        .card {
            transition: all 0.4s ease;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid rgba(0, 0, 0, 0.1);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .card:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2),
                        0 0 15px rgba(76, 175, 80, 0.3);
        }
        
        /* Modern button with pulse effect */
        .stButton>button {
            border: none;
            border-radius: 25px;
            padding: 12px 28px;
            transition: all 0.4s;
            background: linear-gradient(135deg, #4CAF50, #2E7D32);
            color: white !important;
            font-weight: 600;
            box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 15px rgba(76, 175, 80, 0.4);
        }
        
        .stButton>button:active {
            transform: scale(0.98);
        }
        
        /* Ripple effect */
        .stButton>button:after {
            content: "";
            display: block;
            position: absolute;
            left: 50%;
            top: 50%;
            width: 120px;
            height: 120px;
            margin-left: -60px;
            margin-top: -60px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 100%;
            opacity: 0;
            transform: scale(0);
        }
        
        .stButton>button:active:after {
            animation: ripple 1s ease-out;
        }
        
        @keyframes ripple {
            0% {
                transform: scale(0);
                opacity: 1;
            }
            100% {
                transform: scale(2);
                opacity: 0;
            }
        }
        
        /* Slider styling */
        .stSlider .thumb {
            width: 20px !important;
            height: 20px !important;
            box-shadow: 0 0 5px rgba(0,0,0,0.2) !important;
        }
        
        /* Date input styling */
        .stDateInput>div>div>input {
            border-radius: 15px !important;
            padding: 12px !important;
            border: 1px solid #ddd !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced mouse particles with trail effect
    html("""
    <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
    <div id="particles-js" style="position:fixed;top:0;left:0;z-index:-1;"></div>
    <script>
        particlesJS('particles-js', {
            particles: {
                number: { 
                    value: 80,
                    density: { enable: true, value_area: 800 }
                },
                color: { value: "#4CAF50" },
                shape: { type: "circle" },
                opacity: { 
                    value: 0.7,
                    random: true,
                    anim: { enable: true, speed: 1, opacity_min: 0.1 }
                },
                size: {
                    value: 3,
                    random: true,
                    anim: { enable: true, speed: 2, size_min: 0.1 }
                },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: "#4CAF50",
                    opacity: 0.4,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 3,
                    direction: "none",
                    random: true,
                    straight: false,
                    out_mode: "out",
                    bounce: false,
                    attract: { enable: true, rotateX: 600, rotateY: 1200 }
                }
            },
            interactivity: {
                detect_on: "canvas",
                events: {
                    onhover: { 
                        enable: true, 
                        mode: "grab",
                        parallax: { enable: true, force: 60, smooth: 10 }
                    },
                    onclick: { 
                        enable: true, 
                        mode: "push",
                        particles_nb: 4
                    },
                    resize: true
                },
                modes: {
                    grab: { distance: 140, line_linked: { opacity: 1 } },
                    bubble: { distance: 400, size: 40, duration: 2, opacity: 8 },
                    push: { particles_nb: 4 },
                    remove: { particles_nb: 2 }
                }
            },
            retina_detect: true
        });
        
        // Mouse trail effect
        document.addEventListener('mousemove', function(e) {
            const trail = document.createElement('div');
            trail.className = 'mouse-trail';
            trail.style.left = e.pageX + 'px';
            trail.style.top = e.pageY + 'px';
            document.body.appendChild(trail);
            
            setTimeout(() => {
                trail.style.opacity = '0';
                trail.style.transform = 'scale(2)';
                setTimeout(() => trail.remove(), 300);
            }, 50);
        });
        
        const style = document.createElement('style');
        style.textContent = `
            .mouse-trail {
                position: absolute;
                width: 10px;
                height: 10px;
                background: rgba(76, 175, 80, 0.5);
                border-radius: 50%;
                pointer-events: none;
                transform: translate(-50%, -50%);
                z-index: 9999;
                transition: all 0.3s ease-out;
            }
        `;
        document.head.appendChild(style);
    </script>
    """)

@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\bhbt\Desktop\Superstore-Sales-Analysis-main\model\xgboost_model.pkl")

model = load_model()

inject_custom_style()
st.title("📈 Intelligent Sales Forecasting System")
st.markdown("Explore future sales predictions with our AI-powered forecasting engine")

col1, col2 = st.columns([2, 1])

with col1:
    with st.expander("🔮 Prediction Controls", expanded=True):
        col_year, col_month, col_day = st.columns(3)
        
        with col_year:
            selected_year = st.slider(
                "Year",
                min_value=2020,
                max_value=2030,
                value=datetime.now().year,
                key="year_slider"
            )
        
        with col_month:
            selected_month = st.slider(
                "Month",
                min_value=1,
                max_value=12,
                value=datetime.now().month,
                key="month_slider"
            )
        
        with col_day:
            _, last_day = monthrange(selected_year, selected_month)
            selected_day = st.slider(
                "Day",
                min_value=1,
                max_value=last_day,
                value=datetime.now().day,
                key="day_slider"
            )
        
        date_input = datetime(selected_year, selected_month, selected_day).date()
        
        st.markdown("""<div class='card'>""", unsafe_allow_html=True)
        if st.button("Generate Prediction", key="main_btn"):
            st.session_state['predict'] = True
        st.markdown("""</div>""", unsafe_allow_html=True)

def get_date_features(date):
    return {
        'Year': date.year,
        'Month': date.month,
        'Quarter': (date.month - 1) // 3 + 1,
        'Day': date.day,
        'DayOfWeek': date.weekday(),
        'DayOfYear': date.timetuple().tm_yday,
        'WeekOfYear': date.isocalendar().week
    }

if 'predict' in st.session_state:
    date_features = get_date_features(date_input)
    input_df = pd.DataFrame([date_features])
    
    try:
        prediction = model.predict(input_df)[0]
        
        st.markdown(f"""
        <div class='card' style='animation: fadeIn 1s;'>
            <h3 style='color: #4CAF50;'>📅 {date_input.strftime('%Y-%m-%d')}</h3>
            <h2 style='color: #2196F3;'>Predicted Sales: ${prediction:,.2f}</h2>
            <p>🗓️ {date_input.strftime('%A')} | 📅 Q{date_features['Quarter']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Generate full year data for visualizations
        months = pd.date_range(start=f"{selected_year}-01-01", end=f"{selected_year}-12-31")
        data = [get_date_features(d) for d in months]
        df = pd.DataFrame(data)
        df['Prediction'] = model.predict(df)

        # Time Series Animation
        st.markdown("### 🎥 Sales Evolution")
        fig = px.line(df, x=months, y='Prediction', 
                     template='plotly_dark',
                     labels={'y': 'Predicted Sales'},
                     hover_data={'date': months.strftime("%Y-%m-%d")})
        fig.update_traces(line=dict(width=3, color='#4CAF50'))
        fig.add_vline(x=date_input, line_dash="dot", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        # Seasonal Pattern Radar Chart
        st.markdown("### 🌸 Seasonal Patterns")
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        avg_sales = [df[df['Quarter'] == i]['Prediction'].mean() for i in range(1,5)]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=avg_sales,
            theta=quarters,
            fill='toself',
            line_color='#4CAF50'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(avg_sales)*1.1]
                )
            ),
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")

with col2:
    st.markdown("### 🏆 Feature Impact")
    try:
        features = ['Year', 'Month', 'Quarter', 'Day', 'DayOfWeek', 'DayOfYear', 'WeekOfYear']
        importances = model.feature_importances_
        
        fig = px.bar(x=importances, y=features, orientation='h',
                    color=importances, color_continuous_scale='Bluered')
        fig.update_layout(showlegend=False, 
                         xaxis_title='Importance Score',
                         yaxis_title='Features',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Feature importance not available for this model")

html("""
<script>
document.querySelectorAll('.stButton button').forEach(button => {
    button.addEventListener('click', function() {
        // Ripple effect
        const ripple = document.createElement('span');
        ripple.className = 'ripple';
        this.appendChild(ripple);
        
        const x = event.clientX - event.target.getBoundingClientRect().left;
        const y = event.clientY - event.target.getBoundingClientRect().top;
        
        ripple.style.left = `${x}px`;
        ripple.style.top = `${y}px`;
        
        setTimeout(() => ripple.remove(), 1000);
        
        // Scroll to results
        window.scrollTo({
            top: document.documentElement.scrollHeight,
            behavior: 'smooth'
        });
    });
});

document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('mouseenter', () => {
        card.style.transform = 'translateY(-5px) scale(1.02)';
    });
    card.addEventListener('mouseleave', () => {
        card.style.transform = 'none';
    });
});
</script>
<style>
.ripple {
    position: absolute;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.7);
    transform: scale(0);
    animation: ripple 0.6s linear;
    pointer-events: none;
}

@keyframes ripple {
    to {
        transform: scale(4);
        opacity: 0;
    }
}
</style>
""")