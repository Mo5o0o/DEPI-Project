# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# تحميل النموذج والسكيلر وخصائص التدريب
model = joblib.load("C:/Users/bhbt/Desktop/Sales-Forcasting-Project/3. models/xgb_model.pkl")
model_features = joblib.load("C:/Users/bhbt/Desktop/Sales-Forcasting-Project/3. models/model_features.pkl")
scaler = joblib.load("C:/Users/bhbt/Desktop/Sales-Forcasting-Project/3. models/scaler.pkl")

# إعداد الصفحة
st.set_page_config(
    page_title="Sales Forecaster",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحميل صورة شعار (اختياري)
# logo = Image.open("logo.png")

# CSS مخصص
st.markdown("""
    <style>
        .main {
            background-color: #F5F5F5;
        }
        .sidebar .sidebar-content {
            background-color: #2C3E50;
            color: white;
        }
        .stSelectbox, .stSlider, .stButton>button {
            border-radius: 8px;
        }
        .stButton>button {
            background-color: #3498DB;
            color: white;
            font-weight: bold;
            border: none;
            padding: 10px 24px;
        }
        .stButton>button:hover {
            background-color: #2980B9;
        }
        .prediction-box {
            background-color: #E8F4FC;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            border-left: 5px solid #3498DB;
        }
        .feature-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Header section
col1, col2 = st.columns([1, 4])
# with col1:
#     st.image(logo, width=80)
with col2:
    st.title("📊 Sales Forecaster Pro")
    st.markdown("Predict future sales with machine learning precision")

# Sidebar
with st.sidebar:
    st.header("⚙️ Input Parameters")
    st.markdown("Adjust the settings to match your scenario")
    
    with st.expander("🛳 Shipping Details", expanded=True):
        ship_mode = st.selectbox("Shipping Mode", ['Same Day', 'Second Class', 'Standard Class'])
        shipping_time = st.slider("Shipping Time (days)", 1, 10, 3, help="Estimated delivery duration")
    
    with st.expander("👥 Customer Info", expanded=True):
        segment = st.selectbox("Customer Segment", ['Corporate', 'Home Office', 'Consumer'])
    
    with st.expander("📦 Product Details", expanded=True):
        category_options = {
            'Furniture': ['Bookcases', 'Chairs', 'Furnishings', 'Tables'],
            'Office Supplies': ['Appliances', 'Binders', 'Envelopes', 'Fasteners', 'Labels', 'Paper', 'Storage', 'Supplies'],
            'Technology': ['Accessories', 'Copiers', 'Machines', 'Phones']
        }
        category = st.selectbox("Product Category", list(category_options.keys()))
        sub_category_list = category_options.get(category, [])
        sub_category = st.selectbox("Product Sub-Category", sub_category_list)
    
    with st.expander("📅 Order Timing", expanded=True):
        order_year = st.selectbox("Order Year", list(range(2018, 2031)), index=0, 
                                help="Model trained on data up to 2018. Later years are extrapolated")
        order_month = st.select_slider("Order Month", options=list(range(1,13)), value=6)
        order_day = st.select_slider("Order Day of Week", 
                                    options=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 
                                    value='Wed')

# تحويل اليوم من نص إلى رقم
day_mapping = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
order_day_num = day_mapping[order_day]

# Feature engineering
is_weekend = 1 if order_day_num in [5, 6] else 0
quarter = (order_month - 1) // 3 + 1
holiday_season = 1 if order_month in [11, 12] else 0

# بناء القيم المبدئية للـ DataFrame
input_dict = {
    'Shipping Time (days)': shipping_time,
    'Order Month': order_month,
    'Order Day Of Week': order_day_num,
    'Order Year': order_year,
    'Is_Weekend': is_weekend,
    'Quarter': quarter,
    'Holiday_Season': holiday_season,
    f'Ship Mode_{ship_mode}': 1,
    f'Segment_{segment}': 1,
    f'Category_{category}': 1,
    f'Sub-Category_{sub_category}': 1
}

# إنشاء الداتا فريم
df_input = pd.DataFrame([input_dict])

# إعادة ترتيب الأعمدة لتتناسب مع النموذج
df_input = df_input.reindex(columns=model_features, fill_value=0)

# Main content area
st.markdown("## 📋 Selected Parameters")
cols = st.columns(4)
with cols[0]:
    st.markdown("<div class='feature-card'>"
                f"<h4>🛳 Shipping</h4>"
                f"<p>Mode: <strong>{ship_mode}</strong></p>"
                f"<p>Time: <strong>{shipping_time} days</strong></p>"
                "</div>", unsafe_allow_html=True)
    
with cols[1]:
    st.markdown("<div class='feature-card'>"
                f"<h4>👥 Customer</h4>"
                f"<p>Segment: <strong>{segment}</strong></p>"
                "</div>", unsafe_allow_html=True)
    
with cols[2]:
    st.markdown("<div class='feature-card'>"
                f"<h4>📦 Product</h4>"
                f"<p>Category: <strong>{category}</strong></p>"
                f"<p>Sub-Category: <strong>{sub_category}</strong></p>"
                "</div>", unsafe_allow_html=True)
    
with cols[3]:
    st.markdown("<div class='feature-card'>"
                f"<h4>📅 Timing</h4>"
                f"<p>Date: <strong>{order_day}, Month {order_month}, {order_year}</strong></p>"
                f"<p>Quarter: <strong>Q{quarter}</strong></p>"
                "</div>", unsafe_allow_html=True)

# زر التنبؤ في وسط الصفحة
col_left, col_center, col_right = st.columns([1,2,1])
with col_center:
    predict_btn = st.button("🚀 Predict Sales", use_container_width=True)

# التنبؤ وعرض النتائج
if predict_btn:
    with st.spinner('🔮 Predicting sales...'):
        # تطبيق السكيلر
        df_input_scaled = scaler.transform(df_input)
        
        # التنبؤ
        prediction_log = model.predict(df_input_scaled)[0]
        prediction = np.expm1(prediction_log)  # عكس log1p
        
        # عرض النتيجة بطريقة جذابة
        st.markdown(f"""
            <div class='prediction-box'>
                <h2>📈 Prediction Result</h2>
                <p style='font-size: 24px; color: #3498DB; font-weight: bold;'>
                    Estimated Sales: <span style='color: #2ECC71;'>${prediction:,.2f}</span>
                </p>
                <p><small>Note: This is an estimate based on machine learning model</small></p>
            </div>
        """, unsafe_allow_html=True)
        
        # عرض تفسير بسيط (اختياري)
        with st.expander("ℹ️ About this prediction"):
            st.markdown("""
                - This prediction is based on historical sales patterns
                - The model considers factors like:
                    - Shipping method and time
                    - Product category and customer segment
                    - Seasonal trends and holidays
                - For years beyond 2018, the model extrapolates trends
            """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #7F8C8D;'><small>Sales Forecaster Pro v1.0 • Powered by XGBoost</small></div>", 
            unsafe_allow_html=True)