import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# تحميل الموديل
MODEL_PATH = "C:/Users/bhbt/Desktop/Sales-Forcasting-Project/3. models/best_model.pkl"
model = joblib.load(MODEL_PATH)

# جميع الفيتشرز اللي اتدرب عليها الموديل
features = [
    'Ship Mode_Same Day', 'Ship Mode_Second Class', 'Ship Mode_Standard Class',
    'Segment_Corporate', 'Segment_Home Office',
    'City_Arlington', 'City_Aurora', 'City_Charlotte', 'City_Chicago',
    'City_Columbia', 'City_Columbus', 'City_Dallas', 'City_Detroit',
    'City_Henderson', 'City_Houston', 'City_Jackson', 'City_Jacksonville',
    'City_Long Beach', 'City_Los Angeles', 'City_Louisville', 'City_Miami',
    'City_New York City', 'City_Newark', 'City_Philadelphia',
    'City_Phoenix', 'City_Richmond', 'City_Rochester', 'City_San Antonio',
    'City_San Diego', 'City_San Francisco', 'City_Seattle',
    'City_Springfield', 'State_Arizona', 'State_Arkansas',
    'State_California', 'State_Colorado', 'State_Connecticut',
    'State_Delaware', 'State_Florida', 'State_Georgia', 'State_Illinois',
    'State_Indiana', 'State_Kentucky', 'State_Maryland', 'State_Massachusetts',
    'State_Michigan', 'State_Minnesota', 'State_Mississippi', 'State_Missouri',
    'State_New Jersey', 'State_New York', 'State_North Carolina', 'State_Ohio',
    'State_Oklahoma', 'State_Oregon', 'State_Pennsylvania', 'State_Rhode Island',
    'State_Tennessee', 'State_Texas', 'State_Utah', 'State_Virginia',
    'State_Washington', 'State_Wisconsin', 'Region_East', 'Region_South',
    'Region_West', 'Category_Office Supplies', 'Category_Technology',
    'Sub-Category_Appliances', 'Sub-Category_Art', 'Sub-Category_Binders',
    'Sub-Category_Bookcases', 'Sub-Category_Chairs', 'Sub-Category_Copiers',
    'Sub-Category_Envelopes', 'Sub-Category_Fasteners', 'Sub-Category_Furnishings',
    'Sub-Category_Labels', 'Sub-Category_Machines', 'Sub-Category_Paper',
    'Sub-Category_Phones', 'Sub-Category_Storage', 'Sub-Category_Supplies',
    'Sub-Category_Tables'
]

# واجهة التطبيق
st.title("🔮 Sales Prediction App")
st.write("أدخل بيانات الطلب، وسيتم التنبؤ بالمبيعات 🎯")

# إدخالات المستخدم (بسيطة - نفعّل فقط العناصر المطلوبة)
ship_mode = st.selectbox("Ship Mode", ['Same Day', 'Second Class', 'Standard Class'])
segment = st.selectbox("Segment", ['Consumer', 'Corporate', 'Home Office'])
city = st.selectbox("City", [f.replace('City_', '') for f in features if f.startswith('City_')])
state = st.selectbox("State", [f.replace('State_', '') for f in features if f.startswith('State_')])
region = st.selectbox("Region", ['East', 'South', 'West'])
category = st.selectbox("Category", ['Furniture', 'Office Supplies', 'Technology'])
sub_category = st.selectbox("Sub-Category", [f.replace('Sub-Category_', '') for f in features if f.startswith('Sub-Category_')])

# بناء صف الداتا من إدخالات المستخدم (one-hot)
input_data = pd.DataFrame(columns=features)
input_row = pd.Series(0, index=features)

# ضبط القيم
if f"Ship Mode_{ship_mode}" in input_row:
    input_row[f"Ship Mode_{ship_mode}"] = 1

if f"Segment_{segment}" in input_row:
    input_row[f"Segment_{segment}"] = 1

if f"City_{city}" in input_row:
    input_row[f"City_{city}"] = 1

if f"State_{state}" in input_row:
    input_row[f"State_{state}"] = 1

if f"Region_{region}" in input_row:
    input_row[f"Region_{region}"] = 1

if f"Category_{category}" in input_row:
    input_row[f"Category_{category}"] = 1

if f"Sub-Category_{sub_category}" in input_row:
    input_row[f"Sub-Category_{sub_category}"] = 1

input_data.loc[0] = input_row

# توقع
if st.button("🔎 توقع المبيعات"):
    prediction = model.predict(input_data)[0]
    st.success(f"✅ التنبؤ بالمبيعات: ${prediction:,.2f}")
