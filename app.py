# -*- coding: utf-8 -*-
"""
🏠 California Housing Price Predictor
แอป Streamlit สำหรับทำนายราคาบ้านจากโมเดล Regression ที่เทรนไว้บน Google Colab

วิธีรัน:
    streamlit run app.py

โครงสร้างไฟล์ที่ต้องมี:
    app.py
    model/regression_model.pkl
    model/scaler.pkl
    model/feature_names.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# CSS แบบเรียบง่าย (minimal, clean)
# ------------------------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .main {
        background-color: #FAFAFA;
    }
    .block-container {
        padding-top: 2.5rem;
        max-width: 760px;
    }
    .title-box {
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .title-box h1 {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 0.2rem;
    }
    .title-box p {
        color: #6B7280;
        font-size: 1rem;
        margin-top: 0;
    }
    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        padding: 1.8rem 1.8rem 0.8rem 1.8rem;
        border-radius: 16px;
        border: 1px solid #ECECEC;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    .stButton > button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 0;
        font-weight: 600;
        font-size: 1rem;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    .result-card {
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        color: white;
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 1.4rem;
    }
    .result-card h2 {
        font-size: 2.3rem;
        margin: 0.2rem 0;
    }
    .result-card p {
        opacity: 0.85;
        margin: 0;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# โหลดโมเดล / scaler / feature names (แคชไว้ ไม่ต้องโหลดซ้ำทุกครั้ง)
# ------------------------------------------------------------------
MODEL_DIR = "model"

@st.cache_resource
def load_artifacts():
    model_path = os.path.join(MODEL_DIR, "regression_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_names.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ------------------------------------------------------------------
# หัวข้อ
# ------------------------------------------------------------------
st.markdown("""
<div class="title-box">
    <h1>🏠 House Price Predictor</h1>
    <p>ทำนายราคาบ้านในรัฐแคลิฟอร์เนีย ด้วยโมเดล Machine Learning</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(
        "❌ ไม่พบไฟล์โมเดล กรุณาวางโฟลเดอร์ `model/` "
        "(ที่มี regression_model.pkl, scaler.pkl, feature_names.pkl) "
        "ไว้ในโฟลเดอร์เดียวกับ app.py"
    )
    st.stop()

# ------------------------------------------------------------------
# คำอธิบายสั้น ๆ ของแต่ละฟีเจอร์ (ให้ผู้ใช้เข้าใจง่ายกว่าชื่อ column ดิบ)
# ------------------------------------------------------------------
FIELD_CONFIG = {
    "MedInc":     {"label": "รายได้เฉลี่ยในพื้นที่ (หมื่นดอลลาร์)", "min": 0.5, "max": 15.0, "default": 3.9,  "step": 0.1},
    "HouseAge":   {"label": "อายุบ้านโดยเฉลี่ย (ปี)",              "min": 1.0, "max": 52.0, "default": 28.0, "step": 1.0},
    "AveRooms":   {"label": "จำนวนห้องเฉลี่ยต่อครัวเรือน",         "min": 1.0, "max": 20.0, "default": 5.4,  "step": 0.1},
    "AveBedrms":  {"label": "จำนวนห้องนอนเฉลี่ยต่อครัวเรือน",       "min": 0.5, "max": 6.0,  "default": 1.1,  "step": 0.1},
    "Population": {"label": "จำนวนประชากรในพื้นที่",               "min": 3.0, "max": 35000.0, "default": 1425.0, "step": 10.0},
    "AveOccup":   {"label": "จำนวนผู้อยู่อาศัยเฉลี่ยต่อครัวเรือน",   "min": 0.5, "max": 15.0, "default": 3.0,  "step": 0.1},
    "Latitude":   {"label": "ละติจูด (Latitude)",                  "min": 32.0, "max": 42.0, "default": 34.2, "step": 0.01},
    "Longitude":  {"label": "ลองจิจูด (Longitude)",                "min": -125.0, "max": -114.0, "default": -118.4, "step": 0.01},
}

# ------------------------------------------------------------------
# ฟอร์มกรอกข้อมูล
# ------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("📋 กรอกข้อมูลพื้นที่ / บ้าน")

    col1, col2 = st.columns(2)
    user_input = {}

    for i, feat in enumerate(feature_names):
        cfg = FIELD_CONFIG.get(feat, {"label": feat, "min": 0.0, "max": 100.0, "default": 1.0, "step": 0.1})
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            user_input[feat] = st.number_input(
                cfg["label"],
                min_value=float(cfg["min"]),
                max_value=float(cfg["max"]),
                value=float(cfg["default"]),
                step=float(cfg["step"]),
                key=feat,
            )

    submitted = st.form_submit_button("🔮 ทำนายราคาบ้าน")

# ------------------------------------------------------------------
# ทำนายผล
# ------------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([user_input])[feature_names]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    price_usd = prediction * 100_000  # แปลงจาก "แสนดอลลาร์" เป็นดอลลาร์เต็มจำนวน

    st.markdown(f"""
    <div class="result-card">
        <p>ราคาบ้านที่คาดการณ์</p>
        <h2>${price_usd:,.0f}</h2>
        <p>({prediction:.2f} แสนดอลลาร์สหรัฐ)</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ดูรายละเอียดข้อมูลที่กรอก"):
        st.dataframe(input_df.T.rename(columns={0: "ค่าที่กรอก"}), use_container_width=True)

st.markdown(
    "<p style='text-align:center; color:#9CA3AF; font-size:0.85rem; margin-top:2rem;'>"
    "สร้างด้วย Streamlit • โมเดลเทรนบน Google Colab</p>",
    unsafe_allow_html=True
)