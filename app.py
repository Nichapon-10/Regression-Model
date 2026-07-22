# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Streamlit Web App — Regression Prediction
โหลดโมเดล Regression ที่เทรนไว้จาก Google Colab (regression_model.pkl)
แล้วให้ผู้ใช้กรอกข้อมูล หรืออัปโหลด CSV เพื่อทำนายค่าตัวเลข (เช่น ราคาบ้าน)
=======
🏠 California Housing Price Predictor
แอป Streamlit สำหรับทำนายราคาบ้านจากโมเดล Regression ที่เทรนไว้บน Google Colab

วิธีรัน:
    streamlit run app.py

โครงสร้างไฟล์ที่ต้องมี:
    app.py
    model/regression_model.pkl
    model/scaler.pkl
    model/feature_names.pkl
>>>>>>> 450bdf23b2690053f5bb5872310a8433be06a666
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
<<<<<<< HEAD
import plotly.express as px
=======
import os
>>>>>>> 450bdf23b2690053f5bb5872310a8433be06a666

# ------------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ------------------------------------------------------------------
st.set_page_config(
<<<<<<< HEAD
    page_title="Regression Predictor",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------
# ธีมสีและ CSS แบบเรียบง่าย (Minimal / Clean)
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #fafafa;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            font-weight: 600;
        }
        .stButton>button {
            background-color: #0F766E;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #0B5A54;
            color: white;
        }
        div[data-testid="metric-container"] {
            background-color: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem;
        }
        .result-badge {
            display: inline-block;
            padding: 0.4rem 1rem;
            border-radius: 999px;
            background-color: #ECFDF5;
            color: #0F766E;
            font-weight: 600;
            font-size: 1.1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
=======
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
>>>>>>> 450bdf23b2690053f5bb5872310a8433be06a666

@st.cache_resource
<<<<<<< HEAD
def load_model(path: str = "regression_model.pkl"):
    bundle = joblib.load(path)
    return (
        bundle["scaler"],
        bundle["model"],
        bundle["model_name"],
        bundle["feature_cols"],
        bundle["target_col"],
    )
=======
def load_artifacts():
    model_path = os.path.join(MODEL_DIR, "regression_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_names.pkl")
>>>>>>> 450bdf23b2690053f5bb5872310a8433be06a666

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        return None, None, None

<<<<<<< HEAD
try:
    scaler, reg_model, model_name, feature_cols, target_col = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ---------------------------------------------------------------
# ส่วนหัวของหน้าเว็บ
# ---------------------------------------------------------------
st.title("📈 Regression Predictor")
st.write("เว็บแอปสำหรับทำนายค่าตัวเลข (เช่น ราคาบ้าน) โดยใช้โมเดล Regression ที่เทรนไว้แล้ว")
=======
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()
>>>>>>> 450bdf23b2690053f5bb5872310a8433be06a666

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
<<<<<<< HEAD
        "ไม่พบไฟล์โมเดล `regression_model.pkl` กรุณาวางไฟล์นี้ไว้ในโฟลเดอร์เดียวกับ `app.py` "
        "(ไฟล์นี้ได้จากการรัน train_regression_colab.ipynb บน Google Colab แล้วดาวน์โหลดมา)"
=======
        "❌ ไม่พบไฟล์โมเดล กรุณาวางโฟลเดอร์ `model/` "
        "(ที่มี regression_model.pkl, scaler.pkl, feature_names.pkl) "
        "ไว้ในโฟลเดอร์เดียวกับ app.py"
>>>>>>> 450bdf23b2690053f5bb5872310a8433be06a666
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

<<<<<<< HEAD
# ---------------------------------------------------------------
# แถบด้านข้าง: ข้อมูลโมเดล + เลือกโหมด
# ---------------------------------------------------------------
st.sidebar.header("⚙️ ตั้งค่า")
mode = st.sidebar.radio("เลือกวิธีการใส่ข้อมูล", ["กรอกข้อมูลเอง", "อัปโหลดไฟล์ CSV"])

st.sidebar.divider()
st.sidebar.write(f"**โมเดลที่ใช้:** {model_name}")
st.sidebar.write(f"**ตัวแปรเป้าหมาย:** {target_col}")
with st.sidebar.expander("ดูรายชื่อ Feature ที่โมเดลใช้"):
    for col in feature_cols:
        st.write(f"- {col}")

# ค่าเริ่มต้นแบบคร่าวๆ (อ้างอิงช่วงค่าของ California Housing dataset)
default_values = {
    "MedInc": 3.87, "HouseAge": 28.0, "AveRooms": 5.4, "AveBedrms": 1.1,
    "Population": 1425.0, "AveOccup": 3.0, "Latitude": 35.6, "Longitude": -119.5,
}

# ---------------------------------------------------------------
# โหมดที่ 1: กรอกข้อมูลเอง
# ---------------------------------------------------------------
if mode == "กรอกข้อมูลเอง":
    st.subheader("📝 กรอกข้อมูลเพื่อทำนายค่า")
=======
# ------------------------------------------------------------------
# ฟอร์มกรอกข้อมูล
# ------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("📋 กรอกข้อมูลพื้นที่ / บ้าน")
>>>>>>> 450bdf23b2690053f5bb5872310a8433be06a666

    col1, col2 = st.columns(2)
    user_input = {}

<<<<<<< HEAD
    for i, feature in enumerate(feature_cols):
        target_col_widget = col1 if i % 2 == 0 else col2
        with target_col_widget:
            default_val = default_values.get(feature, 0.0)
            input_values[feature] = st.number_input(
                feature, value=float(default_val), step=0.1, format="%.3f", key=feature
            )

    if st.button("🔍 ทำนายค่า", use_container_width=False):
        input_df = pd.DataFrame([input_values])[feature_cols]
        scaled_input = scaler.transform(input_df)
        prediction = reg_model.predict(scaled_input)[0]

        st.divider()
        m1, m2 = st.columns([1, 2])
        with m1:
            st.metric(f"ค่าทำนาย ({target_col})", f"{prediction:,.3f}")
        with m2:
            st.markdown(
                f'<span class="result-badge">ผลการทำนาย: {prediction:,.3f}</span>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------
# โหมดที่ 2: อัปโหลดไฟล์ CSV
# ---------------------------------------------------------------
else:
    st.subheader("📂 อัปโหลดไฟล์ CSV เพื่อทำนายหลายแถวพร้อมกัน")
    st.caption(f"ไฟล์ CSV ต้องมีคอลัมน์: {', '.join(feature_cols)}")
=======
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
>>>>>>> 450bdf23b2690053f5bb5872310a8433be06a666

    with st.expander("ดูรายละเอียดข้อมูลที่กรอก"):
        st.dataframe(input_df.T.rename(columns={0: "ค่าที่กรอก"}), use_container_width=True)

<<<<<<< HEAD
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        missing_cols = [c for c in feature_cols if c not in df_upload.columns]
        if missing_cols:
            st.error(f"ไฟล์ CSV ขาดคอลัมน์: {missing_cols}")
        else:
            X_new = df_upload[feature_cols].fillna(df_upload[feature_cols].median())
            X_scaled = scaler.transform(X_new)
            predictions = reg_model.predict(X_scaled)

            df_result = df_upload.copy()
            df_result[f"predicted_{target_col}"] = predictions

            st.success(f"ทำนายสำเร็จ {len(df_result)} แถว")
            st.dataframe(df_result, use_container_width=True)

            # กราฟแสดงการกระจายตัวของค่าทำนาย
            fig = px.histogram(
                df_result,
                x=f"predicted_{target_col}",
                nbins=30,
                title=f"การกระจายตัวของค่าทำนาย ({target_col})",
                color_discrete_sequence=["#0F766E"],
            )
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=13),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ปุ่มดาวน์โหลดผลลัพธ์
            csv_out = df_result.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ ดาวน์โหลดผลลัพธ์ (CSV)",
                data=csv_out,
                file_name="regression_result.csv",
                mime="text/csv",
            )

st.divider()
st.caption("สร้างด้วย Streamlit + scikit-learn (Regression)")
=======
st.markdown(
    "<p style='text-align:center; color:#9CA3AF; font-size:0.85rem; margin-top:2rem;'>"
    "สร้างด้วย Streamlit • โมเดลเทรนบน Google Colab</p>",
    unsafe_allow_html=True
)
>>>>>>> 450bdf23b2690053f5bb5872310a8433be06a666
