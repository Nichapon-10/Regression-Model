"""
Streamlit Web App — Regression Prediction
โหลดโมเดล Regression ที่เทรนไว้จาก Google Colab (regression_model.pkl)
แล้วให้ผู้ใช้กรอกข้อมูล หรืออัปโหลด CSV เพื่อทำนายค่าตัวเลข (เช่น ราคาบ้าน)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ---------------------------------------------------------------
st.set_page_config(
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

# ---------------------------------------------------------------
# โหลดโมเดล (cache ไว้ไม่ต้องโหลดซ้ำทุกครั้ง)
# ---------------------------------------------------------------
@st.cache_resource
def load_model(path: str = "regression_model.pkl"):
    bundle = joblib.load(path)
    return (
        bundle["scaler"],
        bundle["model"],
        bundle["model_name"],
        bundle["feature_cols"],
        bundle["target_col"],
    )


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

if not model_loaded:
    st.error(
        "ไม่พบไฟล์โมเดล `regression_model.pkl` กรุณาวางไฟล์นี้ไว้ในโฟลเดอร์เดียวกับ `app.py` "
        "(ไฟล์นี้ได้จากการรัน train_regression_colab.ipynb บน Google Colab แล้วดาวน์โหลดมา)"
    )
    st.stop()

st.divider()

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

    col1, col2 = st.columns(2)
    input_values = {}

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

    uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type=["csv"])

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