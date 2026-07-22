"""
Streamlit Web App — K-Means Cluster Prediction
โหลดโมเดล K-Means ที่เทรนไว้จาก Google Colab (kmeans_model.pkl)
แล้วให้ผู้ใช้กรอกข้อมูล หรืออัปโหลด CSV เพื่อทำนายว่าข้อมูลอยู่กลุ่มไหน
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.decomposition import PCA

# ---------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ---------------------------------------------------------------
st.set_page_config(
    page_title="K-Means Cluster Predictor",
    page_icon="🔮",
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
            background-color: #4F46E5;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #4338CA;
            color: white;
        }
        div[data-testid="metric-container"] {
            background-color: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem;
        }
        .cluster-badge {
            display: inline-block;
            padding: 0.4rem 1rem;
            border-radius: 999px;
            background-color: #EEF2FF;
            color: #4F46E5;
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
def load_model(path: str = "kmeans_model.pkl"):
    bundle = joblib.load(path)
    return bundle["scaler"], bundle["kmeans"], bundle["feature_cols"], bundle["n_clusters"]


try:
    scaler, kmeans_model, feature_cols, n_clusters = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ---------------------------------------------------------------
# ส่วนหัวของหน้าเว็บ
# ---------------------------------------------------------------
st.title("🔮 K-Means Cluster Predictor")
st.write("เว็บแอปสำหรับทำนายกลุ่ม (Cluster) ของข้อมูล โดยใช้โมเดล K-Means ที่เทรนไว้แล้ว")

if not model_loaded:
    st.error(
        "ไม่พบไฟล์โมเดล `kmeans_model.pkl` กรุณาวางไฟล์นี้ไว้ในโฟลเดอร์เดียวกับ `app.py` "
        "(ไฟล์นี้ได้จากการรัน train_kmeans_colab.ipynb บน Google Colab แล้วดาวน์โหลดมา)"
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------
# แถบด้านข้าง: เลือกโหมดการทำนาย
# ---------------------------------------------------------------
st.sidebar.header("⚙️ ตั้งค่า")
mode = st.sidebar.radio("เลือกวิธีการใส่ข้อมูล", ["กรอกข้อมูลเอง", "อัปโหลดไฟล์ CSV"])

st.sidebar.divider()
st.sidebar.write(f"**จำนวนกลุ่มของโมเดล:** {n_clusters} clusters")
st.sidebar.write(f"**Feature ที่โมเดลใช้:** {len(feature_cols)} ตัว")
with st.sidebar.expander("ดูรายชื่อ Feature"):
    for col in feature_cols:
        st.write(f"- {col}")

# ---------------------------------------------------------------
# โหมดที่ 1: กรอกข้อมูลเอง
# ---------------------------------------------------------------
if mode == "กรอกข้อมูลเอง":
    st.subheader("📝 กรอกข้อมูลเพื่อทำนายกลุ่ม")

    col1, col2 = st.columns(2)
    input_values = {}

    for i, feature in enumerate(feature_cols):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            input_values[feature] = st.number_input(
                feature, value=0.0, step=0.1, format="%.2f", key=feature
            )

    if st.button("🔍 ทำนายกลุ่ม", use_container_width=False):
        input_df = pd.DataFrame([input_values])[feature_cols]
        scaled_input = scaler.transform(input_df)
        cluster = kmeans_model.predict(scaled_input)[0]

        st.divider()
        m1, m2 = st.columns([1, 2])
        with m1:
            st.metric("ผลการทำนาย", f"Cluster {cluster}")
        with m2:
            st.markdown(
                f'<span class="cluster-badge">ข้อมูลนี้อยู่ในกลุ่มที่ {cluster}</span>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------
# โหมดที่ 2: อัปโหลดไฟล์ CSV
# ---------------------------------------------------------------
else:
    st.subheader("📂 อัปโหลดไฟล์ CSV เพื่อทำนายกลุ่มหลายแถวพร้อมกัน")
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
            clusters = kmeans_model.predict(X_scaled)

            df_result = df_upload.copy()
            df_result["cluster"] = clusters

            st.success(f"ทำนายสำเร็จ {len(df_result)} แถว")
            st.dataframe(df_result, use_container_width=True)

            # กราฟแสดงผลด้วย PCA 2 มิติ
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)
            plot_df = pd.DataFrame(coords, columns=["PCA1", "PCA2"])
            plot_df["cluster"] = clusters.astype(str)

            fig = px.scatter(
                plot_df,
                x="PCA1",
                y="PCA2",
                color="cluster",
                title="การกระจายตัวของกลุ่มข้อมูล (PCA 2D)",
                color_discrete_sequence=px.colors.qualitative.Set2,
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
                file_name="cluster_result.csv",
                mime="text/csv",
            )

st.divider()
st.caption("สร้างด้วย Streamlit + scikit-learn (K-Means Clustering)")