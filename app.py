import streamlit as st
import tensorflow as tf
import gdown
import os
from PIL import Image
import numpy as np
import cv2
import time


# -----------------------------
# Streamlit App Config
# -----------------------------
st.set_page_config(
    page_title="Chest X-ray AI",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -----------------------------
# Sidebar UI (Clean & Minimal)
# -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;">
            <h2>🩺 Chest X-ray AI</h2>
            <p style="color:gray;font-size:14px;">
                Pneumonia Detection System
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # --- Model Settings ---
    st.markdown("### ⚙️ Model Settings")

    k_value = st.slider(
        "SVD Components",
        min_value=10,
        max_value=100,
        value=40,
        step=5,
        help="Controls how much image detail is preserved"
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.5, 0.05,
        help="Minimum confidence required for reliable prediction"
    )

    st.divider()

    # --- Optional Tools (Hidden by default) ---
    with st.expander("ℹ️ How it works"):
        st.write(
            """
            • X-ray is compressed using **SVD**
            • Important lung structures are preserved
            • CNN predicts **Normal** or **Pneumonia**
            • Confidence score reflects prediction strength
            """
        )

    with st.expander("🧠 AI Disclaimer"):
        st.write(
            "This system is for **educational and research purposes only**. "
            "It does not replace professional medical diagnosis."
        )

# -----------------------------
# Download & Load Model (cached)
# -----------------------------
@st.cache_resource
def load_xray_model():
    model_path = "model.keras"
    if not os.path.exists(model_path):
        gdown.download(id="1gzOhv1qfBwTto2hNkVnFHyiK3tLWsCoV", output=model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_xray_model()
labels = ["Normal", "Pneumonia"]
colors = {"Normal": "green", "Pneumonia": "red"}

# -----------------------------
# SVD Preprocessing Function
# -----------------------------
def apply_svd(img_gray, k=40):
    img_gray = img_gray.astype(np.float32) / 255.0
    U, S, Vt = np.linalg.svd(img_gray, full_matrices=False)
    img_svd = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    img_svd = np.clip(img_svd, 0, 1)
    return (img_svd * 255).astype(np.uint8)

# -----------------------------
# File Upload & Prediction
# -----------------------------
st.title("🩺 DeepTech Chest X-ray AI (SVD + XAI)")
uploaded = st.file_uploader("Upload Lung X-ray", ["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L")
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (224, 224))
    img_svd = apply_svd(img_resized, k=k_value)
    x_input = (img_svd.astype(np.float32) / 255.0)[None, ..., None]

    # Prediction
    pred = model.predict(x_input)[0]
    pred_label = labels[pred.argmax()]
    confidence = float(pred.max())

    # -----------------------------
    # Display Images Side by Side
    # -----------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_resized, caption="Original Image", width=400)
    with col2:
        st.image(img_svd, caption=f"SVD Processed (k={k_value})", width=400)


    # -----------------------------
    # Prediction Result Box
    # -----------------------------
    st.markdown(
        f"<div style='text-align:center; background-color:#f0f0f0; padding:15px; border-radius:15px;'>"
        f"<h2 style='color:{colors[pred_label]};'>🧪 Prediction: {pred_label}</h2>"
        f"<h4>Confidence: {confidence*100:.2f}%</h4>"
        "</div>",
        unsafe_allow_html=True
    )

    # Confidence Bar
    conf_color = "green" if confidence > confidence_threshold else "red"
    st.progress(confidence)
    st.markdown(f"<p style='color:{conf_color}; font-weight:bold;'>Confidence: {confidence*100:.2f}%</p>", unsafe_allow_html=True)

    if pred_label == "Normal":
        st.balloons()

        # Show AI insight button
show_ai_insight = st.button("Click for AI insight 🧠")

if show_ai_insight:
    with st.spinner("AI is reviewing the scan patterns..."):
        time.sleep(1.5)

    st.write(
        f"🩻 The AI detected **{pred_label} lung patterns** with "
        f"**{confidence * 100:.1f}% confidence**.\n\n"
        "SVD preprocessing preserved key lung structures before the CNN analysis.\n\n"
        "⚠️ This is AI interpretation, not medical advice."
    )




    # Download SVD Image
    buf = BytesIO()
    Image.fromarray(img_svd).save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("💾 Download SVD Image", data=byte_im, file_name="svd_image.png", mime="image/png")




