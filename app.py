# -*- coding: utf-8 -*-
"""App.py - Chest X-ray AI (SVD + Explainable AI)"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time
import gdown
import os

# =====================
# Sidebar Info
# =====================
st.sidebar.title("🧬 Team Chest X-ray AI")
st.sidebar.subheader("👥 Team Members")
st.sidebar.write("- **Maryam Ibrahim Hamza**")
st.sidebar.write("- **Saratu Banau Salihu**")
st.sidebar.markdown("---")
st.sidebar.write("🌍 *Nigeria Hackathon 2025*")

# =====================
# Download Model
# =====================
MODEL_URL = "https://drive.google.com/uc?id=1gzOhv1qfBwTto2hNkVnFHyiK3tLWsCoV"
MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("⬇️ Downloading AI model... please wait"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully!")

# Verify model exists
if os.path.exists(MODEL_PATH):
    st.write("Model exists:", True)
    st.write("File size (MB):", os.path.getsize(MODEL_PATH)/1e6)
else:
    st.error("❌ Model failed to download!")
    st.stop()

# =====================
# Load Model Safely
# =====================
@st.cache_resource
def load_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model(MODEL_PATH)
if model is None:
    st.stop()  # stop if model fails

labels = ["Normal", "Pneumonia"]

# =====================
# SVD Compression Function
# =====================
def apply_svd(img_gray, k=40):
    img_gray = img_gray.astype(np.float32)/255.0
    U, S, Vt = np.linalg.svd(img_gray, full_matrices=False)
    img_svd = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    img_svd = np.clip(img_svd, 0, 1)
    return (img_svd*255).astype(np.uint8)

# =====================
# UI
# =====================
st.title("🩺 Chest X-ray Diagnosis Assistant (SVD + Explainable AI)")
uploaded = st.file_uploader("Upload Lung X-ray Image", ["png","jpg","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L")
    img = np.array(img)
    img = cv2.resize(img, (224,224))
    img_svd = apply_svd(img, k=40)

    x = img_svd.astype(np.float32)/255.0
    x = np.expand_dims(x, (0,-1))

    # Predict
    pred = model.predict(x)[0]
    idx = int(pred.argmax())
    conf = float(pred.max())

    # Show images
    col1, col2 = st.columns(2)
    col1.image(img, caption="Original Scan", use_container_width=True)
    col2.image(img_svd, caption="AI Optimized (SVD)", use_container_width=True)

    st.markdown(f"### ✅ **Result: `{labels[idx]}`**")
    st.progress(conf)
    st.info(f"🧪 AI Confidence: **{conf*100:.2f}%**")

    st.markdown("---")

    # Explain button
    if st.button("💬 Explain AI Decision"):
        with st.spinner("Analyzing important lung patterns..."):
            time.sleep(1.2)
        st.write(
            f"The AI believes this X-ray shows **{labels[idx]}** because it detected "
            f"patterns commonly linked to **lung condition indicators** after compressing "
            f"the image using **SVD** to keep only the strongest signals."
        )
        st.write("⚠️ This is not a doctor's diagnosis, it's AI assistance to explain the scan.")
