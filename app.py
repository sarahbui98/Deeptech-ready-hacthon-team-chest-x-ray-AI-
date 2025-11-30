import streamlit as st
import tensorflow as tf
import gdown
import os
from PIL import Image
import numpy as np
import cv2
import time

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import time

# ------------------------------
# Sidebar: Team Info
# ------------------------------
st.sidebar.title("Team Info")
team_name = st.sidebar.text_input("Enter your team name:", "Team Chest-Xray AI")
st.sidebar.write(f"Team: **{team_name}**")

members = ["Maryam Ibrahim Hamza", "Saratu Banau Salihu"]
st.sidebar.markdown("### 🌟 **Team Members:**")
for m in members:
    st.sidebar.markdown(f"✅ **{m}**")

st.sidebar.markdown("---")
st.sidebar.markdown("💡 _AI system for pneumonia detection using SVD + CNN_")

# ------------------------------
# Load Trained Model
# ------------------------------
MODEL_PATH = "assets/model.h5"

@st.cache_resource(show_spinner=True)
def load_xray_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_xray_model(MODEL_PATH)

labels = ["Normal", "Pneumonia"]
colors = {"Normal": "green", "Pneumonia": "red"}

# ------------------------------
# SVD Preprocessing
# ------------------------------
def apply_svd(img_gray, k=40):
    img_gray = img_gray.astype(np.float32) / 255.0
    U, S, Vt = np.linalg.svd(img_gray, full_matrices=False)
    img_svd = np.dot(np.dot(U[:, :k], np.diag(S[:k])), Vt[:k, :])
    img_svd = np.clip(img_svd, 0, 1)
    img_svd = (img_svd * 255).astype(np.uint8)
    return img_svd

# ------------------------------
# Main Interface
# ------------------------------
st.title("DeepTech Chest-Xray AI 🩺 (SVD + XAI)")

uploaded = st.file_uploader("Upload Lung X-ray", ["png", "jpg", "jpeg"])

if uploaded and model:
    # Load and preprocess image
    img = Image.open(uploaded).convert("L")
    img = np.array(img)
    img = cv2.resize(img, (224, 224))
    img_svd = apply_svd(img, k=40)

    x_input = img_svd.astype(np.float32) / 255.0
    x_input = np.expand_dims(x_input, (0, -1))  # (1, H, W, 1)

    # Model prediction
    pred = model.predict(x_input)[0]
    class_idx = int(pred.argmax())
    confidence = float(pred.max())
    pred_label = labels[class_idx]

    # Display images
    st.image(img, caption="Original Image", use_column_width=True)
    st.image(img_svd, caption="SVD Processed", use_column_width=True)

    # Colored prediction box
    st.markdown(f"<h2 style='color:{colors[pred_label]};'>Prediction: {pred_label}</h2>", unsafe_allow_html=True)

    # Confidence bar
    st.write("**Confidence:**")
    st.progress(confidence)

    # Celebration if Normal
    if pred_label == "Normal":
        st.balloons()

    # Chatbox: Explain AI result
    st.write("### 💬 Explain result")
    if st.button("Explain AI Decision"):
        with st.spinner("AI is analyzing the scan..."):
            time.sleep(1.5)
        explanation = (
            f"The AI predicts **{pred_label}** "
            f"with **{confidence*100:.1f}% confidence**.\n\n"
            "The image was compressed using **SVD** to focus on key patterns. "
            "A CNN then analyzed the lungs for infection signs.\n\n"
            "⚠️ This is AI interpretation only and not medical advice."
        )
        st.write(explanation)

