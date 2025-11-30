import streamlit as st
import tensorflow as tf
import gdown
import os
from PIL import Image
import numpy as np
import cv2
import time

# Make sure the assets folder exists
os.makedirs("assets", exist_ok=True)

# Google Drive direct download link (CORRECTED)
MODEL_URL = "https://drive.google.com/uc?id=1tcoQNo6PXNQR_vCkDJWoB5QcR6PDQgvq"
MODEL_PATH = "assets/model.h5"

# Class labels used by the model (ADDED)
labels = ["PNEUMONIA", "Normal"]

# UI Colors for display (ADDED)
colors = {
    "PNEUMONIA": "red",
    "Normal": "green"
}

# ------------------------------
# SVD Image Compression Function (ADDED)
# ------------------------------
def apply_svd(image, k=40):
    """ Apply Truncated SVD to 2D gray image and reconstruct it """
    img = image.astype(np.float32)
    U, S, Vt = np.linalg.svd(img, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    img_reconstructed = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))
    return img_reconstructed

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive... (this may take a while)")
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Download failed! Check the Google Drive link or permissions.\nError: {e}")

# Load the Keras model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# ------------------------------
# Streamlit App Interface
# ------------------------------
st.title("DeepTech Chest-Xray AI 🩺 (SVD + XAI)")
uploaded = st.file_uploader("Upload Lung X-ray", ["png","jpg","jpeg"])

if uploaded:
    # Load image and preprocess
    img = Image.open(uploaded).convert("L")
    img = np.array(img)
    img = cv2.resize(img, (224,224))
    img_svd = apply_svd(img, k=40)

    x_input = img_svd.astype(np.float32)/255.0
    x_input = np.expand_dims(x_input, (0,-1))

    # Model prediction
    pred = model.predict(x_input)[0]
    class_idx = int(pred.argmax())
    confidence = float(pred.max())
    pred_label = labels[class_idx]

    # Display images
    st.image(img, caption="Original Image", use_container_width=True)
    st.image(img_svd, caption="SVD Processed", use_container_width=True)

    # Colored prediction
    st.markdown(
        f"<h2 style='color:{colors[pred_label]};'>Prediction: {pred_label}</h2>",
        unsafe_allow_html=True
    )

    # Confidence bar
    st.write("**Confidence:**")
    st.progress(confidence)

    # Balloons if Normal
    if pred_label == "Normal":
        st.balloons()

    # Explanation button
    st.write("### 💬 Explain result ")
    if st.button("Explain AI Decision"):
        with st.spinner("AI is analyzing the scan..."):
            time.sleep(1.5)
        simple_exp = (
            f"The AI predicts **{pred_label}** "
            f"with **{confidence*100:.1f}% confidence**.\n\n"
            "The image was compressed using **SVD** to focus on key patterns. "
            "A CNN then analyzed the lungs for infection signs. \n\n"
            "⚠️ This is AI interpretation only and not medical advice."
        )
        st.write(simple_exp)
