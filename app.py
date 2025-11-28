import streamlit as st
import tensorflow as tf
import gdown
import os
from PIL import Image
import numpy as np
import cv2
import time

# ------------------------------
import os
import streamlit as st
import tensorflow as tf
import gdown

# Model URL and path
MODEL_URL = "https://drive.google.com/uc?id=1tcoQNo6PXNQR_vCkDJWoB5QcR6PDQgvq"
MODEL_PATH = "assets/model.h5"

# Ensure assets folder exists
if not os.path.exists("assets"):
    os.makedirs("assets")  # <-- Create the folder if missing

# Download model if not present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive... (this may take a while)")
    try:
        # gdown.download returns the path to the downloaded file
        downloaded_path = gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        if downloaded_path is None:
            st.error("Download failed! Check the Google Drive link or permissions.")
        else:
            st.success(f"Model downloaded successfully to {downloaded_path}")
    except Exception as e:
        st.error(f"Failed to download model: {e}")

# Load the Keras model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")


# ------------------------------
# Streamlit App Interface
# ------------------------------
st.title("Chest X-Ray AI")
st.write("Upload an X-Ray image for classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # adjust size as per your model input
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    if 'model' in locals():
        with st.spinner("Predicting..."):
            prediction = model.predict(img)
            st.write(f"Prediction: {prediction}")
    else:
        st.error("Model is not loaded yet.")

