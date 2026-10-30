
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import urllib.request
from PIL import Image

st.set_page_config(page_title="Skin Disease Detection")

st.title("Skin Disease Detection Using CNN")
st.write("Upload a skin image for prediction.")

MODEL_PATH = "skin_disease_cnn.keras"

MODEL_URL = "https://github.com/KAINATIMRAN2007/Skin-Disease-Detection-CNN/releases/download/v1.0/skin_disease_cnn.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading CNN model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

with open("class_names.json", "r") as f:
    class_names = json.load(f)

uploaded_file = st.file_uploader(
    "Upload skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    index = np.argmax(prediction[0])

    st.subheader("Prediction")
    st.write(class_names[index])
    st.write(f"Confidence: {prediction[0][index] * 100:.2f}%")

st.warning("Educational use only. Not a medical diagnosis.")
