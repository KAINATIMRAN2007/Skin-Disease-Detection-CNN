
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Skin Disease Detection", page_icon="🔬")

st.title("Skin Disease Detection Using CNN")
st.write("Upload a skin image to see the model's prediction.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_disease_cnn.keras")

model = load_model()

with open("class_names.json", "r") as file:
    class_names = json.load(file)

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction[0])

    st.subheader("Model Prediction")
    st.write(class_names[predicted_index])
    st.write(f"Model confidence: {prediction[0][predicted_index] * 100:.2f}%")

st.warning(
    "Educational demonstration only. This model is not a medical "
    "diagnostic tool. Consult a qualified healthcare professional."
)
