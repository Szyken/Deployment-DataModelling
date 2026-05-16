import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
IMG_SIZE = 100
categories = ['Normal', 'COVID']

# =========================
# LOAD MODEL
# =========================
from tensorflow.keras.models import model_from_json

@st.cache_resource
def load_cnn_model():

    # Load architecture
    with open("model_architecture.json", "r") as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    # Load weights
    model.load_weights("model_weights.h5")

    return model

model = load_cnn_model()

# =========================
# PREPROCESSING
# =========================
def preprocess_image(image):

    # Convert PIL -> NumPy
    image = np.array(image)

    # Convert RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize sesuai training
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Normalisasi
    image = image.astype('float32') / 255.0

    # Tambahkan dimensi batch
    image = np.expand_dims(image, axis=0)

    return image

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title='COVID CNN Detection', layout='centered')

st.title('COVID-19 Detection Using CNN')
st.write('Upload gambar X-Ray untuk mendeteksi Normal atau COVID.')

uploaded_file = st.file_uploader(
    'Upload Gambar X-Ray',
    type=['jpg', 'jpeg', 'png']
)

st.caption('Deep Learning CNN Model - Streamlit Deployment')
