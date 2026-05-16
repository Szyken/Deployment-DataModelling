import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

@st.cache_resource
def load_cnn_model():

    model = Sequential([

        Conv2D(
            32,
            (3,3),
            activation='relu',
            padding='same',
            input_shape=(100,100,3)
        ),

        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(
            64,
            (3,3),
            activation='relu',
            padding='same'
        ),

        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(
            128,
            (3,3),
            activation='relu',
            padding='same'
        ),

        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),

        Dense(128, activation='relu'),

        Dropout(0.5),

        Dense(1, activation='sigmoid')
    ])

    model.load_weights("model_weights.weights.h5")

    return model


model = load_cnn_model()

st.title("Deteksi Pneumonia CNN")

uploaded_file = st.file_uploader(
    "Upload Gambar X-Ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    img = np.array(image)

    img = cv2.resize(img, (100,100))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    st.image(image, caption="Gambar Uploaded")

    if prediction[0][0] > 0.5:
        st.error("Pneumonia")
    else:
        st.success("Normal")
