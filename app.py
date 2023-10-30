


import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

# Load your trained model
model = tf.keras.models.load_model("m1.h5")

# Define a function to make predictions
def predict_emotion(image):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    img = cv2.resize(img, (256, 256))
    img = img / 255
    img = np.expand_dims(img, 0)
    prediction = model.predict(img)
    return "Sad" if prediction > 0.5 else "Happy"

st.title("Emotion Classification")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_image is not None:
    image = uploaded_image.read()
    st.image(image, caption="Uploaded Image", use_column_width=True)
    prediction = predict_emotion(image)

    st.write(f"Prediction: {prediction}")
