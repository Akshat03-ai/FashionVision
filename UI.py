import streamlit as st
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import joblib

# Load model
model = joblib.load("model.pkl")

labels_map = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Title & info
st.title("👕 FashionVision")
st.write("Upload an image or enter a URL to predict the clothing type.")
st.markdown("**Possible Categories:**")
for i, label in enumerate(labels_map):
    st.write(f"{i} → {label}")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# URL input
url = st.text_input("Or enter an image URL")

image = None
if uploaded_file:
    image = Image.open(uploaded_file)
elif url:
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
    except:
        st.error("Could not load image from URL")

if image:
    st.image(image, caption="Input Image", use_column_width=True)

    # Convert to Fashion-MNIST format
    img_resized = ImageOps.grayscale(image)
    img_resized = ImageOps.invert(img_resized)  # match Fashion-MNIST style
    img_resized = img_resized.resize((28, 28))
    img_array = np.array(img_resized).reshape(1, -1) / 255.0
    img_resized = ImageOps.autocontrast(img_resized)  # normalize contrast

    # Predict
    prediction = model.predict(img_array)[0]
    probabilities = model.predict_proba(img_array)[0]

    # Show main prediction
    st.success(f"Prediction: **{labels_map[prediction]}**")

    # Show confidence scores
    st.subheader("Confidence Scores:")
    for label, prob in sorted(zip(labels_map, probabilities), key=lambda x: x[1], reverse=True):
        st.write(f"{label}: {prob*100:.2f}%")
