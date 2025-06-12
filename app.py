import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import base64

st.set_page_config(
    page_title="Green Guard - Guava Leaf Classifier",
    page_icon="🌿",
    layout="centered"
)

# Function to set the background image from local file
def set_bg_from_local(img_file):
    with open(img_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string.decode()}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            font-family: 'Segoe UI', sans-serif;
        }}
        .title {{
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            color: #1B4332;
            margin-bottom: 5px;
            text-shadow: 2px 2px 5px #FFFFFF;
        }}
        .subheader {{
            text-align: center;
            font-size: 1.2em;
            color: #1B4332;
            font-weight: bold;
            margin-bottom: 30px;
            text-shadow: 1px 1px 2px #FFFFFF;
        }}
        .upload-box {{
            background: rgba(255, 255, 255, 0.7);
            border-radius: 15px;
            padding: 15px;
            margin: 20px auto;
            width: 450px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            color: black;
            font-weight: bold;
        }}
        .center-btn button {{
            display: block;
            margin: 20px auto 10px auto;
            border-radius: 12px !important;
            background-color: #40916C !important;
            color: white !important;
            padding: 12px 24px !important;
            font-size: 16px !important;
            font-weight: bold !important;
        }}
        .result-box {{
            background: rgba(255, 255, 255, 0.75);
            border-radius: 15px;
            padding: 20px;
            margin: 20px auto;
            width: 450px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            font-size: 18px;
            color: #000;
            font-weight: bold;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
set_bg_from_local("Background_img.jpg")

@st.cache_resource
def load_cnn_model():
    return load_model("GLD_Binary_Classification_Final.h5")

model = load_cnn_model()

# Header
st.markdown('<div class="title">🌿 Green Guard 🌿</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subheader">AI-powered Guava Leaf Disease Detector</div>', 
    unsafe_allow_html=True
)

# Upload Box
st.markdown(
    '<div class="upload-box">📤 Upload Guava Leaf Images (JPG/PNG):</div>', 
    unsafe_allow_html=True)

# Allow multiple file uploads
uploaded_files = st.file_uploader(
    "", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True, 
    label_visibility="collapsed"
)

if uploaded_files:
    # Analyze Button
    analyze_btn = st.button("🔍 Analyze All Uploaded Leaves", key="analyze_btn")
    
    if analyze_btn:
        for uploaded_file in uploaded_files:
            st.markdown(f"**File: {uploaded_file.name}**")
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_container_width=True)

            img_resized = img.resize((256, 256))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            with st.spinner('Analyzing...'):
                time.sleep(1)
                prediction = model.predict(img_array)[0][0]

                if prediction > 0.5:
                    label = "🟢 Healthy Leaf"
                    confidence = prediction
                else:
                    label = "🔴 Diseased Leaf"
                    confidence = 1 - prediction

                # Result Box
                st.markdown(
                    f'<div class="result-box">Result: {label}<br>Confidence: {confidence*100:.2f}%</div>',
                    unsafe_allow_html=True
                )
                # Progress Bar
                st.progress(int(confidence * 100), text=f"Confidence: {confidence*100:.2f}%")

else:
    st.markdown(
        '<div class="upload-box">👈 Please upload one or more guava leaf images to begin analysis.</div>', 
        unsafe_allow_html=True)