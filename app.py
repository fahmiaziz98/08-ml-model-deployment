import os
import torch
import streamlit as st
from transformers import pipeline, AutoImageProcessor
from PIL import Image
from utils import download_model_from_s3

model_paths = {
    "TinyBert Sentiment Analysis": "ml-models/tinybert-sentiment-analysis/",
    "TinyBert Disaster Classification": "ml-models/tinybert-disaster-tweet/",
    "VIT Pose Classification": "ml-models/vit-human-pose-classification/"
}


st.title("Machine Learning Model Deployment")

model_choice = st.selectbox(
    "Select Model:",[
        "TinyBert Sentiment Analysis",
        "TinyBert Disaster Classification",
        "VIT Pose Classification"   
    ]
)

local_path = model_choice.lower().replace(" ", "-")
s3_prefix = model_paths[model_choice]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if "downloaded_models" not in st.session_state:
    st.session_state.downloaded_models = set()


if model_choice not in st.session_state.downloaded_models:
    if st.button(f"Download {model_choice}"):
        with st.spinner(f"Downloading {model_choice}... Please wait!"):
            download_model_from_s3(local_path, s3_prefix)
        st.session_state.downloaded_models.add(model_choice)
        st.toast(f"✅ {model_choice} Succesfuly Download!", icon="🎉")

# **1. Sentiment Analysis Model**
if model_choice == "TinyBert Sentiment Analysis":
    text = st.text_area("Enter Text:", "This movie was horrible, the plot was really boring. acting was okay")
    predict = st.button("Predict Sentiment")
    try:
        classifier = pipeline("text-classification", model=local_path, device=device)
    except OSError:
         st.error("❌ Model not found. Please download the model first.")
         classifier = None

    if predict and classifier:
        with st.spinner("Predicting..."):
            output = classifier(text)
            st.write(output)

# **2. Disaster Classification**
if model_choice == "TinyBert Disaster Classification":
    text = st.text_area("Enter Text:", "There is a fire in the building")
    predict = st.button("Predict Sentiment")
    try:
        classifier = pipeline("text-classification", model=local_path, device=device)
    except OSError:
        st.error("❌ Model not found. Please download the model first.")
        classifier = None

    if predict and classifier:
        with st.spinner("Predicting..."):
            output = classifier(text)
            st.write(output)

# **3. Image Classification**
if model_choice == "VIT Pose Classification":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    predict = st.button("Predict Image")  

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Your Image", width=300)

        with col2:
            try:
                image_processor = AutoImageProcessor.from_pretrained(local_path, use_fast=True)
                pipe = pipeline('image-classification', model=local_path, image_processor=image_processor, device=device)
            except OSError:
                st.error("❌ Model not found. Please download the model first.")
                pipe = None

            if predict and pipe:
                with st.spinner("Predicting..."):
                    output = pipe(image)
                    st.write("### Prediction Results:")
                    st.json(output)  
