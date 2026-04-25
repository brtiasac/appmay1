# Import needed libraries
import streamlit as st
import torch
import timm
import json
import io
import os
import numpy as np
import gdown
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title = "SpotChek - AI Skin Lesion Detection App",
    page_icon = "🔍",
    layout = "centered"
)

# Load class map and model (cached so it only loads once per session)
@st.cache_resource
def load_model():
    model_path = "best_model.pth"
    # Download model weights
    if not os.path.exists(model_path):
        gdown.download(
            id = "1WOcN14qduuF0XypP_J1MfAUml5GSLAaT",
            output = model_path,
            quiet = False
        )

    with open("class_map.json") as f:
        mapping = json.load(f)
    idx2class = {int(k): v for k, v in mapping["idx2class"].items()}
    classes = mapping["classes"]

    # Create model architecture
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, idx2class, classes

# Load everything when the app starts
model, idx2class, classes = load_model()

# Transformations to apply to uploaded images before feeding to the model
eval_tfm = T.Compose([
    T.Resize(330),
    T.CenterCrop(300),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# Prediction function
def predict(img, top_k=5):
    tensor = eval_tfm(img).unsqueeze(0)
    # Run model inference
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    # Select top 5 predicted classes
    top_idx = probs.argsort()[::-1][:top_k]
    return [(idx2class[i], float(probs[i])) for i in top_idx]

# Add page contents
st.title("SpotChek - AI Skin Lesion Detection")
st.write(
    "Upload an image of your skin concern and our model will show the 5 "
    "most likely skin lesion matches from our 10 categories."
)

with st.expander("View all 10 categories"):
    st.markdown("""
        - Psoriasis
        - Eczema
        - Basal Cell Carcinoma
        - Squamous Cell Carcinoma
        - Folliculitis
        - Melanocytic Nevus
        - Actinic Keratosis
        - Seborrheic Keratosis
        - Pyogenic Granuloma
        - Dermatofibroma
    """)

st.warning(
    "⚠️ This tool is for educational and research purposes only. "
    "It should not be used as a substitute for professional medical diagnosis."
)

# Prompt user to upload image for prediction
uploaded = st.file_uploader(
    "Upload an image of your skin concern below:",
    type = ["jpg", "jpeg", "png", "webp", "bmp"]
)

# Process uploaded image and display prediction
if uploaded:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded image", use_column_width=True)

    with col2:
        with st.spinner("Analysing image..."):
            results = predict(img, top_k=5)
    # Create bar chart to visualise predictions
        st.subheader("Top 5 Predictions")
        for rank, (label, prob) in enumerate(results, 1):
            st.write(f"**{rank}. {label.title()}**")
            st.progress(prob, text=f"{prob:.1%}")
