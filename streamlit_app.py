import streamlit as st
import torch
import timm
import json
import io
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Condition Classifier",
    page_icon="🔬",
    layout="centered"
)

# ── Load model and class map ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("class_map.json") as f:
        mapping = json.load(f)
    idx2class = {int(k): v for k, v in mapping["idx2class"].items()}
    classes   = mapping["classes"]

    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model, idx2class, classes

model, idx2class, classes = load_model()

# ── Transform ──────────────────────────────────────────────────────────────────
eval_tfm = T.Compose([
    T.Resize(330),
    T.CenterCrop(300),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# ── Predict function ───────────────────────────────────────────────────────────
def predict(img, top_k=5):
    tensor = eval_tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    top_idx = probs.argsort()[::-1][:top_k]
    return [(idx2class[i], float(probs[i])) for i in top_idx]

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🔬 Skin Condition Classifier")
st.write(
    "Upload a skin image and the model will return the top 5 most likely "
    "conditions based on training across the SCIN, Fitzpatrick17k, and DDI datasets."
)

st.warning(
    "⚠️ This tool is for educational and research purposes only. "
    "It is not a medical device and should not be used as a substitute "
    "for professional medical advice or diagnosis."
)

uploaded = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded image", use_column_width=True)

    with col2:
        with st.spinner("Analysing image..."):
            results = predict(img, top_k=5)

        st.subheader("Top 5 predictions")
        for rank, (label, prob) in enumerate(results, 1):
            st.write(f"**{rank}. {label.title()}**")
            st.progress(prob, text=f"{prob:.1%}")

    # Bar chart
    st.subheader("Prediction probabilities")
    labels_p = [r[0].title() for r in results]
    probs_p  = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(8, 3))
    colours = ["#2196F3" if i == 0 else "#90CAF9" for i in range(len(labels_p))]
    ax.barh(labels_p[::-1], probs_p[::-1], color=colours[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    for i, (prob, label) in enumerate(zip(probs_p[::-1], labels_p[::-1])):
        ax.text(prob + 0.01, i, f"{prob:.1%}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
