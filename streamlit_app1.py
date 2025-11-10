import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob

st.set_page_config(page_title="🐟 Fish Species Classifier", layout="centered")

st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload a fish image to predict its species using a deep learning model trained on multiple fish categories.")

# ---- Helper Functions ----
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(img, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_class_labels():
    """
    Priority:
    1. Try outputs/class_indices.txt (saved from training)
    2. If not found, infer from train folder structure
    3. Fallback to generic names
    """
    labels = []
    if os.path.exists("outputs/class_indices.txt"):
        with open("outputs/class_indices.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
    elif os.path.exists("data/train"):
        subfolders = sorted(os.listdir("data/train"))
        labels = [f for f in subfolders if os.path.isdir(os.path.join("data/train", f))]
    else:
        labels = [f"Class {i}" for i in range(10)]  # fallback
    return labels

# ---- Sidebar Model Selection ----
st.sidebar.header("⚙️ Model Selection")

# Automatically find models in outputs/ folder
available_models = glob.glob("outputs/*.h5")
if not available_models:
    st.sidebar.warning("⚠️ No models found in outputs/. Please train and save at least one .h5 model.")
    model_path = None
else:
    model_path = st.sidebar.selectbox("Select a trained model", available_models, index=0)
    model = load_model(model_path)
    st.sidebar.success(f"✅ Loaded model: {os.path.basename(model_path)}")

# Load labels
labels = load_class_labels()
st.sidebar.info(f"📘 Loaded {len(labels)} class labels.")

# ---- Upload Image ----
uploaded_file = st.file_uploader("📤 Upload a fish image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        if model_path is None:
            st.error("Please select or load a model first!")
        else:
            st.write("⏳ Classifying... Please wait...")
            img_array = preprocess_image(img)
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds)
            confidence = float(np.max(preds))

            # Ensure label index is valid
            pred_label = labels[pred_idx] if pred_idx < len(labels) else f"Class {pred_idx}"

            st.success(f"✅ Prediction: **{pred_label}**")
            st.info(f"Confidence: **{confidence*100:.2f}%**")

            # ---- Show top 3 predictions ----
            top3_idx = preds[0].argsort()[-3:][::-1]
            st.write("### 🔝 Top 3 Predictions")
            for i in top3_idx:
                lbl = labels[i] if i < len(labels) else f"Class {i}"
                st.write(f"- {lbl}: {preds[0][i]*100:.2f}%")

# ---- Footer ----
st.markdown("---")
st.caption("Developed with ❤️ using TensorFlow & Streamlit")
