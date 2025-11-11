# 🐟 Multiclass Fish Image Classification (TensorFlow + Streamlit)

This project is a **multiclass image classification pipeline** for identifying fish species using **deep learning models** (both custom CNN and transfer learning models like VGG16, ResNet50, MobileNetV2, EfficientNetB0).  
It is **Colab-ready** and includes an optional **Streamlit web app** for image prediction.

---

## 🚀 Features
- Supports **five models**:
  - Scratch CNN (custom architecture)
  - VGG16 (Transfer Learning)
  - ResNet50
  - MobileNetV2
  - EfficientNetB0
- Auto data augmentation using `ImageDataGenerator`
- Automatic training time and accuracy comparison
- Classification report and confusion matrix visualization
- Streamlit UI for uploading and predicting fish species
- Google Drive integration for dataset and model saving

---

## 📂 Folder Structure
multiclass_fish_image_classsification/
│
├── streamlit_app1.py # Streamlit web app
├── fish_classify(1).ipynb # Colab notebook (training + evaluation)
└── README.md
## 📊 Dataset

- The dataset must be placed in your Google Drive at:
- https://drive.google.com/file/d/1aSpJ9T6I8cQUjzMVlfN662fuewDN8V1-/view?usp=sharing

🧠 Training the Models (Colab)

Open the notebook in Google Colab.

Mount your Google Drive:
'from google.colab import drive
drive.mount('/content/drive')'
Set your dataset path:
DATA_DIR = "/content/drive/MyDrive/data/data"

Run all cells to:

Train all 5 models

Save trained models to /content/outputs

Generate evaluation metrics

The best-performing models will be saved as:
'/content/outputs/{model_name}_best.h5'

The app (streamlit_app.py) allows users to:

Upload an image (jpg, jpeg, or png)

Choose a trained model from the outputs/ folder

Predict fish species with confidence scores

Display top 3 predicted classes
Run locally: streamlit run streamlit_app1.py
Output models from fish_classify.ipynb file.
  https://drive.google.com/file/d/1broyoIRtlEDPmeaaJFzjsgzUsCvgZVmA/view?usp=sharing

