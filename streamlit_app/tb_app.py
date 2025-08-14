# tb_app.py
import os
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# --- Device ---
device = torch.device("cpu")  # change to "cuda" if GPU available

# --- Load model architecture and trained weights ---
model = models.densenet121(pretrained=False)
num_features = model.classifier.in_features
model.classifier = torch.nn.Linear(num_features, 2)  # 2 classes: Normal, TB
MODEL_PATH = os.path.join(os.path.dirname(__file__), "tb_detector_densenet121.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- Image preprocessing (match training exactly) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),         # same size used in training
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],       # same normalization as training
        std=[0.229, 0.224, 0.225]
    )
])

def load_image(image):
    # Convert image to RGB and apply training preprocessing
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)  # add batch dimension
    return image.to(device)

# --- Prediction function with threshold ---
def predict(image, threshold=0.5):
    image = load_image(image)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred_class_idx = outputs.argmax(1).item()

    classes = ['Normal', 'Tuberculosis']
    pred_prob = probs[0][pred_class_idx].item()

    # Apply threshold: if TB probability < threshold, classify as Normal
    if pred_class_idx == 1 and pred_prob < threshold:
        pred_class_idx = 0
        pred_prob = probs[0][0].item()

    return classes[pred_class_idx], pred_prob

# --- Streamlit UI ---
st.title("💻 TB Chest X-ray Detector")
st.write("Upload a chest X-ray image and get a prediction for Tuberculosis.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Optional: adjust threshold for more conservative TB detection
    threshold = st.slider("TB Probability Threshold", 0.0, 1.0, 0.7, 0.01)

    pred_class, confidence = predict(image, threshold=threshold)
    st.write(f"**Prediction:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.4f}")
