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

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_image(image):
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)  # add batch dimension
    return image.to(device)

# --- Prediction function ---
def predict(image):
    image = load_image(image)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(1).item()
    
    classes = ['Normal', 'Tuberculosis']
    return classes[pred_class], probs[0][pred_class].item()

# --- Prediction with threshold ---
def predict_with_threshold(image, threshold=0.5):
    pred_class, confidence = predict(image)
    # Override to Normal if confidence is below threshold for TB
    if pred_class == "Tuberculosis" and confidence < threshold:
        pred_class = "Normal"
    return pred_class, confidence

# --- Streamlit UI ---
st.title("💻 TB Chest X-ray Detector")
st.write("Upload a chest X-ray image and get a prediction for Tuberculosis.")

# --- Confidence threshold slider ---
threshold = st.slider(
    "TB Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Only predict Tuberculosis if probability is above this threshold"
)

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    pred_class, confidence = predict_with_threshold(image, threshold)
    st.write(f"**Prediction:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.4f}")
