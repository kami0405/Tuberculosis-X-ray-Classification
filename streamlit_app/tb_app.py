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

# --- Image preprocessing (match training) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),           # resize exactly as in training
    transforms.ToTensor(),                   # convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # mean from training
                         [0.229, 0.224, 0.225])  # std from training
])

def load_image(image):
    image = image.convert('RGB')             # ensure RGB
    image = transform(image).unsqueeze(0)    # add batch dimension
    return image.to(device)

# --- Prediction function with threshold ---
CONFIDENCE_THRESHOLD = 0.8  # adjust to reduce false positives

def predict(image):
    image = load_image(image)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred_class_index = outputs.argmax(1).item()
        pred_confidence = probs[0][pred_class_index].item()

    classes = ['Normal', 'Tuberculosis']
    
    # Apply threshold
    if pred_class_index == 1 and pred_confidence < CONFIDENCE_THRESHOLD:
        pred_class_index = 0
        pred_confidence = 1 - pred_confidence  # adjust confidence

    return classes[pred_class_index], pred_confidence

# --- Streamlit UI ---
st.title("💻 TB Chest X-ray Detector")
st.write("Upload a chest X-ray image and get a prediction for Tuberculosis.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    pred_class, confidence = predict(image)
    st.write(f"**Prediction:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.4f}")
