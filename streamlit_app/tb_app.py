# tb_app.py
import os
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

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
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Test-Time Augmentation: flips + rotations
def tta_transforms(image):
    aug_images = [image, image.transpose(Image.FLIP_LEFT_RIGHT), image.transpose(Image.FLIP_TOP_BOTTOM)]
    return [base_transform(img).unsqueeze(0).to(device) for img in aug_images]

def load_image(image):
    image = image.convert('RGB')
    return base_transform(image).unsqueeze(0).to(device)

# --- Prediction function with temperature scaling & TTA ---
TEMPERATURE = 2.0  # adjust this (higher = softer probabilities)

def predict(image):
    tta_images = tta_transforms(image)
    outputs = []

    with torch.no_grad():
        for img in tta_images:
            logits = model(img)
            probs = F.softmax(logits / TEMPERATURE, dim=1)
            outputs.append(probs)
    
    avg_probs = torch.mean(torch.cat(outputs, dim=0), dim=0)
    pred_class = torch.argmax(avg_probs).item()

    classes = ['Normal', 'Tuberculosis']
    return classes[pred_class], avg_probs[pred_class].item()

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
