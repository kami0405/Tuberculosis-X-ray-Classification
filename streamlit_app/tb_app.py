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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # same as training
                         [0.229, 0.224, 0.225])
])

def load_image(image):
    image = image.convert('RGB')  # convert grayscale to RGB if needed
    image = transform(image).unsqueeze(0)  # add batch dimension
    return image.to(device)

# --- Prediction function with threshold and class comparison ---
def predict(image, threshold=0.7, temperature=1.0):
    image = load_image(image)
    with torch.no_grad():
        outputs = model(image)
        # optional temperature scaling
        probs = torch.softmax(outputs / temperature, dim=1)
        prob_normal, prob_TB = probs[0][0].item(), probs[0][1].item()

        if prob_TB > prob_normal and prob_TB >= threshold:
            pred_class = "Tuberculosis"
            confidence = prob_TB
        else:
            pred_class = "Normal"
            confidence = prob_normal

    return pred_class, confidence

# --- Streamlit UI ---
st.title("💻 TB Chest X-ray Detector")
st.write("Upload a chest X-ray image and get a prediction for Tuberculosis.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    pred_class, confidence = predict(image, threshold=0.7, temperature=1.0)
    st.write(f"**Prediction:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.4f}")
