# Tuberculosis Chest X-ray Detector

## Project Overview
This project presents a machine learning system that detects Tuberculosis (TB) from chest X-ray images using a fine-tuned DenseNet-121 convolutional neural network. The model classifies chest X-rays as **Normal** or **Tuberculosis**, providing **high accuracy and interpretability**. A Streamlit web app is included for real-time predictions on single X-ray images, allowing fast and practical use for medical purposes.

---

## Features
- **Automatic TB Detection:** Classifies X-ray images into Normal or Tuberculosis.
- **High Accuracy:** Fine-tuned DenseNet-121 achieves strong performance on the test set.
- **Evaluation Metrics:** Tracks accuracy, precision, recall, F1-score, and confusion matrix.
- **Confidence Scores:** Provides probability scores for each class.
- **Streamlit Deployment:** Upload a chest X-ray image and get an instant prediction.
- **User-Friendly Interface:** Clear UI suitable for non-technical users like medical staff.

---

## Dataset
- **Source:** Tuberculosis (TB) Chest X-ray Database  ["https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset"]
- **Split:** 70% training, 15% validation, 15% testing  
- **Preprocessing:** Resize to 224x224, normalize, convert to tensors

---

## Model Architecture
- **Base Model:** DenseNet-121 (pretrained on ImageNet)  
- **Classifier:** Fully connected layer with 2 outputs (Normal, TB)  
- **Trainable Layers:** Only the final classifier layer is fine-tuned for speed; optionally, last dense block can be unfrozen for better accuracy.

---

## Training
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam  
- **Batch Size:** 32  
- **Epochs:** 5 (adjustable)  
- **Device:** CPU (can be GPU if available)  
- **Metrics Tracked:** Training and validation loss, accuracy

**Training Loop:**
1. Forward pass through the model  
2. Compute loss  
3. Backpropagation  
4. Update weights  
5. Track metrics (loss & accuracy)

**Validation Loop:**
- Forward pass only  
- Compute loss & accuracy  
- Track metrics for evaluation

---

## Evaluation
- **Test Set Metrics:** Accuracy, precision, recall, F1-score, and confusion matrix  


## Deployment
- **Streamlit App:** `tb_app.py`  
- **Functionality:** Upload a single X-ray image to get:  
  - Predicted class (Normal / TB)  
  - Confidence score  
