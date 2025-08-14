import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# --- Device ---
device = torch.device("cpu")  # or "cuda"

# --- Load your model ---
model = models.densenet121(pretrained=False)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 2)  # 2 classes: Normal, TB
model.load_state_dict(torch.load("tb_detector_densenet121.pth", map_location=device))
model.to(device)
model.eval()

# --- Load the training/validation data for calibration ---
# Replace with the same transforms you used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Replace "path_to_data" with your folder containing Normal/TB subfolders
calib_dataset = datasets.ImageFolder("path_to_data", transform=transform)
calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=True)

# --- Temperature scaling class ---
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # start >1 to reduce overconfidence

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

# Wrap the model
model_temp = ModelWithTemperature(model).to(device)
model_temp.eval()

# --- Optimize temperature ---
optimizer = torch.optim.LBFGS([model_temp.temperature], lr=0.01, max_iter=50)

nll_criterion = nn.CrossEntropyLoss()

def eval():
    optimizer.zero_grad()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in calib_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model_temp(images)
            logits_list.append(logits)
            labels_list.append(labels)

    logits_all = torch.cat(logits_list)
    labels_all = torch.cat(labels_list)
    loss = nll_criterion(logits_all, labels_all)
    loss.backward()
    return loss

optimizer.step(eval)

print("Optimal temperature:", model_temp.temperature.item())

# --- Save calibrated model ---
torch.save({
    'model_state_dict': model.state_dict(),
    'temperature': model_temp.temperature.item()
}, "tb_detector_calibrated.pth")
