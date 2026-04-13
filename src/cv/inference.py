import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import torch.nn.functional as F
# Recreate your model class
class CatOrDogModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Feature extractor ----
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # 🔥 replaces huge flatten (KEY FIX)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---- Classifier (tiny now) ----
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 224 → 112
        x = self.pool(F.relu(self.conv2(x)))   # 112 → 56
        x = self.pool(F.relu(self.conv3(x)))   # 56 → 28

        # 🔥 compress spatial dimensions safely
        x = self.global_pool(x)                # 28x28 → 1x1

        x = torch.flatten(x, 1)                # (B, 64)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
def model_fn(model_dir):
    model = CatOrDogModel()

    model_path = os.path.join(model_dir, "model.pth")

    print("Loading model from:", model_path)

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    print("Model ready")

    model.eval()
    return model

# Load model
# def model_fn(model_dir):
#     model = CatOrDogModel()
#     model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
#     model.eval()
#     return model


# Input processing
def input_fn(request_body, content_type):
    if content_type == "application/x-image":
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        return transform(image).unsqueeze(0)
    
    raise ValueError("Unsupported content type")


# Prediction
def predict_fn(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


# Output formatting
def output_fn(prediction, accept):
    return str(prediction)
