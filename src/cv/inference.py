import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import torch.nn.functional as F
from model import CatOrDogModel
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
    return outputs()


# Output formatting
def output_fn(prediction, accept):
    return str(prediction)
