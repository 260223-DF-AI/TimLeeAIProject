import os
import io
import json
import torch
from torchvision import transforms
from PIL import Image
from model import VGG19

def model_fn(model_dir):
    print("Loading model...")
    model = VGG19()
    model_path = os.path.join(model_dir, "model.pth")
    print("Loading model from:", model_path)
    # state_dict = torch.load(model_path, map_location="cpu")
    # model.load_state_dict(state_dict)

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    model.eval()
    print("Model ready")
    return model

def input_fn(request_body, content_type):
    if content_type == "application/x-image":
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    with torch.no_grad():
        outputs = model(input_data)
    return outputs

def output_fn(prediction, accept):
    probs = torch.softmax(prediction, dim=1)
    confidence, pred_class = torch.max(probs, dim=1)
    return json.dumps({
        "class": int(pred_class.item()),
        "confidence": float(confidence.item()),
        "probabilities": probs.squeeze().tolist()
    })