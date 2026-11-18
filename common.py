"""Shared PyTorch model loading and image inference helpers."""
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def load_classifier(artifact_dir="artifacts"):
    model_path = Path(artifact_dir) / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Chưa có {model_path}. Hãy chạy `python train.py` trước.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    class_names = checkpoint["class_names"]
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    metadata = {
        "class_names": class_names,
        "species_by_breed": checkpoint["species_by_breed"],
        "image_size": checkpoint.get("image_size", 224),
    }
    return model, metadata, device, EfficientNet_B0_Weights.DEFAULT.transforms()


def predict_top_k(model, metadata, device, transform, image: Image.Image, top_k=3):
    image = ImageOps.exif_transpose(image).convert("RGB")
    batch = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch), dim=1)[0]
    values, indices = torch.topk(probabilities, k=max(1, min(top_k, len(metadata["class_names"]))))
    results = []
    for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        breed = metadata["class_names"][index]
        results.append({
            "breed": breed,
            "species": metadata["species_by_breed"][breed],
            "confidence": value,
        })
    return results
