from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_PATH = Path("models/deep_learning/malaria_model.pth")
CLASSES    = ["Parasitised", "Uninfected"]
IMG_SIZE   = 224


def _placeholder_response(image_name: str | None) -> dict[str, Any]:
    return {
        "module":                 "module_1_malaria",
        "task":                   "automatic_malaria_detection",
        "model_family":           "deep_learning",
        "model_name":             "resnet34_malaria_placeholder",
        "library":                "pytorch",
        "expected_model_path":    str(MODEL_PATH).replace("\\", "/"),
        "input_image":            image_name,
        "prediction":             None,
        "confidence":             None,
        "supported_output_classes": CLASSES,
        "explainability":         {"method": "Grad-CAM", "status": "not_generated"},
        "deployment_mode":        "offline_cpu_ready",
        "status":                 "scaffold_ready",
    }


def predict_malaria(image_path: str | None = None) -> dict[str, Any]:
    """Inférence malaria — ResNet34 PyTorch.

    Entraîner le modèle d'abord :
        python scripts/train_malaria_pytorch.py
    """
    image_name = Path(image_path).name if image_path else None
    if not image_path:
        return _placeholder_response(image_name)

    model_path = Path.cwd() / MODEL_PATH
    if not model_path.exists():
        return _placeholder_response(image_name)

    try:
        import torch
        from PIL import Image
        from torchvision import models, transforms
    except ImportError:
        response = _placeholder_response(image_name)
        response["status"] = "torchvision_not_installed"
        return response

    try:
        # ── Chargement modèle ──────────────────────────────────────────────────
        model = models.resnet34(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features, len(CLASSES)),
        )
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        # ── Prétraitement image ────────────────────────────────────────────────
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        image  = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        # ── Inférence ─────────────────────────────────────────────────────────
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())

        return {
            "module":                 "module_1_malaria",
            "task":                   "automatic_malaria_detection",
            "model_family":           "deep_learning",
            "model_name":             "resnet34_malaria_production",
            "library":                "pytorch",
            "expected_model_path":    str(MODEL_PATH).replace("\\", "/"),
            "input_image":            image_name,
            "prediction":             CLASSES[pred_idx],
            "confidence":             round(float(probs[pred_idx].item()), 4),
            "probabilities": {
                cls: round(float(probs[i].item()), 4)
                for i, cls in enumerate(CLASSES)
            },
            "supported_output_classes": CLASSES,
            "explainability":         {"method": "Grad-CAM", "status": "pending_overlay_generation"},
            "deployment_mode":        "offline_cpu_ready",
            "status":                 "model_loaded",
        }

    except Exception as exc:
        response = _placeholder_response(image_name)
        response["status"] = "inference_error"
        response["error"]  = str(exc)
        return response
