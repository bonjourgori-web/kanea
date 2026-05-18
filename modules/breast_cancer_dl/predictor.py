from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_PATH = Path("models/deep_learning/breast_cancer_model.pth")
CLASSES = ["Normal", "Benign", "Malignant"]


def _placeholder_response(image_name: str | None) -> dict[str, Any]:
    return {
        "module": "module_4_breast_cancer",
        "task": "breast_cancer_detection_from_mammography",
        "model_family": "deep_learning",
        "model_name": "efficientnet_b0_placeholder",
        "library": "pytorch",
        "expected_model_path": str(MODEL_PATH).replace("\\", "/"),
        "input_image": image_name,
        "prediction": None,
        "confidence": None,
        "supported_output_classes": CLASSES,
        "explainability": {"method": "Grad-CAM", "status": "not_generated"},
        "deployment_mode": "offline_cpu_ready",
        "status": "scaffold_ready",
    }


def predict_breast_cancer(image_path: str | None = None) -> dict[str, Any]:
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
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, len(CLASSES))
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        image = Image.open(image_path).convert("L")
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())

        return {
            "module": "module_4_breast_cancer",
            "task": "breast_cancer_detection_from_mammography",
            "model_family": "deep_learning",
            "model_name": "efficientnet_b0_production",
            "library": "pytorch",
            "expected_model_path": str(MODEL_PATH).replace("\\", "/"),
            "input_image": image_name,
            "prediction": CLASSES[pred_idx],
            "confidence": float(probs[pred_idx].item()),
            "supported_output_classes": CLASSES,
            "explainability": {"method": "Grad-CAM", "status": "pending_overlay_generation"},
            "deployment_mode": "offline_cpu_ready",
            "status": "model_loaded",
        }
    except Exception as exc:
        response = _placeholder_response(image_name)
        response["status"] = "inference_error"
        response["error"] = str(exc)
        return response
