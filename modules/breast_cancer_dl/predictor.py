from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

MODEL_PATH = Path("models/deep_learning/breast_cancer_model.pth")
CLASSES    = ["Normal", "Benign", "Malignant"]
IMG_SIZE   = 224

RISK_COLORS = {
    "Normal":    "#27AE60",
    "Benign":    "#F39C12",
    "Malignant": "#E74C3C",
}


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


def _generate_gradcam_efficientnet(model, tensor, pred_idx) -> str | None:
    """Grad-CAM pour EfficientNet-B0 — cible model.features[-1]."""
    try:
        import numpy as np
        import torch
        from PIL import Image

        gradients: list = []
        activations: list = []

        def save_grad(grad):
            gradients.append(grad)

        def forward_hook(module, input, output):
            activations.append(output)
            output.register_hook(save_grad)

        # Dernière couche features d'EfficientNet
        target_layer = model.features[-1]
        hook = target_layer.register_forward_hook(forward_hook)

        model.eval()
        logits = model(tensor)
        model.zero_grad()
        logits[0, pred_idx].backward()
        hook.remove()

        if not gradients or not activations:
            return None

        grad = gradients[0].squeeze(0)
        act  = activations[0].squeeze(0)
        weights = grad.mean(dim=(1, 2))

        cam = torch.zeros(act.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * act[i]

        cam = torch.clamp(cam, min=0)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        cam_np  = cam.detach().numpy()
        cam_img = Image.fromarray((cam_np * 255).astype("uint8")).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )

        cam_arr = np.array(cam_img, dtype=np.float32) / 255.0
        r = np.clip(2.0 * cam_arr, 0, 1)
        g = np.clip(2.0 * (1.0 - cam_arr), 0, 1)
        b = np.zeros_like(cam_arr)
        heatmap = (np.stack([r, g, b], axis=2) * 255).astype("uint8")

        buf = io.BytesIO()
        Image.fromarray(heatmap).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception:
        return None


def predict_breast_cancer(image_path: str | None = None) -> dict[str, Any]:
    """Inférence cancer du sein — EfficientNet-B0 PyTorch + Grad-CAM."""
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

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        image  = Image.open(image_path).convert("L")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits   = model(tensor)
            probs    = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())

        # Grad-CAM (nécessite un tensor avec grad)
        tensor_grad = transform(image).unsqueeze(0).requires_grad_(True)
        gradcam_b64 = _generate_gradcam_efficientnet(model, tensor_grad, pred_idx)
        gradcam_status = "generated" if gradcam_b64 else "generation_failed"

        return {
            "module": "module_4_breast_cancer",
            "task": "breast_cancer_detection_from_mammography",
            "model_family": "deep_learning",
            "model_name": "efficientnet_b0_production",
            "library": "pytorch",
            "expected_model_path": str(MODEL_PATH).replace("\\", "/"),
            "input_image": image_name,
            "prediction": CLASSES[pred_idx],
            "confidence": round(float(probs[pred_idx].item()), 4),
            "probabilities": {
                cls: round(float(probs[i].item()), 4)
                for i, cls in enumerate(CLASSES)
            },
            "risk_level": CLASSES[pred_idx],
            "supported_output_classes": CLASSES,
            "explainability": {
                "method":      "Grad-CAM",
                "status":      gradcam_status,
                "heatmap_b64": gradcam_b64,
                "target_layer": "features[-1]",
            },
            "deployment_mode": "offline_cpu_ready",
            "status": "model_loaded",
        }

    except Exception as exc:
        response = _placeholder_response(image_name)
        response["status"] = "inference_error"
        response["error"]  = str(exc)
        return response
