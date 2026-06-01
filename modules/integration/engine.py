from __future__ import annotations

from typing import Any

from modules.bioid_ml.predictor import predict_bioid
from modules.breast_cancer_dl.predictor import predict_breast_cancer
from modules.malaria_dl.predictor import predict_malaria
from modules.nutrition_ml.predictor import predict_nutrition
from modules.hemato_ai.predictor import predict_hemato


def multibio_predict(
    image_path: str | None = None,
    breast_cancer_image_path: str | None = None,
    nutrition_data: dict[str, Any] | None = None,
    bioid_data: dict[str, Any] | None = None,
    hemato_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fonction centrale d'orchestration conforme au Module 4."""

    results: dict[str, Any] = {
        "application": "KANEA",
        "full_name": "Knowledge Anthropology & Neural Engine for Africa",
        "mode": "multimodal_decision_support",
        "integration_function": "multibio_predict",
        "results": {},
    }

    if image_path:
        results["results"]["module_1_malaria"] = predict_malaria(image_path=image_path)

    if nutrition_data:
        results["results"]["module_2_biometry"] = predict_nutrition(nutrition_data)

    if bioid_data:
        results["results"]["module_3_forensic"] = predict_bioid(bioid_data)

    if breast_cancer_image_path:
        results["results"]["module_4_breast_cancer"] = predict_breast_cancer(
            image_path=breast_cancer_image_path
        )

    if hemato_data:
        results["results"]["module_5_hemato"] = predict_hemato(params=hemato_data)

    results["available_modules"] = list(results["results"].keys())
    results["status"] = "ok" if results["results"] else "no_input_provided"
    return results
