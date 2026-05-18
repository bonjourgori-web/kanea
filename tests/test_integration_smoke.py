from modules.integration.engine import multibio_predict
from modules.bioid_ml.predictor import predict_bioid
from modules.breast_cancer_dl.predictor import predict_breast_cancer
from modules.malaria_dl.predictor import predict_malaria


def test_multibio_predict_without_inputs() -> None:
    result = multibio_predict()
    assert result["application"] == "KANEA"
    assert result["status"] == "no_input_provided"


def test_multibio_predict_with_nutrition() -> None:
    result = multibio_predict(
        nutrition_data={
            "age_months": 24,
            "weight_kg": 12.0,
            "height_cm": 85.0,
            "sex": "F",
            "waz": -1.1,
            "haz": -0.8,
            "whz": -0.3,
        }
    )
    assert "module_2_biometry" in result["results"]


def test_predict_malaria_without_model_returns_scaffold() -> None:
    result = predict_malaria("sample.png")
    assert result["module"] == "module_1_malaria"
    assert result["expected_model_path"] == "models/deep_learning/malaria_model.pkl"


def test_predict_bioid_without_model_returns_scaffold() -> None:
    result = predict_bioid(
        {
            "cranial_measurements": {"bizygomatic_breadth_mm": 128},
            "postcranial_measurements": {"femur_length_cm": 44.5},
            "aims_pcs": {"AIM_PC1": 0.1},
        }
    )
    assert result["module"] == "module_3_forensic"
    assert result["expected_model_path"] == "models/machine_learning/forensic_model.pkl"


def test_multibio_predict_with_forensic_only() -> None:
    result = multibio_predict(
        bioid_data={
            "cranial_measurements": {"bizygomatic_breadth_mm": 128},
            "postcranial_measurements": {"femur_length_cm": 44.5},
            "aims_pcs": {"AIM_PC1": 0.1},
        }
    )
    assert "module_3_forensic" in result["results"]


def test_predict_breast_cancer_without_model_returns_scaffold() -> None:
    result = predict_breast_cancer("mammography.png")
    assert result["module"] == "module_4_breast_cancer"
    assert result["expected_model_path"] == "models/deep_learning/breast_cancer_model.pth"
