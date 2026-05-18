from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, UploadFile

from api.schemas import BioIDInput, BreastCancerPathInput, MalariaPathInput, MultibioRequest, NutritionInput
from modules.bioid_ml.predictor import predict_bioid
from modules.breast_cancer_dl.predictor import predict_breast_cancer
from modules.malaria_dl.predictor import predict_malaria
from modules.integration.engine import multibio_predict
from modules.nutrition_ml.predictor import predict_nutrition

app = FastAPI(
    title="KANEA API",
    description="API d'aide a la decision biomedicale et medico-legale pour KANEA.",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "application": "KANEA",
        "status": "running",
        "message": "Knowledge Anthropology & Neural Engine for Africa",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    return {
        "application": "KANEA",
        "full_name": "Knowledge Anthropology & Neural Engine for Africa",
        "modules": [
            {
                "name": "module_1_malaria",
                "approach": "deep_learning",
                "technology": "fast.ai + PyTorch + ResNet34",
            },
            {
                "name": "module_2_biometry",
                "approach": "machine_learning",
                "technology": "RandomForest + XGBoost",
            },
            {
                "name": "module_3_forensic",
                "approach": "machine_learning",
                "technology": "scikit-learn + PCA + regressions",
            },
            {
                "name": "module_4_breast_cancer",
                "approach": "deep_learning",
                "technology": "PyTorch + EfficientNet / ResNet",
            },
        ],
        "integration": "multibio_predict",
    }


@app.post("/predict/malaria")
def predict_malaria_from_path(request: MalariaPathInput) -> dict:
    return predict_malaria(request.image_path)


@app.post("/predict/malaria/upload")
async def predict_malaria_from_upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    return predict_malaria(temp_path)


@app.post("/predict/breast-cancer")
def predict_breast_cancer_from_path(request: BreastCancerPathInput) -> dict:
    return predict_breast_cancer(request.image_path)


@app.post("/predict/breast-cancer/upload")
async def predict_breast_cancer_from_upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    return predict_breast_cancer(temp_path)


@app.post("/predict/biometry")
def predict_biometry(request: NutritionInput) -> dict:
    return predict_nutrition(request.model_dump())


@app.post("/predict/forensic")
def predict_forensic(request: BioIDInput) -> dict:
    return predict_bioid(request.model_dump())


@app.post("/predict")
def predict(request: MultibioRequest) -> dict:
    nutrition_data = (
        request.nutrition_data.model_dump() if request.nutrition_data else None
    )
    bioid_data = request.bioid_data.model_dump() if request.bioid_data else None

    return multibio_predict(
        image_path=request.image_path,
        breast_cancer_image_path=request.breast_cancer_image_path,
        nutrition_data=nutrition_data,
        bioid_data=bioid_data,
    )
