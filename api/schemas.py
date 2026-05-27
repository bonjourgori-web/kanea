from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class NutritionInput(BaseModel):
    age_months: Optional[int]   = Field(default=None, ge=0)
    weight_kg:  Optional[float] = Field(default=None, gt=0)
    height_cm:  Optional[float] = Field(default=None, gt=0)
    sex:        Optional[str]   = Field(default=None, description="M or F")
    muac_cm:    Optional[float] = Field(default=None, gt=0)
    waz:        Optional[float] = None
    haz:        Optional[float] = None
    whz:        Optional[float] = None
    waist_hip_ratio: Optional[float] = Field(default=None, gt=0)


class BioIDInput(BaseModel):
    # Identifiants du dossier (optionnels pour la compatibilité ascendante)
    case_id:  Optional[str] = Field(default=None, description="Identifiant du dossier médico-légal")
    examiner: Optional[str] = Field(default=None, description="Nom de l'examinateur")

    # Mesures morphologiques
    cranial_measurements:     dict[str, Any] = Field(default_factory=dict)
    postcranial_measurements: dict[str, Any] = Field(default_factory=dict)

    # Marqueurs ancestraux — deux modes acceptés :
    #   • aims_pcs  : composantes PCA déjà calculées (AIM_PC1/PC2/PC3)
    #   • aims_raw  : marqueurs bruts AIM_raw_1..5 (PCA appliquée côté serveur)
    aims_pcs: dict[str, Any]      = Field(default_factory=dict)
    aims_raw: Optional[list[float]] = Field(
        default=None,
        description="Marqueurs AIMs bruts [AIM_raw_1..5] — PCA appliquée automatiquement",
    )

    # Options
    generate_pdf: bool = Field(default=False, description="Générer un rapport PDF après prédiction")


class MalariaPathInput(BaseModel):
    image_path: str = Field(..., description="Chemin local vers une image JPEG/PNG")


class BreastCancerPathInput(BaseModel):
    image_path: str = Field(..., description="Chemin local vers une mammographie JPEG/PNG")


class MultibioRequest(BaseModel):
    image_path:              Optional[str]           = None
    breast_cancer_image_path: Optional[str]          = None
    nutrition_data:          Optional[NutritionInput] = None
    bioid_data:              Optional[BioIDInput]     = None
