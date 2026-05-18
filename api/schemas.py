from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class NutritionInput(BaseModel):
    age_months: Optional[int] = Field(default=None, ge=0)
    weight_kg: Optional[float] = Field(default=None, gt=0)
    height_cm: Optional[float] = Field(default=None, gt=0)
    sex: Optional[str] = Field(default=None, description="M or F")
    muac_cm: Optional[float] = Field(default=None, gt=0)
    waz: Optional[float] = None
    haz: Optional[float] = None
    whz: Optional[float] = None
    waist_hip_ratio: Optional[float] = Field(default=None, gt=0)


class BioIDInput(BaseModel):
    cranial_measurements: dict[str, Any] = Field(default_factory=dict)
    postcranial_measurements: dict[str, Any] = Field(default_factory=dict)
    aims_pcs: dict[str, Any] = Field(default_factory=dict)


class MalariaPathInput(BaseModel):
    image_path: str = Field(..., description="Chemin local vers une image JPEG/PNG")


class BreastCancerPathInput(BaseModel):
    image_path: str = Field(..., description="Chemin local vers une mammographie JPEG/PNG")


class MultibioRequest(BaseModel):
    image_path: Optional[str] = None
    breast_cancer_image_path: Optional[str] = None
    nutrition_data: Optional[NutritionInput] = None
    bioid_data: Optional[BioIDInput] = None
