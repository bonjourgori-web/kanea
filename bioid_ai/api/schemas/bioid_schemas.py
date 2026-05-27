"""
bioid_schemas.py — Schémas Pydantic v2 pour l'API BioID AI.

Définit tous les modèles de données pour les requêtes et réponses
de l'API REST BioID.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Mesures crâniennes (15 variables FORDISC) ─────────────────────────────────
class CranialMeasurements(BaseModel):
    """15 mesures crâniennes standard FORDISC (toutes en mm)."""
    GOL: Optional[float] = Field(None, description="Glabello-occipital length (mm)", ge=100, le=250)
    XCB: Optional[float] = Field(None, description="Maximum cranial breadth (mm)", ge=80, le=200)
    BBH: Optional[float] = Field(None, description="Basion-bregma height (mm)", ge=80, le=200)
    ZYB: Optional[float] = Field(None, description="Bizygomatic breadth (mm)", ge=80, le=180)
    AUB: Optional[float] = Field(None, description="Biauricular breadth (mm)", ge=70, le=170)
    ASB: Optional[float] = Field(None, description="Biasterionic breadth (mm)", ge=70, le=160)
    BNL: Optional[float] = Field(None, description="Basion-nasion length (mm)", ge=60, le=150)
    BPL: Optional[float] = Field(None, description="Basion-prosthion length (mm)", ge=60, le=150)
    NLH: Optional[float] = Field(None, description="Nasal height (mm)", ge=30, le=80)
    NLB: Optional[float] = Field(None, description="Nasal breadth (mm)", ge=15, le=50)
    OBH: Optional[float] = Field(None, description="Orbital height (mm)", ge=20, le=55)
    OBB: Optional[float] = Field(None, description="Orbital breadth (mm)", ge=25, le=60)
    MAB: Optional[float] = Field(None, description="Maxillary breadth (mm)", ge=40, le=90)
    FOL: Optional[float] = Field(None, description="Foramen magnum length (mm)", ge=20, le=60)
    FOB: Optional[float] = Field(None, description="Foramen magnum breadth (mm)", ge=15, le=50)

    model_config = {"extra": "ignore"}

    def to_dict(self) -> dict[str, Optional[float]]:
        return self.model_dump(exclude_none=False)


# ── Mesures post-crâniennes (8 variables) ────────────────────────────────────
class PostcranialMeasurements(BaseModel):
    """8 mesures post-crâniennes (toutes en mm)."""
    femur_max_length:   Optional[float] = Field(None, description="Longueur max fémur (mm)", ge=250, le=600)
    femur_bicondylar:   Optional[float] = Field(None, description="Longueur bicondylaire fémur (mm)", ge=250, le=600)
    tibia_length:       Optional[float] = Field(None, description="Longueur tibia (mm)", ge=200, le=500)
    humerus_max_length: Optional[float] = Field(None, description="Longueur max humérus (mm)", ge=200, le=450)
    radius_max_length:  Optional[float] = Field(None, description="Longueur max radius (mm)", ge=150, le=350)
    fibula_max_length:  Optional[float] = Field(None, description="Longueur max fibula (mm)", ge=200, le=490)
    femur_head_diam:    Optional[float] = Field(None, description="Diamètre tête fémur (mm)", ge=25, le=65)
    humerus_head_diam:  Optional[float] = Field(None, description="Diamètre tête humérus (mm)", ge=25, le=65)

    model_config = {"extra": "ignore"}

    def to_dict(self) -> dict[str, Optional[float]]:
        return self.model_dump(exclude_none=False)


# ── Marqueurs ancestraux bruts (AIMs) ────────────────────────────────────────
class AIMarkers(BaseModel):
    """5 marqueurs ancestraux bruts (Ancestry Informative Markers)."""
    AIM_raw_1: Optional[float] = Field(None, description="Marqueur AIM brut 1")
    AIM_raw_2: Optional[float] = Field(None, description="Marqueur AIM brut 2")
    AIM_raw_3: Optional[float] = Field(None, description="Marqueur AIM brut 3")
    AIM_raw_4: Optional[float] = Field(None, description="Marqueur AIM brut 4")
    AIM_raw_5: Optional[float] = Field(None, description="Marqueur AIM brut 5")

    model_config = {"extra": "ignore"}

    def to_list(self) -> list[Optional[float]]:
        return [self.AIM_raw_1, self.AIM_raw_2, self.AIM_raw_3, self.AIM_raw_4, self.AIM_raw_5]


# ── Requête principale BioID ──────────────────────────────────────────────────
class BioidRequest(BaseModel):
    """Requête complète pour l'analyse BioID AI."""
    case_id:    str = Field(..., description="Identifiant unique du dossier", min_length=1, max_length=100)
    examiner:   str = Field(..., description="Nom de l'examinateur", min_length=1, max_length=100)
    cranial:    Optional[CranialMeasurements]    = Field(None, description="Mesures crâniennes")
    postcranial: Optional[PostcranialMeasurements] = Field(None, description="Mesures post-crâniennes")
    aims:       Optional[AIMarkers]              = Field(None, description="Marqueurs ancestraux bruts")
    notes:      Optional[str]                    = Field(None, description="Notes de l'examinateur", max_length=2000)

    model_config = {"extra": "ignore"}

    @field_validator("case_id", "examiner", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return str(v).strip()

    @model_validator(mode="after")
    def check_at_least_one_measurement(self) -> "BioidRequest":
        has_cranial = self.cranial is not None and any(
            v is not None for v in self.cranial.model_dump().values()
        )
        has_postcranial = self.postcranial is not None and any(
            v is not None for v in self.postcranial.model_dump().values()
        )
        if not has_cranial and not has_postcranial:
            raise ValueError(
                "Au moins une mesure crâniale ou post-crâniale doit être fournie."
            )
        return self


# ── Résultat de prédiction ────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    """Résultats des prédictions du profil biologique."""
    biological_sex:       str   = Field(..., description="Sexe biologique estimé")
    sex_confidence:       float = Field(..., description="Probabilité estimée pour le sexe", ge=0, le=1)
    age_at_death:         float = Field(..., description="Âge au décès estimé (années)", ge=0, le=120)
    age_range:            str   = Field(..., description="Intervalle d'âge [min–max ans]")
    ancestry:             str   = Field(..., description="Ascendance biogéographique estimée")
    ancestry_confidence:  float = Field(..., description="Probabilité estimée pour l'ascendance", ge=0, le=1)
    stature_cm:           Optional[float] = Field(None, description="Stature estimée (cm)")
    stature_method:       Optional[str]   = Field(None, description="Méthode d'estimation de la stature")

    model_config = {"extra": "ignore"}


# ── Réponse complète BioID ────────────────────────────────────────────────────
class BioidResponse(BaseModel):
    """Réponse complète de l'analyse BioID AI."""
    case_id:           str                        = Field(..., description="Identifiant du dossier")
    timestamp:         datetime                   = Field(..., description="Horodatage de l'analyse")
    predictions:       PredictionResult           = Field(..., description="Résultats des prédictions")
    feature_importances: Optional[dict[str, float]] = Field(None, description="Importance des features")
    pca_variance:      Optional[list[float]]      = Field(None, description="Variance expliquée par composante PCA")
    report_path:       Optional[str]              = Field(None, description="Chemin vers le rapport PDF généré")
    status:            str                        = Field(..., description="Statut de l'analyse")
    warnings:          Optional[list[str]]        = Field(None, description="Avertissements éventuels")
    model_version:     Optional[str]              = Field(None, description="Version du modèle utilisé")

    model_config = {"extra": "ignore"}


# ── Requête de génération de rapport ─────────────────────────────────────────
class ReportRequest(BaseModel):
    """Requête pour générer un rapport PDF."""
    case_id:   str                       = Field(..., description="Identifiant du dossier", min_length=1)
    examiner:  str                       = Field(..., description="Nom de l'examinateur", min_length=1)
    predictions: dict[str, Any]          = Field(..., description="Données de prédictions à inclure")
    include_charts: bool                 = Field(True, description="Inclure les graphiques dans le rapport")
    chart_paths: Optional[list[str]]     = Field(None, description="Chemins des graphiques à inclure")
    notes:     Optional[str]             = Field(None, description="Notes additionnelles")

    model_config = {"extra": "ignore"}


# ── Health check ─────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Réponse du health check."""
    status:        str     = Field(..., description="Statut du service")
    version:       str     = Field(..., description="Version de l'API")
    models_loaded: bool    = Field(..., description="Modèles chargés en mémoire")
    timestamp:     datetime = Field(..., description="Horodatage")


# ── Info modèles ──────────────────────────────────────────────────────────────
class ModelInfo(BaseModel):
    """Informations sur les modèles chargés."""
    bundle_path:        str              = Field(..., description="Chemin du bundle")
    models_available:   list[str]        = Field(..., description="Modèles disponibles")
    feature_columns:    list[str]        = Field(..., description="Colonnes de features")
    sex_classes:        Optional[list[str]] = Field(None, description="Classes sexe")
    ancestry_classes:   Optional[list[str]] = Field(None, description="Classes ascendance")
    trained_at:         Optional[str]    = Field(None, description="Date d'entraînement")
    n_samples:          Optional[int]    = Field(None, description="Nombre d'échantillons d'entraînement")
    version:            Optional[str]    = Field(None, description="Version du modèle")
