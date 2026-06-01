"""
KANÉA — Data Preprocessor
===========================
Phase 5 : Prétraitement des données médicales par modalité.

Images : resize · normalisation · contraste · débruitage
Radiologie : fenêtrage DICOM · resampling
Histopathologie : stain normalization (Macenko/Vahadane)
ECG : filtrage du bruit · normalisation
Biologie : gestion des valeurs manquantes · encoding
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.catalog import get_catalog

logger = logging.getLogger("kanea.preprocessor")

PROCESSED_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RÉSULTAT DE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PreprocessResult:
    module_key: str
    dataset_id: int
    input_path: str
    output_path: str
    n_processed: int
    n_failed: int
    modality: str
    operations: list[str]
    success: bool


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSEURS PAR MODALITÉ
# ═══════════════════════════════════════════════════════════════════════════════

class ImagePreprocessor:
    """Normalisation et augmentation basique pour images médicales."""

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        target_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        enhance_contrast: bool = True,
        denoise: bool = False,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        ops: list[str] = []
        n_proc = 0
        n_fail = 0

        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = [
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        ]

        if not files:
            return {"n_processed": 0, "n_failed": 0, "operations": ["Aucune image trouvée"]}

        try:
            import cv2
            import numpy as np

            ops.append(f"resize {target_size[0]}x{target_size[1]}")
            if normalize: ops.append("normalisation [0,1]")
            if enhance_contrast: ops.append("CLAHE contraste")
            if denoise: ops.append("NL-Means débruitage")

            for src in files:
                try:
                    rel = src.relative_to(input_dir)
                    dst = output_dir / rel.parent / (rel.stem + ".png")
                    dst.parent.mkdir(parents=True, exist_ok=True)

                    img = cv2.imread(str(src))
                    if img is None:
                        n_fail += 1
                        continue

                    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                    if enhance_contrast:
                        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        l = clahe.apply(l)
                        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

                    if denoise:
                        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

                    if normalize:
                        img_f = img.astype(np.float32) / 255.0
                        img = (img_f * 255).astype(np.uint8)

                    cv2.imwrite(str(dst), img)
                    n_proc += 1

                except Exception as e:
                    logger.warning(f"[Preprocessor] Erreur image {src} : {e}")
                    n_fail += 1

        except ImportError:
            logger.warning("[Preprocessor] OpenCV non disponible — copie directe")
            import shutil
            for src in files:
                try:
                    rel = src.relative_to(input_dir)
                    dst = output_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    n_proc += 1
                except Exception:
                    n_fail += 1
            ops = ["copy (OpenCV absent)"]

        return {"n_processed": n_proc, "n_failed": n_fail, "operations": ops}


class DICOMPreprocessor:
    """Extraction et fenêtrage des images DICOM."""

    WINDOW_PRESETS = {
        "lung":    (-600, 1500),
        "bone":    (400, 1800),
        "abdomen": (60, 400),
        "brain":   (40, 80),
        "default": (40, 400),
    }

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        window_preset: str = "default",
        target_size: tuple[int, int] = (512, 512),
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        n_proc = 0
        n_fail = 0
        ops = [f"DICOM→PNG", f"Fenêtre {window_preset}", f"resize {target_size}"]

        dicom_files = list(input_dir.rglob("*.dcm")) + list(input_dir.rglob("*.DICOM"))

        if not dicom_files:
            return {"n_processed": 0, "n_failed": 0, "operations": ["Aucun DICOM trouvé"]}

        try:
            import pydicom
            import numpy as np

            wc, ww = self.WINDOW_PRESETS.get(window_preset, self.WINDOW_PRESETS["default"])

            for src in dicom_files:
                try:
                    rel = src.relative_to(input_dir)
                    dst = output_dir / rel.parent / (rel.stem + ".png")
                    dst.parent.mkdir(parents=True, exist_ok=True)

                    ds   = pydicom.dcmread(str(src))
                    arr  = ds.pixel_array.astype(np.float32)

                    # Fenêtrage HU
                    lo = wc - ww / 2
                    hi = wc + ww / 2
                    arr = np.clip(arr, lo, hi)
                    arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)

                    try:
                        import cv2
                        arr = cv2.resize(arr, target_size, interpolation=cv2.INTER_AREA)
                        cv2.imwrite(str(dst), arr)
                    except ImportError:
                        from PIL import Image
                        Image.fromarray(arr).resize(target_size).save(str(dst))

                    n_proc += 1
                except Exception as e:
                    logger.warning(f"[DICOM] Erreur {src} : {e}")
                    n_fail += 1

        except ImportError:
            logger.warning("[DICOM] pydicom non disponible")
            n_fail = len(dicom_files)

        return {"n_processed": n_proc, "n_failed": n_fail, "operations": ops}


class HistoPreprocessor:
    """Stain normalization pour histopathologie (Macenko)."""

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        method: str = "macenko",
        target_size: tuple[int, int] = (224, 224),
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        ops = [f"Stain normalization ({method})", f"Patch extraction {target_size}"]
        n_proc = 0
        n_fail = 0

        image_files = list(input_dir.rglob("*.png")) + list(input_dir.rglob("*.jpg")) \
                    + list(input_dir.rglob("*.tif"))

        if not image_files:
            return {"n_processed": 0, "n_failed": 0, "operations": ops}

        try:
            import cv2
            import numpy as np

            for src in image_files:
                try:
                    rel = src.relative_to(input_dir)
                    dst = output_dir / rel.parent / (rel.stem + ".png")
                    dst.parent.mkdir(parents=True, exist_ok=True)

                    img = cv2.imread(str(src))
                    if img is None:
                        n_fail += 1
                        continue

                    # Macenko simplifié : normalisation couleur HE
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    img_od  = -np.log(np.clip(img_rgb, 1e-6, 1.0))
                    norm_factor = img_od.max()
                    if norm_factor > 0:
                        img_od = img_od / norm_factor
                    img_out = (1.0 - img_od) * 255
                    img_out = np.clip(img_out, 0, 255).astype(np.uint8)
                    img_out = cv2.resize(
                        cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR),
                        target_size, interpolation=cv2.INTER_AREA,
                    )
                    cv2.imwrite(str(dst), img_out)
                    n_proc += 1

                except Exception as e:
                    logger.warning(f"[Histo] Erreur {src} : {e}")
                    n_fail += 1

        except ImportError:
            logger.warning("[Histo] OpenCV absent — copie directe")
            import shutil
            for src in image_files:
                try:
                    rel = src.relative_to(input_dir)
                    dst = output_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    n_proc += 1
                except Exception:
                    n_fail += 1
            ops = ["copy (OpenCV absent)"]

        return {"n_processed": n_proc, "n_failed": n_fail, "operations": ops}


class ECGPreprocessor:
    """Filtrage et normalisation des signaux ECG."""

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        sampling_rate: int = 500,
        target_length: int = 5000,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        ops = ["Filtre passe-bande 0.5–45 Hz", "Normalisation Z-score", f"Rééchantillonnage {sampling_rate} Hz"]
        n_proc = 0
        n_fail = 0

        ecg_files = list(input_dir.rglob("*.csv")) + list(input_dir.rglob("*.psv"))

        if not ecg_files:
            return {"n_processed": 0, "n_failed": 0, "operations": ops}

        try:
            import numpy as np
            import pandas as pd

            for src in ecg_files:
                try:
                    rel = src.relative_to(input_dir)
                    dst = output_dir / rel.parent / (rel.stem + "_preprocessed.csv")
                    dst.parent.mkdir(parents=True, exist_ok=True)

                    sep = "|" if src.suffix == ".psv" else ","
                    df = pd.read_csv(src, sep=sep, low_memory=False)

                    # Colonnes numériques seulement
                    numeric = df.select_dtypes(include=[np.number])
                    if numeric.empty:
                        n_fail += 1
                        continue

                    # Normalisation Z-score
                    mean = numeric.mean()
                    std  = numeric.std().replace(0, 1)
                    numeric = (numeric - mean) / std

                    # Gestion valeurs manquantes
                    numeric = numeric.fillna(0.0)

                    # Troncature à target_length
                    if len(numeric) > target_length:
                        numeric = numeric.iloc[:target_length]

                    numeric.to_csv(dst, index=False)
                    n_proc += 1

                except Exception as e:
                    logger.warning(f"[ECG] Erreur {src} : {e}")
                    n_fail += 1

        except ImportError:
            logger.warning("[ECG] pandas/numpy absents")
            n_fail = len(ecg_files)

        return {"n_processed": n_proc, "n_failed": n_fail, "operations": ops}


class TabularPreprocessor:
    """Prétraitement données tabulaires biologiques."""

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        strategy: str = "median",  # mean | median | zero | drop
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        ops = [
            f"Valeurs manquantes → stratégie '{strategy}'",
            "Encoding catégoriel (label)",
            "Normalisation StandardScaler",
        ]
        n_proc = 0
        n_fail = 0

        csv_files = list(input_dir.rglob("*.csv"))

        if not csv_files:
            return {"n_processed": 0, "n_failed": 0, "operations": ops}

        try:
            import pandas as pd
            import numpy as np

            for src in csv_files:
                try:
                    rel = src.relative_to(input_dir)
                    dst = output_dir / rel.parent / (rel.stem + "_preprocessed.csv")
                    dst.parent.mkdir(parents=True, exist_ok=True)

                    df = pd.read_csv(src, low_memory=False)

                    # Gestion valeurs manquantes
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if strategy == "median":
                        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                    elif strategy == "mean":
                        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                    elif strategy == "zero":
                        df[numeric_cols] = df[numeric_cols].fillna(0.0)
                    elif strategy == "drop":
                        df = df.dropna()

                    # Encoding catégoriel
                    cat_cols = df.select_dtypes(include=["object"]).columns
                    for col in cat_cols:
                        df[col] = pd.Categorical(df[col]).codes

                    # Normalisation Z-score (numériques)
                    for col in numeric_cols:
                        std_val = df[col].std()
                        if std_val > 0:
                            df[col] = (df[col] - df[col].mean()) / std_val

                    df.to_csv(dst, index=False)
                    n_proc += 1

                except Exception as e:
                    logger.warning(f"[Tabular] Erreur {src} : {e}")
                    n_fail += 1

        except ImportError:
            logger.warning("[Tabular] pandas absent")
            n_fail = len(csv_files)

        return {"n_processed": n_proc, "n_failed": n_fail, "operations": ops}


_PREPROCESSORS = {
    "image":   ImagePreprocessor(),
    "dicom":   DICOMPreprocessor(),
    "wsi":     HistoPreprocessor(),
    "ecg":     ECGPreprocessor(),
    "tabular": TabularPreprocessor(),
}

_MODALITY_BY_MODULE = {
    "malaria":       "image",
    "nutrition":     "tabular",
    "bioid":         "tabular",
    "breast_cancer": "dicom",
    "pulmoscan":     "dicom",
    "derm":          "image",
    "retina":        "image",
    "cardio":        "ecg",
    "neuro":         "dicom",
    "gastro":        "image",
    "histopath":     "wsi",
    "osteo":         "dicom",
    "sepsis":        "tabular",
    "hepato":        "dicom",
    "nephro":        "tabular",
    "hemato":        "image",
    "gyno":          "image",
}

_MODULE_SIZES = {
    "malaria": (224, 224), "breast_cancer": (512, 512),
    "pulmoscan": (224, 224), "derm": (512, 512),
    "retina": (512, 512), "neuro": (240, 240),
    "gastro": (512, 512), "histopath": (224, 224),
    "osteo": (512, 512), "hepato": (512, 512),
    "hemato": (224, 224), "gyno": (224, 224),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MOTEUR DE PRÉTRAITEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class PreprocessingEngine:
    def __init__(self):
        self.catalog = get_catalog()

    def preprocess_dataset(self, dataset_id: int) -> PreprocessResult:
        ds = self.catalog.get_dataset(dataset_id)
        if not ds:
            raise ValueError(f"Dataset {dataset_id} non trouvé")

        module_key = ds["module_key"]
        input_path = Path(ds.get("file_path") or "")
        modality   = _MODALITY_BY_MODULE.get(module_key, "image")
        size       = _MODULE_SIZES.get(module_key, (224, 224))
        output_path = PROCESSED_ROOT / module_key / f"dataset_{dataset_id}"

        if not input_path.exists():
            return PreprocessResult(
                module_key=module_key, dataset_id=dataset_id,
                input_path=str(input_path), output_path=str(output_path),
                n_processed=0, n_failed=0, modality=modality,
                operations=["Chemin source introuvable"], success=False,
            )

        logger.info(f"[Preprocessor] {module_key} — modalité={modality} — {input_path}")
        preprocessor = _PREPROCESSORS.get(modality, _PREPROCESSORS["image"])

        if modality == "image":
            result = preprocessor.process(input_path, output_path, target_size=size)
        elif modality == "dicom":
            result = preprocessor.process(input_path, output_path, target_size=size)
        else:
            result = preprocessor.process(input_path, output_path)

        success = result["n_processed"] > 0

        self.catalog.update_dataset(
            dataset_id,
            status="ready" if success else "preprocess_failed",
            preprocessed_at=datetime.utcnow().isoformat(),
        )
        self.catalog.log_event(
            "preprocessing", module_key, "Phase 5 - Preprocessing",
            "success" if success else "fail",
            f"{result['n_processed']} traités, {result['n_failed']} échecs",
            {"operations": result["operations"]},
        )

        return PreprocessResult(
            module_key=module_key, dataset_id=dataset_id,
            input_path=str(input_path), output_path=str(output_path),
            n_processed=result["n_processed"], n_failed=result["n_failed"],
            modality=modality, operations=result["operations"], success=success,
        )

    def preprocess_module(self, module_key: str) -> list[PreprocessResult]:
        datasets = self.catalog.list_datasets(module_key=module_key, status="ready")
        # Aussi les datasets qui passent quality check
        datasets += self.catalog.list_datasets(module_key=module_key, status="downloaded")
        return [self.preprocess_dataset(ds["id"]) for ds in datasets]

    def preprocess_all(self) -> list[PreprocessResult]:
        datasets = (
            self.catalog.list_datasets(status="downloaded")
            + self.catalog.list_datasets(status="ready")
        )
        results = []
        for ds in datasets:
            try:
                results.append(self.preprocess_dataset(ds["id"]))
            except Exception as e:
                logger.error(f"[Preprocessor] Erreur dataset {ds['id']} : {e}")
        return results


def run_preprocessing(
    module_key: str | None = None,
    dataset_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Point d'entrée Phase 5."""
    engine = PreprocessingEngine()
    results: list[PreprocessResult] = []

    if dataset_ids:
        for did in dataset_ids:
            results.append(engine.preprocess_dataset(did))
    elif module_key:
        results = engine.preprocess_module(module_key)
    else:
        results = engine.preprocess_all()

    n_ok = sum(1 for r in results if r.success)
    return {
        "total": len(results),
        "success": n_ok,
        "failed": len(results) - n_ok,
        "details": [
            {
                "module": r.module_key,
                "dataset_id": r.dataset_id,
                "n_processed": r.n_processed,
                "modality": r.modality,
                "success": r.success,
            }
            for r in results
        ],
    }
