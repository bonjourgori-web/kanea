"""
KANÉA — Data Quality Control
==============================
Phase 4 : Contrôle automatisé de la qualité des données médicales.
Score 0–100. Seuil minimal accepté : 80.

Contrôles : images corrompues · flou · doublons · labels manquants · erreurs annotations
"""
from __future__ import annotations

import hashlib
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.catalog import get_catalog

logger = logging.getLogger("kanea.quality")

QUALITY_THRESHOLD = 80.0  # Score minimum pour dataset utilisable


# ═══════════════════════════════════════════════════════════════════════════════
# RÉSULTAT DE QUALITÉ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityReport:
    module_key: str
    dataset_id: int
    dataset_path: str
    score: float
    n_total: int
    n_valid: int
    n_corrupted: int
    n_duplicates: int
    n_missing_labels: int
    n_blurry: int = 0
    issues: list[str] = field(default_factory=list)
    passed: bool = False
    details: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def _is_valid_image(path: Path) -> bool:
    """Vérifie si le fichier est une image lisible (entête PNG/JPG/TIFF/BMP)."""
    try:
        suffix = path.suffix.lower()
        with open(path, "rb") as f:
            header = f.read(16)
        if suffix in (".png",):
            return header[:8] == b"\x89PNG\r\n\x1a\n"
        if suffix in (".jpg", ".jpeg"):
            return header[:2] == b"\xff\xd8"
        if suffix in (".tif", ".tiff"):
            return header[:4] in (b"II*\x00", b"MM\x00*")
        if suffix in (".bmp",):
            return header[:2] == b"BM"
        if suffix in (".dicom", ".dcm"):
            return True  # DICOM validé par pydicom séparément
        return True
    except Exception:
        return False


def _is_valid_dicom(path: Path) -> bool:
    try:
        import pydicom  # type: ignore
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        return bool(ds.get("PatientID") or ds.get("Modality"))
    except ImportError:
        return True
    except Exception:
        return False


def _check_blur(path: Path, threshold: float = 100.0) -> bool:
    """Retourne True si l'image est floue (Laplacian variance < threshold)."""
    try:
        import cv2  # type: ignore
        import numpy as np
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        return float(cv2.Laplacian(img, cv2.CV_64F).var()) < threshold
    except ImportError:
        return False  # OpenCV absent — on suppose OK
    except Exception:
        return False


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(4096))
    return h.hexdigest()


def _is_valid_csv_row(row: dict[str, Any], required_cols: list[str]) -> bool:
    return all(
        row.get(c) is not None and str(row.get(c, "")).strip() != ""
        for c in required_cols
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRÔLEURS PAR MODALITÉ
# ═══════════════════════════════════════════════════════════════════════════════

class ImageQualityChecker:
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm", ".dicom"}

    def check(self, data_dir: Path, module_key: str) -> dict[str, Any]:
        all_files = [
            p for p in data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS
        ]

        if not all_files:
            return {
                "n_total": 0, "n_valid": 0, "n_corrupted": 0,
                "n_duplicates": 0, "n_missing_labels": 0, "n_blurry": 0,
                "issues": ["Aucun fichier image trouvé"],
                "score": 0.0,
            }

        n_total     = len(all_files)
        n_corrupted = 0
        n_blurry    = 0
        hashes: set[str] = set()
        n_duplicates = 0
        issues: list[str] = []

        for p in all_files:
            suffix = p.suffix.lower()
            if suffix in (".dcm", ".dicom"):
                ok = _is_valid_dicom(p)
            else:
                ok = _is_valid_image(p)

            if not ok:
                n_corrupted += 1
                continue

            # Flou
            if suffix in (".jpg", ".jpeg", ".png", ".bmp") and _check_blur(p):
                n_blurry += 1

            # Doublons
            h = _file_hash(p)
            if h in hashes:
                n_duplicates += 1
            else:
                hashes.add(h)

        # Labels (présence de CSV/JSON dans le dossier)
        n_missing_labels = 0
        label_files = list(data_dir.rglob("*.csv")) + list(data_dir.rglob("*.json"))
        if not label_files:
            issues.append("Aucun fichier de labels détecté (CSV/JSON)")
            n_missing_labels = n_total

        # Score
        n_valid  = n_total - n_corrupted - n_duplicates
        corrupt_pen = (n_corrupted / n_total) * 40
        dup_pen     = (n_duplicates / n_total) * 20
        blur_pen    = (n_blurry / n_total) * 10
        label_pen   = 20 if n_missing_labels > 0 else 0
        score = max(0.0, min(100.0, 100 - corrupt_pen - dup_pen - blur_pen - label_pen))

        if n_corrupted > 0:
            issues.append(f"{n_corrupted} images corrompues détectées")
        if n_duplicates > 0:
            issues.append(f"{n_duplicates} doublons détectés")
        if n_blurry > n_total * 0.1:
            issues.append(f"{n_blurry} images floues ({n_blurry/n_total:.1%})")

        return {
            "n_total": n_total, "n_valid": n_valid,
            "n_corrupted": n_corrupted, "n_duplicates": n_duplicates,
            "n_missing_labels": n_missing_labels, "n_blurry": n_blurry,
            "issues": issues, "score": round(score, 1),
        }


class TabularQualityChecker:
    def check(self, data_dir: Path, module_key: str) -> dict[str, Any]:
        csv_files = list(data_dir.rglob("*.csv")) + list(data_dir.rglob("*.xlsx"))

        if not csv_files:
            return {
                "n_total": 0, "n_valid": 0, "n_corrupted": 0,
                "n_duplicates": 0, "n_missing_labels": 0, "n_blurry": 0,
                "issues": ["Aucun fichier CSV/XLSX trouvé"],
                "score": 0.0,
            }

        n_total = 0
        n_missing = 0
        n_duplicates = 0
        issues: list[str] = []

        try:
            import pandas as pd
            dfs = []
            for f in csv_files[:5]:  # Limité à 5 pour éviter surcharge
                try:
                    if f.suffix == ".xlsx":
                        df = pd.read_excel(f)
                    else:
                        df = pd.read_csv(f, low_memory=False)
                    dfs.append(df)
                except Exception as e:
                    issues.append(f"Fichier illisible : {f.name} — {e}")

            if dfs:
                import functools
                combined = functools.reduce(lambda a, b: pd.concat([a, b], ignore_index=True), dfs)
                n_total     = len(combined)
                n_missing   = int(combined.isnull().any(axis=1).sum())
                n_duplicates = int(combined.duplicated().sum())
                n_valid     = n_total - n_missing - n_duplicates

                miss_pct = n_missing / n_total if n_total > 0 else 0
                dup_pct  = n_duplicates / n_total if n_total > 0 else 0
                score    = max(0.0, 100 - miss_pct * 50 - dup_pct * 30)

                if n_missing > 0:
                    issues.append(f"{n_missing} lignes avec valeurs manquantes ({miss_pct:.1%})")
                if n_duplicates > 0:
                    issues.append(f"{n_duplicates} doublons ({dup_pct:.1%})")

                return {
                    "n_total": n_total, "n_valid": n_valid,
                    "n_corrupted": 0, "n_duplicates": n_duplicates,
                    "n_missing_labels": n_missing, "n_blurry": 0,
                    "issues": issues, "score": round(score, 1),
                }
        except ImportError:
            issues.append("pandas non disponible — contrôle partiel")

        # Fallback
        n_total = len(csv_files)
        return {
            "n_total": n_total, "n_valid": n_total, "n_corrupted": 0,
            "n_duplicates": 0, "n_missing_labels": 0, "n_blurry": 0,
            "issues": issues, "score": 70.0,
        }


class WSIQualityChecker:
    """Contrôle qualité Whole Slide Images (TIF/SVS)."""
    WSI_EXTS = {".tif", ".tiff", ".svs", ".ndpi", ".vms", ".scn"}

    def check(self, data_dir: Path, module_key: str) -> dict[str, Any]:
        wsi_files = [
            p for p in data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.WSI_EXTS
        ]
        if not wsi_files:
            return {
                "n_total": 0, "n_valid": 0, "n_corrupted": 0,
                "n_duplicates": 0, "n_missing_labels": 0, "n_blurry": 0,
                "issues": ["Aucune WSI trouvée"], "score": 0.0,
            }

        n_total = len(wsi_files)
        n_corrupted = 0
        issues: list[str] = []

        for p in wsi_files:
            if p.stat().st_size < 1024 * 10:  # < 10 KB = probablement corrompu
                n_corrupted += 1

        score = max(0.0, 100 - (n_corrupted / n_total) * 60)
        return {
            "n_total": n_total, "n_valid": n_total - n_corrupted,
            "n_corrupted": n_corrupted, "n_duplicates": 0,
            "n_missing_labels": 0, "n_blurry": 0,
            "issues": issues, "score": round(score, 1),
        }


class ECGQualityChecker:
    ECG_EXTS = {".hea", ".dat", ".atr", ".csv", ".psv"}

    def check(self, data_dir: Path, module_key: str) -> dict[str, Any]:
        ecg_files = [
            p for p in data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.ECG_EXTS
        ]
        if not ecg_files:
            return {
                "n_total": 0, "n_valid": 0, "n_corrupted": 0,
                "n_duplicates": 0, "n_missing_labels": 0, "n_blurry": 0,
                "issues": ["Aucun fichier ECG trouvé"], "score": 0.0,
            }
        n_total = len(ecg_files)
        n_corrupted = sum(1 for p in ecg_files if p.stat().st_size < 10)
        score = max(0.0, 100 - (n_corrupted / n_total) * 50)
        return {
            "n_total": n_total, "n_valid": n_total - n_corrupted,
            "n_corrupted": n_corrupted, "n_duplicates": 0,
            "n_missing_labels": 0, "n_blurry": 0,
            "issues": [], "score": round(score, 1),
        }


_MODALITY_CHECKERS = {
    "image":   ImageQualityChecker(),
    "tabular": TabularQualityChecker(),
    "wsi":     WSIQualityChecker(),
    "ecg":     ECGQualityChecker(),
    "dicom":   ImageQualityChecker(),  # Partage le checker image
}


# ═══════════════════════════════════════════════════════════════════════════════
# MOTEUR DE CONTRÔLE QUALITÉ
# ═══════════════════════════════════════════════════════════════════════════════

class QualityController:
    def __init__(self):
        self.catalog = get_catalog()

    def _detect_modality(self, data_dir: Path) -> str:
        """Détecte la modalité principale du dossier."""
        exts = {p.suffix.lower() for p in data_dir.rglob("*") if p.is_file()}
        if exts & {".png", ".jpg", ".jpeg", ".bmp", ".dcm"}:
            return "image"
        if exts & {".csv", ".xlsx", ".parquet"}:
            return "tabular"
        if exts & {".tif", ".tiff", ".svs"}:
            return "wsi"
        if exts & {".hea", ".dat", ".atr", ".psv"}:
            return "ecg"
        return "image"

    def check_dataset(self, dataset_id: int) -> QualityReport:
        ds = self.catalog.get_dataset(dataset_id)
        if not ds:
            raise ValueError(f"Dataset {dataset_id} non trouvé")

        module_key = ds["module_key"]
        data_path  = Path(ds.get("file_path") or "")

        if not data_path.exists():
            report = QualityReport(
                module_key=module_key, dataset_id=dataset_id,
                dataset_path=str(data_path), score=0.0,
                n_total=0, n_valid=0, n_corrupted=0,
                n_duplicates=0, n_missing_labels=0,
                issues=["Chemin de données non trouvé"],
                passed=False,
            )
            self.catalog.update_dataset(dataset_id, quality_score=0.0)
            return report

        modality = self._detect_modality(data_path)
        checker  = _MODALITY_CHECKERS.get(modality, _MODALITY_CHECKERS["image"])

        logger.info(f"[Quality] {module_key} — {data_path} — modalité={modality}")
        result = checker.check(data_path, module_key)

        score  = result["score"]
        passed = score >= QUALITY_THRESHOLD

        report = QualityReport(
            module_key=module_key,
            dataset_id=dataset_id,
            dataset_path=str(data_path),
            score=score,
            n_total=result["n_total"],
            n_valid=result["n_valid"],
            n_corrupted=result["n_corrupted"],
            n_duplicates=result["n_duplicates"],
            n_missing_labels=result["n_missing_labels"],
            n_blurry=result.get("n_blurry", 0),
            issues=result["issues"],
            passed=passed,
        )

        self.catalog.save_quality_report(
            dataset_id=dataset_id,
            module_key=module_key,
            score=score,
            n_total=result["n_total"],
            n_valid=result["n_valid"],
            n_corrupted=result["n_corrupted"],
            n_duplicates=result["n_duplicates"],
            n_missing_labels=result["n_missing_labels"],
            issues=result["issues"],
        )
        self.catalog.update_dataset(
            dataset_id,
            quality_score=score,
            status="ready" if passed else "quality_failed",
        )
        self.catalog.log_event(
            "quality_check", module_key, "Phase 4 - Quality",
            "pass" if passed else "fail",
            f"Score {score:.1f}/100 — {'PASS' if passed else 'FAIL'}",
            {"score": score, "threshold": QUALITY_THRESHOLD, "issues": result["issues"]},
        )

        logger.info(
            f"[Quality] Score {score:.1f}/100 — "
            f"{'✓ PASS' if passed else '✗ FAIL (< ' + str(QUALITY_THRESHOLD) + ')'}"
        )
        return report

    def check_all_downloaded(self) -> list[QualityReport]:
        datasets = self.catalog.list_datasets(status="downloaded")
        reports = []
        for ds in datasets:
            try:
                report = self.check_dataset(ds["id"])
                reports.append(report)
            except Exception as e:
                logger.error(f"[Quality] Erreur dataset {ds['id']} : {e}")
        return reports

    def check_module(self, module_key: str) -> list[QualityReport]:
        datasets = self.catalog.list_datasets(module_key=module_key, status="downloaded")
        return [self.check_dataset(ds["id"]) for ds in datasets]

    def get_quality_summary(self, reports: list[QualityReport]) -> dict[str, Any]:
        if not reports:
            return {"total": 0, "passed": 0, "failed": 0, "avg_score": 0.0}
        passed = [r for r in reports if r.passed]
        scores = [r.score for r in reports]
        return {
            "total": len(reports),
            "passed": len(passed),
            "failed": len(reports) - len(passed),
            "avg_score": round(sum(scores) / len(scores), 1),
            "min_score": round(min(scores), 1),
            "max_score": round(max(scores), 1),
        }


def run_quality_check(
    module_key: str | None = None,
    dataset_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Point d'entrée Phase 4."""
    controller = QualityController()
    reports: list[QualityReport] = []

    if dataset_ids:
        for did in dataset_ids:
            reports.append(controller.check_dataset(did))
    elif module_key:
        reports = controller.check_module(module_key)
    else:
        reports = controller.check_all_downloaded()

    return controller.get_quality_summary(reports)
