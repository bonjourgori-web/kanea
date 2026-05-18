"""
KANEA AI — Préprocesseur
Images  : resize 224×224, normalisation, augmentation, DICOM→PNG
Tabulaire: nettoyage, imputation, scaling

Appel principal : preprocess_data()
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import KANEA_ROOT, get_logger, safe_run, ensure_dirs, count_images, timestamp

log = get_logger(__name__)

IMG_SIZE = (224, 224)

# Chemins sources
SRC_MALARIA   = KANEA_ROOT / "data" / "malaria"
SRC_NUTRITION = KANEA_ROOT / "data" / "nutrition" / "nutrition_dataset.csv"
SRC_BREAST    = KANEA_ROOT / "data" / "breast_cancer"

# Chemins de sortie
DST_MALARIA   = KANEA_ROOT / "data" / "malaria_processed"
DST_NUTRITION = KANEA_ROOT / "data" / "nutrition" / "nutrition_processed.csv"
DST_BREAST    = KANEA_ROOT / "data" / "breast_cancer_processed"

SCALER_PATH   = KANEA_ROOT / "models" / "machine_learning" / "nutrition_scaler.pkl"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS IMAGES
# ═══════════════════════════════════════════════════════════════════════════════

def _convert_dicom_folder(src: Path, dst: Path) -> int:
    """Convertit tous les .dcm d'un dossier en PNG 224×224."""
    try:
        import pydicom
    except ImportError:
        log.warning("pydicom non installé — conversion DICOM ignorée")
        return 0

    dcm_files = list(src.glob("**/*.dcm"))
    if not dcm_files:
        return 0

    ensure_dirs(dst)
    converted = 0
    log.info(f"Conversion DICOM : {len(dcm_files)} fichiers dans {src.name}")

    for dcm_path in dcm_files:
        try:
            ds = pydicom.dcmread(str(dcm_path))
            arr = ds.pixel_array.astype(np.float32)

            # Normalisation [0, 255]
            if arr.max() > 255:
                arr = (arr / arr.max() * 255)
            arr = arr.astype(np.uint8)

            img = Image.fromarray(arr)
            if img.mode not in ("L", "RGB"):
                img = img.convert("L")
            img = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
            img.save(dst / f"{dcm_path.stem}.png")
            converted += 1
        except Exception as exc:
            log.warning(f"Échec DICOM {dcm_path.name} : {exc}")

    return converted


def _process_image_folder(src: Path, dst: Path, augment: bool = True) -> int:
    """
    Resize + save chaque image.
    Si augment=True : flip horizontal + rotation aléatoire (×2 images supplémentaires).
    """
    if not src.exists():
        log.warning(f"Dossier source introuvable : {src}")
        return 0

    ensure_dirs(dst)
    images = list(src.glob("*.png")) + list(src.glob("*.jpg")) + list(src.glob("*.jpeg"))
    log.info(f"  {src.name} : {len(images)} images")
    done = 0

    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
            img.save(dst / img_path.name)
            done += 1

            if augment:
                # Flip horizontal
                img.transpose(Image.FLIP_LEFT_RIGHT).save(dst / f"h_{img_path.name}")
                # Rotation légère
                angle = np.random.randint(-15, 16)
                img.rotate(angle).save(dst / f"r{angle}_{img_path.name}")
                done += 2
        except Exception as exc:
            log.warning(f"Image ignorée {img_path.name} : {exc}")

    return done


# ═══════════════════════════════════════════════════════════════════════════════
# PRÉPROCESSEURS PAR MODULE
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_malaria(augment: bool = True) -> bool:
    """Prétraite les images de cellules malaria (Parasitised / Uninfected)."""
    log.info("─── Malaria ───")
    total = 0
    for cls in ["Parasitised", "Uninfected"]:
        src = SRC_MALARIA / cls
        dst = DST_MALARIA / cls

        # Conversion DICOM si présent
        _convert_dicom_folder(src, dst)

        # Images PNG/JPG
        n = _process_image_folder(src, dst, augment=augment)
        total += n

    if total == 0:
        log.warning("Aucune image malaria traitée — relancez download_data()")
        return False

    log.info(f"Malaria : {total} images traitées → {DST_MALARIA}")
    return True


def preprocess_nutrition() -> bool:
    """
    Nettoyage + imputation + scaling des données nutritionnelles.
    Sauvegarde le scaler pour l'inférence.
    """
    from sklearn.preprocessing import StandardScaler

    log.info("─── Nutrition ───")

    if not SRC_NUTRITION.exists():
        log.error(f"Dataset introuvable : {SRC_NUTRITION}")
        return False

    df = pd.read_csv(SRC_NUTRITION)
    log.info(f"Shape initial : {df.shape}")

    # Dédoublonnage
    df = df.drop_duplicates()

    # Colonnes attendues
    num_cols = ["age_months", "weight_kg", "height_cm", "muac_cm", "whz", "haz", "waz", "bmi"]
    num_cols = [c for c in num_cols if c in df.columns]

    # Imputation médiane
    for col in num_cols:
        median = df[col].median()
        n_missing = df[col].isna().sum()
        if n_missing:
            log.info(f"  Imputation {col} : {n_missing} valeurs manquantes → {median:.2f}")
        df[col] = df[col].fillna(median)

    # Encodage sexe
    if "sex" in df.columns:
        df["sex_encoded"] = (df["sex"].str.strip().str.upper() == "M").astype(int)
        num_cols.append("sex_encoded")

    # Suppression lignes sans label
    df = df.dropna(subset=["nutrition_status"])

    # Standardisation
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Sauvegarde scaler
    ensure_dirs(SCALER_PATH.parent)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "features": num_cols}, f)
    log.info(f"Scaler sauvegardé : {SCALER_PATH}")

    # Sauvegarde CSV préprocessé
    df.to_csv(DST_NUTRITION, index=False)
    log.info(f"Nutrition préprocessée : {df.shape} → {DST_NUTRITION.name}")
    log.info(f"Distribution :\n{df['nutrition_status'].value_counts().to_string()}")
    return True


def preprocess_breast_cancer(augment: bool = True) -> bool:
    """
    Prétraite les mammographies.
    Convertit les DICOM en PNG si nécessaires, puis resize + augmentation.
    """
    log.info("─── Cancer du sein ───")
    total = 0
    for cls in ["Normal", "Benign", "Malignant"]:
        src = SRC_BREAST / cls
        dst = DST_BREAST / cls

        _convert_dicom_folder(src, dst)
        n = _process_image_folder(src, dst, augment=augment)
        total += n

    if total == 0:
        log.warning("Aucune image breast cancer traitée — relancez download_data()")
        return False

    log.info(f"Cancer du sein : {total} images traitées → {DST_BREAST}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

@safe_run
def preprocess_data(augment: bool = True) -> dict:
    """
    Lance le prétraitement complet des 3 modules.

    Args:
        augment: active l'augmentation de données pour les images.

    Returns:
        dict statut par module.
    """
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   KANEA AI — PRÉTRAITEMENT DES DONNÉES               ║")
    log.info(f"║   {timestamp()}                              ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    results = {
        "malaria":       preprocess_malaria(augment=augment),
        "nutrition":     preprocess_nutrition(),
        "breast_cancer": preprocess_breast_cancer(augment=augment),
    }

    ok = sum(v for v in results.values() if v)
    log.info(f"Prétraitement terminé : {ok}/{len(results)} modules OK")
    for module, status in results.items():
        icon = "✓" if status else "✗"
        log.info(f"  {icon} {module}")

    return results
