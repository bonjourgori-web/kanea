"""
KANEA AI — Data Loader
Téléchargement automatique des datasets :
  - Malaria    : Kaggle API → NIH URL → synthétique
  - Nutrition  : génération synthétique (WHO standards)
  - Cancer sein: MIAS URL → synthétique

Appel principal : download_data()
"""

import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from src.utils import KANEA_ROOT, get_logger, safe_run, ensure_dirs, count_images, timestamp

log = get_logger(__name__)

# ─── Chemins ──────────────────────────────────────────────────────────────────

DATA_MALARIA      = KANEA_ROOT / "data" / "malaria"
DATA_NUTRITION    = KANEA_ROOT / "data" / "nutrition"
DATA_BREAST       = KANEA_ROOT / "data" / "breast_cancer"

MALARIA_PARASITISED = DATA_MALARIA / "Parasitised"
MALARIA_UNINFECTED  = DATA_MALARIA / "Uninfected"

BREAST_CLASSES = ["Normal", "Benign", "Malignant"]

# URLs de téléchargement direct
NIH_MALARIA_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — Téléchargement & extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _try_http_download(url: str, dst_path: Path, timeout: int = 60) -> bool:
    """Télécharge un fichier via HTTP avec suivi de progression."""
    try:
        import requests
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Téléchargement : {url}")
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk = 8 * 1024

        with open(dst_path, "wb") as f:
            for data in resp.iter_content(chunk_size=chunk):
                if data:
                    f.write(data)
                    downloaded += len(data)
                    if total and downloaded % (20 * 1024 * 1024) < chunk:
                        pct = 100 * downloaded / total
                        log.info(f"  {pct:.0f}% ({downloaded // 1_048_576} Mo / {total // 1_048_576} Mo)")

        log.info(f"Téléchargement terminé : {dst_path.name}")
        return True

    except Exception as exc:
        log.warning(f"Échec HTTP ({url}) : {exc}")
        if dst_path.exists():
            dst_path.unlink()
        return False


def _extract_zip(zip_path: Path, dst_dir: Path) -> bool:
    try:
        log.info(f"Extraction : {zip_path.name} → {dst_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dst_dir)
        zip_path.unlink()
        return True
    except Exception as exc:
        log.warning(f"Échec extraction : {exc}")
        return False


def _try_kaggle(dataset_id: str, dst_dir: Path) -> bool:
    """Télécharge via l'API Kaggle si les credentials sont présents."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        log.warning("~/.kaggle/kaggle.json introuvable — API Kaggle désactivée")
        return False

    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Kaggle : téléchargement de {dataset_id}")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(dst_dir), "--unzip"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            log.info(f"Kaggle : succès pour {dataset_id}")
            return True
        log.warning(f"Kaggle : échec → {result.stderr.strip()}")
        return False

    except FileNotFoundError:
        log.warning("Commande 'kaggle' introuvable (pip install kaggle)")
        return False
    except Exception as exc:
        log.warning(f"Kaggle : erreur inattendue : {exc}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# GÉNÉRATEURS SYNTHÉTIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_malaria(n: int = 500) -> None:
    """
    Génère des images synthétiques de cellules sanguines.
    Parasitisées : cercle rouge avec tache sombre (forme en anneau plasmodium).
    Non infectées : cellule rouge propre, forme biconcave.
    """
    log.info(f"Génération synthétique malaria : {n} images/classe")
    ensure_dirs(MALARIA_PARASITISED, MALARIA_UNINFECTED)
    rng = np.random.default_rng(42)

    for i in range(n):
        size = 100

        # ── Cellule non infectée ──────────────────────────────────────────────
        img = Image.new("RGB", (size, size), (25, 25, 25))
        draw = ImageDraw.Draw(img)
        r_cell = int(rng.integers(32, 42))
        cx, cy = size // 2, size // 2
        color_cell = (
            int(rng.integers(185, 220)),
            int(rng.integers(45, 75)),
            int(rng.integers(45, 75)),
        )
        color_center = tuple(max(0, c - 40) for c in color_cell)
        draw.ellipse([cx - r_cell, cy - r_cell, cx + r_cell, cy + r_cell], fill=color_cell)
        draw.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=color_center)
        img.save(MALARIA_UNINFECTED / f"cell_u_{i:05d}.png")

        # ── Cellule parasitisée ───────────────────────────────────────────────
        img2 = img.copy()
        draw2 = ImageDraw.Draw(img2)
        px = int(rng.integers(cx - r_cell + 8, cx + r_cell - 8))
        py = int(rng.integers(cy - r_cell + 8, cy + r_cell - 8))
        pr = int(rng.integers(5, 11))
        # Anneau plasmodium (bord sombre, centre clair)
        draw2.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(45, 15, 75))
        if pr > 4:
            draw2.ellipse([px - pr + 2, py - pr + 2, px + pr - 2, py + pr - 2], fill=color_cell)
        img2.save(MALARIA_PARASITISED / f"cell_p_{i:05d}.png")

    log.info(
        f"Malaria synthétique créé : "
        f"{count_images(MALARIA_PARASITISED)} parasitisées, "
        f"{count_images(MALARIA_UNINFECTED)} non infectées"
    )


def _generate_synthetic_nutrition(n: int = 2000) -> None:
    """
    Génère un dataset tabulaire de nutrition suivant les distributions WHO.
    Classes : MAS (5%), MAM (15%), Normal (70%), Overweight (10%).
    """
    csv_path = DATA_NUTRITION / "nutrition_dataset.csv"
    ensure_dirs(DATA_NUTRITION)
    log.info(f"Génération synthétique nutrition : {n} lignes")

    rng = np.random.default_rng(42)

    class_config = {
        "MAS":        {"frac": 0.05, "whz": (-3.8, 0.3), "haz": (-3.2, 0.5), "waz": (-3.8, 0.4), "muac": (7.5,  11.5)},
        "MAM":        {"frac": 0.15, "whz": (-2.5, 0.3), "haz": (-2.0, 0.5), "waz": (-2.3, 0.4), "muac": (11.5, 12.5)},
        "Normal":     {"frac": 0.70, "whz": ( 0.0, 0.8), "haz": ( 0.0, 1.0), "waz": ( 0.0, 0.9), "muac": (13.5, 16.0)},
        "Overweight": {"frac": 0.10, "whz": ( 2.6, 0.4), "haz": ( 0.5, 0.8), "waz": ( 2.6, 0.5), "muac": (17.0, 21.0)},
    }

    rows = []
    for cls, cfg in class_config.items():
        n_cls = int(n * cfg["frac"])
        for _ in range(n_cls):
            sex = rng.choice(["M", "F"])
            age = int(rng.integers(6, 60))

            whz = float(rng.normal(*cfg["whz"]))
            haz = float(rng.normal(*cfg["haz"]))
            waz = float(rng.normal(*cfg["waz"]))
            muac = float(rng.uniform(*cfg["muac"]))

            # Approximation poids/taille à partir des z-scores (référence médiane OMS)
            w_med = 5.0 + age / 60 * 15.0
            h_med = 65.0 + age / 60 * 30.0
            weight_kg = max(2.5, w_med + waz * w_med * 0.12)
            height_cm = max(50.0, h_med + haz * h_med * 0.05)
            bmi = weight_kg / (height_cm / 100) ** 2

            rows.append({
                "age_months":    age,
                "sex":           sex,
                "weight_kg":     round(weight_kg, 2),
                "height_cm":     round(height_cm, 1),
                "muac_cm":       round(muac, 1),
                "whz":           round(whz, 3),
                "haz":           round(haz, 3),
                "waz":           round(waz, 3),
                "bmi":           round(bmi, 2),
                "nutrition_status": cls,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    log.info(f"Nutrition synthétique créé : {csv_path} ({len(df)} lignes)")
    log.info(f"Distribution classes :\n{df['nutrition_status'].value_counts().to_string()}")


def _generate_synthetic_breast_cancer(n_per_class: int = 150) -> None:
    """
    Génère des images synthétiques de mammographies grises.
    Normal : tissu homogène.
    Benign : masse ronde bien délimitée.
    Malignant : masse irrégulière avec spicules.
    """
    log.info(f"Génération synthétique cancer du sein : {n_per_class} images/classe")
    rng = np.random.default_rng(42)

    for cls in BREAST_CLASSES:
        cls_dir = DATA_BREAST / cls
        ensure_dirs(cls_dir)

        for i in range(n_per_class):
            # Tissu de fond
            base = {"Normal": 175, "Benign": 155, "Malignant": 140}[cls]
            noise = {"Normal": 12, "Benign": 18, "Malignant": 22}[cls]
            arr = rng.normal(base, noise, (224, 224)).astype(np.float32)

            if cls != "Normal":
                cx = int(rng.integers(55, 169))
                cy = int(rng.integers(55, 169))
                r  = int(rng.integers(12, 32))
                y_g, x_g = np.ogrid[:224, :224]

                if cls == "Benign":
                    mask = (x_g - cx) ** 2 + (y_g - cy) ** 2 <= r ** 2
                    arr[mask] = np.clip(arr[mask] + 65, 0, 255)

                else:  # Malignant
                    # Noyau irrégulier
                    for _ in range(6):
                        dx = int(rng.integers(-r // 3, r // 3 + 1))
                        dy = int(rng.integers(-r // 3, r // 3 + 1))
                        rx = r + int(rng.integers(-4, 8))
                        ry = r + int(rng.integers(-4, 8))
                        rx, ry = max(rx, 1), max(ry, 1)
                        mask = (x_g - cx - dx) ** 2 / rx ** 2 + (y_g - cy - dy) ** 2 / ry ** 2 <= 1
                        arr[mask] = np.clip(arr[mask] + 72, 0, 255)
                    # Spicules
                    for _ in range(10):
                        angle = rng.uniform(0, 2 * np.pi)
                        length = r + int(rng.integers(8, 28))
                        ex = int(cx + length * np.cos(angle))
                        ey = int(cy + length * np.sin(angle))
                        pts_x = np.linspace(cx, ex, 40).astype(int)
                        pts_y = np.linspace(cy, ey, 40).astype(int)
                        valid = (0 <= pts_x) & (pts_x < 224) & (0 <= pts_y) & (pts_y < 224)
                        arr[pts_y[valid], pts_x[valid]] = np.clip(
                            arr[pts_y[valid], pts_x[valid]] + 55, 0, 255
                        )

            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L").convert("RGB")
            img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
            img.save(cls_dir / f"{cls.lower()}_{i:05d}.png")

    for cls in BREAST_CLASSES:
        log.info(f"  {cls} : {count_images(DATA_BREAST / cls)} images")


# ═══════════════════════════════════════════════════════════════════════════════
# TÉLÉCHARGEMENTS PAR MODULE
# ═══════════════════════════════════════════════════════════════════════════════

def _download_malaria(force: bool = False) -> bool:
    n_para = count_images(MALARIA_PARASITISED)
    n_uninf = count_images(MALARIA_UNINFECTED)

    if not force and n_para > 100 and n_uninf > 100:
        log.info(f"Malaria déjà présent ({n_para} / {n_uninf} images) — ignoré")
        return True

    log.info("═══ MODULE 1 : Malaria ═══")

    # 1. Kaggle
    if _try_kaggle("iarunava/cell-images-for-detecting-malaria", DATA_MALARIA):
        # Réorganise si nécessaire (le zip Kaggle a un sous-dossier cell_images/)
        _reorganize_malaria_kaggle()
        return True

    # 2. NIH direct
    zip_path = DATA_MALARIA / "cell_images.zip"
    if _try_http_download(NIH_MALARIA_URL, zip_path):
        if _extract_zip(zip_path, DATA_MALARIA):
            _reorganize_malaria_kaggle()
            return True

    # 3. Mode hors-ligne : synthétique
    log.warning("Aucune source disponible → mode synthétique (démo)")
    _generate_synthetic_malaria(n=500)
    return True


def _reorganize_malaria_kaggle() -> None:
    """Déplace Parasitised/ et Uninfected/ depuis cell_images/ si besoin."""
    sub = DATA_MALARIA / "cell_images"
    if sub.exists():
        for cls in ["Parasitised", "Uninfected"]:
            src = sub / cls
            dst = DATA_MALARIA / cls
            if src.exists() and not dst.exists():
                src.rename(dst)
        if sub.exists() and not any(sub.iterdir()):
            sub.rmdir()


def _download_nutrition(force: bool = False) -> bool:
    csv_path = DATA_NUTRITION / "nutrition_dataset.csv"

    if not force and csv_path.exists() and csv_path.stat().st_size > 5_000:
        log.info(f"Nutrition déjà présent ({csv_path.name}) — ignoré")
        return True

    log.info("═══ MODULE 2 : Nutrition ═══")
    log.info("DHS / UNICEF nécessitent une inscription → génération synthétique (WHO)")
    _generate_synthetic_nutrition(n=2000)
    return True


def _download_breast_cancer(force: bool = False) -> bool:
    total = sum(count_images(DATA_BREAST / cls) for cls in BREAST_CLASSES if (DATA_BREAST / cls).exists())

    if not force and total > 100:
        log.info(f"Cancer du sein déjà présent ({total} images) — ignoré")
        return True

    log.info("═══ MODULE 3 : Cancer du sein ═══")
    log.info("CBIS-DDSM / MIAS nécessitent une inscription → mode synthétique (démo)")
    _generate_synthetic_breast_cancer(n_per_class=150)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

@safe_run
def download_data(force: bool = False) -> dict:
    """
    Télécharge (ou génère) les datasets pour les 3 modules.

    Priorité par module :
      Malaria     → Kaggle API → NIH direct → synthétique
      Nutrition   → génération synthétique (WHO standards)
      Cancer sein → synthétique (MIAS/CBIS-DDSM requièrent inscription)

    Args:
        force: force le re-téléchargement même si données déjà présentes.

    Returns:
        dict avec statut par module.
    """
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   KANEA AI — TÉLÉCHARGEMENT DES DONNÉES              ║")
    log.info(f"║   {timestamp()}                              ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    results = {
        "malaria":      _download_malaria(force),
        "nutrition":    _download_nutrition(force),
        "breast_cancer": _download_breast_cancer(force),
    }

    ok = sum(v for v in results.values() if v)
    log.info(f"Téléchargement terminé : {ok}/{len(results)} modules OK")
    for module, status in results.items():
        icon = "✓" if status else "✗"
        log.info(f"  {icon} {module}")

    return results
