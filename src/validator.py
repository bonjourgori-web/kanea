"""
KANEA AI — Validateur de données (grade médical)
═════════════════════════════════════════════════
Contrôles avant tout entraînement :
  ▸ Intégrité des images (corruption)
  ▸ Doublons
  ▸ Déséquilibre de classes
  ▸ Valeurs aberrantes physiologiques
  ▸ Biais géographiques / démographiques
  ▸ Cohérence des labels

Appel principal : validate_data()
Sortie          : reports/validation_report.json
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import KANEA_ROOT, get_logger, safe_run, timestamp

log = get_logger(__name__)

REPORTS_DIR = KANEA_ROOT / "reports"

# Seuils médicaux (OMS)
WHO_ZSCORE_RANGE = (-6.0, 6.0)
IMBALANCE_WARN   = 2.0   # ratio max/min → warning
IMBALANCE_CRIT   = 5.0   # ratio → critique


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _verify_image(path: Path) -> tuple[bool, str]:
    """Vérifie qu'une image est lisible (non corrompue)."""
    try:
        with Image.open(path) as img:
            img.verify()         # contrôle intégrité header
        with Image.open(path) as img:
            img.load()           # charge les pixels
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _sample_verify(img_paths: list, max_check: int = 200) -> dict:
    """
    Vérifie un échantillon d'images.
    Retourne : { "total": n, "checked": k, "corrupted": [...], "corruption_rate": r }
    """
    total = len(img_paths)
    sample = random.sample(img_paths, min(max_check, total)) if total else []

    corrupted = []
    for p in sample:
        ok, err = _verify_image(p)
        if not ok:
            corrupted.append({"path": str(p), "error": err})

    return {
        "total":           total,
        "checked":         len(sample),
        "corrupted":       corrupted,
        "corruption_rate": round(len(corrupted) / max(len(sample), 1), 4),
    }


def _balance_report(counts: dict[str, int]) -> dict:
    """Rapport équilibre des classes + alertes médicales."""
    total = sum(counts.values())
    if total == 0:
        return {"balanced": False, "reason": "Aucun échantillon"}

    max_cls = max(counts, key=counts.get)
    min_cls = min(counts, key=counts.get)
    ratio = counts[max_cls] / max(counts[min_cls], 1)

    warnings = []
    if ratio >= IMBALANCE_CRIT:
        warnings.append(
            f"⚠️ CRITIQUE : déséquilibre {ratio:.1f}x ({max_cls}={counts[max_cls]} vs "
            f"{min_cls}={counts[min_cls]}) — utiliser class_weight + SMOTE"
        )
    elif ratio >= IMBALANCE_WARN:
        warnings.append(
            f"⚠️ BIAIS CLASSE : ratio {ratio:.1f}x — appliquer class_weight ou oversampling"
        )

    percentages = {k: round(100 * v / total, 1) for k, v in counts.items()}

    return {
        "per_class":     counts,
        "percentages":   percentages,
        "total":         total,
        "ratio_max_min": round(ratio, 2),
        "balanced":      ratio < IMBALANCE_WARN,
        "warnings":      warnings,
        "recommendation": (
            "class_weight='balanced' dans sklearn / CrossEntropyLoss(weight=…) dans PyTorch"
            if ratio >= IMBALANCE_WARN else "OK"
        ),
    }


def _collect_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return (
        list(directory.glob("*.png")) +
        list(directory.glob("*.jpg")) +
        list(directory.glob("*.jpeg"))
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — MALARIA
# ═══════════════════════════════════════════════════════════════════════════════

def validate_malaria() -> dict:
    """
    Valide le dataset malaria.
    Contrôles : corruption, balance, présence des deux classes, biais géographique.
    """
    log.info("─── Validation Malaria ───")
    data_dir = KANEA_ROOT / "data" / "malaria"
    classes  = ["Parasitised", "Uninfected"]

    report: dict = {
        "module":         "malaria",
        "source_dir":     str(data_dir),
        "classes":        classes,
        "bias_warnings":  [],
        "ready":          False,
    }

    # ── Comptage + vérification intégrité ─────────────────────────────────────
    class_counts: dict[str, int] = {}
    integrity: dict[str, dict] = {}

    for cls in classes:
        imgs = _collect_images(data_dir / cls)
        class_counts[cls] = len(imgs)
        integrity[cls] = _sample_verify(imgs, max_check=150)

        if integrity[cls]["corruption_rate"] > 0:
            log.warning(
                f"Malaria/{cls} : {len(integrity[cls]['corrupted'])} images corrompues "
                f"({integrity[cls]['corruption_rate']*100:.1f}%)"
            )

    # ── Doublons (par nom de fichier) ─────────────────────────────────────────
    all_names = []
    for cls in classes:
        all_names += [f.name for f in _collect_images(data_dir / cls)]
    n_duplicates = len(all_names) - len(set(all_names))

    # ── Détection données synthétiques ───────────────────────────────────────
    sample_names = all_names[:20]
    is_synthetic = any(
        kw in n.lower() for n in sample_names
        for kw in ("cell_p_", "cell_u_", "synthetic", "fake", "generated")
    )

    # ── Biais géographiques (toujours applicable) ─────────────────────────────
    report["bias_warnings"] = [
        "⚠️ BIAIS GÉO : le dataset NIH/Kaggle contient principalement des cellules de "
        "patients du Bangladesh et de Colombie. Les morphologies plasmodiales africaines "
        "(P. falciparum dominant) peuvent différer — valider avec des images africaines.",
        "⚠️ BIAIS COLORIMÉTRIE : les conditions de coloration (Giemsa) varient selon les "
        "laboratoires africains. Risque de drift de distribution.",
    ]
    if is_synthetic:
        report["bias_warnings"].insert(
            0,
            "⚠️ DONNÉES SYNTHÉTIQUES détectées — performance réelle non garantie. "
            "Ne pas déployer sans validation sur données réelles."
        )

    # ── Assemblage rapport ────────────────────────────────────────────────────
    report["nb_samples"]  = sum(class_counts.values())
    report["class_counts"] = class_counts
    report["balance"]      = _balance_report(class_counts)
    report["integrity"]    = integrity
    report["duplicates"]   = n_duplicates
    report["synthetic"]    = is_synthetic

    if n_duplicates > 0:
        log.warning(f"Malaria : {n_duplicates} noms de fichiers dupliqués")

    ready = report["nb_samples"] >= 200 and all(v > 0 for v in class_counts.values())
    report["ready"] = ready

    status = "✓" if ready else "✗"
    log.info(f"  {status} Malaria — {report['nb_samples']} images | balance={report['balance']['balanced']}")
    for w in report["bias_warnings"]:
        log.warning(f"    {w}")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — NUTRITION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_nutrition() -> dict:
    """
    Valide le dataset nutrition.
    Contrôles : colonnes requises, valeurs manquantes, doublons,
    outliers OMS, balance des classes, biais genre/âge/géographique.
    """
    log.info("─── Validation Nutrition ───")
    csv_path = KANEA_ROOT / "data" / "nutrition" / "nutrition_dataset.csv"

    report: dict = {
        "module":        "nutrition",
        "source":        str(csv_path),
        "bias_warnings": [],
        "ready":         False,
    }

    if not csv_path.exists():
        log.error(f"Dataset nutrition introuvable : {csv_path}")
        report["error"] = "Fichier CSV introuvable"
        return report

    df = pd.read_csv(csv_path)
    report["nb_samples"] = len(df)
    report["columns"]    = list(df.columns)

    # ── Colonnes requises ─────────────────────────────────────────────────────
    required = ["age_months", "weight_kg", "height_cm", "nutrition_status"]
    missing_cols = [c for c in required if c not in df.columns]
    report["missing_columns"] = missing_cols
    if missing_cols:
        log.error(f"Colonnes manquantes : {missing_cols}")
        report["error"] = f"Colonnes manquantes : {missing_cols}"
        return report

    # ── Valeurs manquantes ────────────────────────────────────────────────────
    mv = df.isnull().sum()
    report["missing_values"] = {
        col: {"count": int(n), "pct": round(n / len(df) * 100, 1)}
        for col, n in mv.items() if n > 0
    }
    if report["missing_values"]:
        log.warning(f"Valeurs manquantes : {report['missing_values']}")

    # ── Doublons ─────────────────────────────────────────────────────────────
    n_dup = int(df.duplicated().sum())
    report["duplicates"] = n_dup
    if n_dup > 0:
        log.warning(f"Nutrition : {n_dup} lignes dupliquées")

    # ── Distribution des classes ──────────────────────────────────────────────
    cls_counts = df["nutrition_status"].value_counts().to_dict()
    report["class_distribution"] = {str(k): int(v) for k, v in cls_counts.items()}
    report["balance"] = _balance_report({str(k): int(v) for k, v in cls_counts.items()})

    # ── Outliers Z-score OMS ──────────────────────────────────────────────────
    zscore_cols = [c for c in ["whz", "haz", "waz"] if c in df.columns]
    outliers = {}
    for col in zscore_cols:
        lo, hi = WHO_ZSCORE_RANGE
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n_out > 0:
            outliers[col] = {
                "count": n_out,
                "pct":   round(n_out / len(df) * 100, 2),
                "action": f"Supprimer ou cap à [{lo}, {hi}]",
            }
            log.warning(f"Nutrition : {n_out} outliers {col} hors [{lo}, {hi}]")
    report["outliers_zscore"] = outliers

    # ── Cohérence physiologique poids/taille ─────────────────────────────────
    if "weight_kg" in df.columns and "height_cm" in df.columns:
        invalid_wh = int(((df["weight_kg"] < 1) | (df["weight_kg"] > 150) |
                          (df["height_cm"] < 40) | (df["height_cm"] > 250)).sum())
        if invalid_wh > 0:
            log.warning(f"Nutrition : {invalid_wh} valeurs poids/taille hors plage physiologique")
            report["invalid_anthropometry"] = invalid_wh

    # ── Biais genre ───────────────────────────────────────────────────────────
    if "sex" in df.columns:
        sex_dist = df["sex"].str.upper().value_counts().to_dict()
        report["sex_distribution"] = {str(k): int(v) for k, v in sex_dist.items()}
        vals = list(sex_dist.values())
        if len(vals) == 2:
            ratio_sex = max(vals) / max(min(vals), 1)
            if ratio_sex > 1.5:
                report["bias_warnings"].append(
                    f"⚠️ BIAIS GENRE : ratio M/F = {ratio_sex:.1f} — risque de biais dans prédiction "
                    f"(sous-nutrition affecte différemment garçons/filles)"
                )

    # ── Biais âge ────────────────────────────────────────────────────────────
    if "age_months" in df.columns:
        age_stats = df["age_months"].describe()
        report["age_distribution"] = {
            "min": round(float(age_stats["min"]), 1),
            "max": round(float(age_stats["max"]), 1),
            "mean": round(float(age_stats["mean"]), 1),
            "std":  round(float(age_stats["std"]), 1),
        }
        if age_stats["max"] < 12:
            report["bias_warnings"].append(
                "⚠️ BIAIS ÂGE : dataset ne contient pas d'enfants > 12 mois — "
                "modèle non généralisable aux 1-5 ans"
            )

    # ── Biais géographique ────────────────────────────────────────────────────
    report["bias_warnings"].append(
        "⚠️ BIAIS GÉO : standards OMS établis sur populations mixtes (MGRS, 8 pays). "
        "Les tables de référence peuvent ne pas refléter la variabilité génétique "
        "des populations sahéliennes et subsahariennes."
    )

    report["ready"] = report["nb_samples"] >= 100 and not missing_cols and not report.get("error")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — CANCER DU SEIN
# ═══════════════════════════════════════════════════════════════════════════════

def validate_breast_cancer() -> dict:
    """
    Valide le dataset mammographies.
    Contrôles : corruption, balance (3 classes), fichiers DICOM, biais racial.
    """
    log.info("─── Validation Cancer du sein ───")
    data_dir = KANEA_ROOT / "data" / "breast_cancer"
    classes  = ["Normal", "Benign", "Malignant"]

    report: dict = {
        "module":        "breast_cancer",
        "source_dir":    str(data_dir),
        "classes":       classes,
        "bias_warnings": [],
        "ready":         False,
    }

    class_counts: dict[str, int] = {}
    integrity: dict[str, dict]   = {}
    n_dicom_total = 0

    for cls in classes:
        cls_dir = data_dir / cls
        imgs = _collect_images(cls_dir)
        class_counts[cls] = len(imgs)
        integrity[cls]    = _sample_verify(imgs, max_check=100)

        # DICOM
        n_dicom = len(list(cls_dir.glob("**/*.dcm"))) if cls_dir.exists() else 0
        n_dicom_total += n_dicom

        if integrity[cls]["corruption_rate"] > 0:
            log.warning(
                f"BreastCancer/{cls} : {len(integrity[cls]['corrupted'])} images corrompues"
            )

    # Doublons par nom de fichier
    all_names = []
    for cls in classes:
        all_names += [f.name for f in _collect_images(data_dir / cls)]
    n_duplicates = len(all_names) - len(set(all_names))

    # Synthétique
    is_synthetic = any(
        cls.lower() in n.lower()
        for n in all_names[:20]
        for cls in ["benign_", "normal_", "malignant_"]
        if n.split("_")[0].lower() in ["benign", "normal", "malignant"]
    )

    # DICOM
    if n_dicom_total > 0:
        report["dicom_files_detected"] = n_dicom_total
        report["dicom_action"] = "Lancer preprocess_data() pour conversion automatique PNG"
        log.info(f"BreastCancer : {n_dicom_total} fichiers DICOM → conversion nécessaire")

    # Biais géographiques / raciaux
    report["bias_warnings"] = [
        "⚠️ BIAIS RACIAL : CBIS-DDSM (USA) et MIAS (UK) contiennent principalement des "
        "patientes caucasiennes. La densité mammaire africaine est statistiquement plus élevée "
        "(BI-RADS 3-4), ce qui affecte la sensibilité des modèles CNN.",
        "⚠️ BIAIS TECHNOLOGIQUE : les mammographies africaines sont souvent analogiques "
        "(résolution plus faible). Un modèle entraîné sur FFDM numérique peut dégrader.",
        "⚠️ BIAIS STADE : en Afrique subsaharienne, les cancers sont détectés à un stade plus "
        "avancé (stade III-IV dominant). L'entraînement sur stades précoces peut sous-performer.",
    ]
    if is_synthetic:
        report["bias_warnings"].insert(
            0,
            "⚠️ DONNÉES SYNTHÉTIQUES détectées — ne pas déployer sans validation clinique."
        )

    report["nb_samples"]   = sum(class_counts.values())
    report["class_counts"] = class_counts
    report["balance"]      = _balance_report(class_counts)
    report["integrity"]    = integrity
    report["duplicates"]   = n_duplicates
    report["synthetic"]    = is_synthetic

    report["ready"] = (
        report["nb_samples"] >= 50
        and all(v > 0 for v in class_counts.values())
    )

    status = "✓" if report["ready"] else "✗"
    log.info(
        f"  {status} Cancer sein — {report['nb_samples']} images | "
        f"balance={report['balance']['balanced']}"
    )
    for w in report["bias_warnings"]:
        log.warning(f"    {w}")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

@safe_run
def validate_data() -> dict:
    """
    Valide les 3 datasets (malaria, nutrition, cancer du sein).

    Contrôles effectués :
        ▸ Images corrompues (échantillon aléatoire)
        ▸ Doublons
        ▸ Déséquilibre de classes (seuils médicaux)
        ▸ Valeurs aberrantes physiologiques (Z-scores OMS)
        ▸ Cohérence des colonnes
        ▸ Biais géographiques / démographiques / raciaux

    Sorties :
        reports/validation_report.json

    Returns:
        dict complet avec rapport par module + recommandations.
    """
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   KANEA AI — VALIDATION DES DONNÉES                  ║")
    log.info(f"║   {timestamp()}                              ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    reports = {
        "malaria":       validate_malaria(),
        "nutrition":     validate_nutrition(),
        "breast_cancer": validate_breast_cancer(),
        "validated_at":  timestamp(),
    }

    # ── Résumé global ─────────────────────────────────────────────────────────
    log.info("─── Résumé validation ───")
    all_ready = True
    for module in ["malaria", "nutrition", "breast_cancer"]:
        r = reports[module]
        status = "✓ PRÊT    " if r.get("ready") else "✗ INCOMPLET"
        n = r.get("nb_samples", 0)
        balance = r.get("balance", {}).get("balanced", "—")
        log.info(f"  {status} | {module:15s} | {n:6d} échantillons | équilibré={balance}")
        if not r.get("ready"):
            all_ready = False

    reports["all_ready"]    = all_ready
    reports["total_samples"] = sum(
        reports[m].get("nb_samples", 0)
        for m in ["malaria", "nutrition", "breast_cancer"]
    )

    # ── Sauvegarde JSON ───────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "validation_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Rapport sauvegardé : {out}")

    return reports
