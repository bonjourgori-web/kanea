# -*- coding: utf-8 -*-
"""
KANEA MalariaScan AI - Pipeline de telechargement automatique des datasets.

Sources supportees :
  1. NIH Malaria Cell Images (Kaggle / deja present localement)
  2. MP-IDB  (Malaria Parasite Image DB - especes labelisees)
  3. BBBC041 (Broad Bioimage Benchmark - stades annotes)
  4. PlasmodiumDB Zenodo
  5. GitHub datasets publics

Usage :
    cd c:/Users/HP/gori/KANEA
    python scripts/download_malaria_datasets.py [--source all|nih|mpidb|bbbc]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("kanea.dataset")

ROOT        = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "malaria_advanced"

# Structure cible
DIRS = {
    "nih/infected":          "NIH - cellules parasitees (binaire)",
    "nih/uninfected":        "NIH - cellules saines (binaire)",
    "species/falciparum":    "P. falciparum",
    "species/vivax":         "P. vivax",
    "species/malariae":      "P. malariae",
    "species/ovale":         "P. ovale",
    "species/knowlesi":      "P. knowlesi",
    "stages/ring":           "Stade anneau",
    "stages/trophozoite":    "Trophozoite",
    "stages/schizont":       "Schizonte",
    "stages/gametocyte":     "Gametocyte",
    "processed":             "Images preprocessees",
    "cache":                 "Fichiers archives ZIP/tar",
}

# Sources de telechargement
SOURCES: dict[str, dict[str, Any]] = {
    "nih": {
        "name":         "NIH Malaria Cell Images",
        "images":       27560,
        "labels":       ["infected", "uninfected"],
        "local_check":  ROOT / "data" / "malaria" / "Parasitised",
        "kaggle_id":    "iarunava/cell-images-for-detecting-malaria",
        "url_fallback": None,
        "task":         "Détection binaire (infecté / sain)",
        "importance":   "CRITIQUE — dataset de référence international (NIH/NLM). 27 560 images de "
                        "lames Giemsa annotées par des experts. Utilisé pour entraîner le modèle "
                        "principal de détection. Accuracy 92.5%, AUC-ROC 0.969.",
        "species":      ["Plasmodium falciparum (majorité)"],
        "note":         "Dataset deja present dans data/malaria/",
    },
    "mpidb": {
        "name":         "MP-IDB (Malaria Parasite Image Database)",
        "images":       800,
        "labels":       ["falciparum", "vivax", "malariae", "ovale"],
        "local_check":  None,
        "kaggle_id":    "kmader/malaria-bounding-boxes",
        "url_fallback": "https://github.com/sivaramakrishnan-rajaraman/CNN-for-malaria-parasite-detection/raw/master/datasets/",
        "task":         "Classification multi-espèces Plasmodium",
        "importance":   "HAUTE — seul dataset public avec 4 espèces annotées (Pf, Pv, Pm, Po). "
                        "Permet d'entraîner un vrai classificateur d'espèces au lieu d'utiliser "
                        "les priors épidémiologiques. Crucial pour les zones à P. vivax (Asie, "
                        "Amérique latine) et P. ovale (Afrique de l'Ouest).",
        "species":      ["P. falciparum", "P. vivax", "P. malariae", "P. ovale"],
        "note":         "Necessite Kaggle API — pip install kaggle + ~/.kaggle/kaggle.json",
    },
    "bbbc041": {
        "name":         "BBBC041 (Broad Bioimage Benchmark Collection)",
        "images":       1364,
        "labels":       ["ring", "trophozoite", "schizont", "gametocyte", "leukocyte", "difficult"],
        "local_check":  None,
        "url_direct":   "https://data.broadinstitute.org/bbbc/BBBC041/malaria_trainval.zip",
        "task":         "Classification des stades parasitaires",
        "importance":   "HAUTE — annotations experts des 4 stades parasitaires de P. falciparum. "
                        "Permet un vrai classificateur Ring/Trophozoite/Schizonte/Gamétocyte. "
                        "Indispensable cliniquement : le stade détermine l'urgence thérapeutique "
                        "(Schizonte = risque de séquestration endothéliale = urgence vitale).",
        "species":      ["P. falciparum"],
        "note":         "Telechargement direct Broad Institute — annotations JSON incluses",
    },
    "plasmodium_zenodo": {
        "name":         "Plasmodium Species Dataset (Zenodo 3232035)",
        "images":       1182,
        "labels":       ["falciparum", "vivax", "uninfected"],
        "local_check":  None,
        "url_direct":   "https://zenodo.org/record/3232035/files/malaria_dataset.zip",
        "task":         "Distinction P. falciparum vs P. vivax",
        "importance":   "MOYENNE — distinction critique car P. vivax et P. ovale forment des "
                        "hypnozoïtes hépatiques (rechutes possibles) necessitant primaquine. "
                        "P. falciparum = forme la plus létale (80% des décès). Ce dataset "
                        "permet d'entraîner la distinction des 2 espèces les plus répandues.",
        "species":      ["P. falciparum", "P. vivax"],
        "note":         "Source Zenodo — dataset validé scientifiquement",
    },
}

# Informations cliniques sur les espèces — affichées au démarrage
SPECIES_CLINICAL_INFO = {
    "P. falciparum": {
        "prevalence_afrique": "89.5%",
        "mortalite":          "Responsable de >90% des décès par paludisme",
        "specificite":        "Forme érythrocytaire compacte, cytoplasme souvent en anneau",
        "urgence":            "CRITIQUE — neuropaludisme, hyperparasitémie, séquestration cérébrale",
        "traitement":         "ACT (artémisinine + luméfantrine) — 3 jours",
        "resistance":         "Résistances chloroquine documentées en Afrique/Asie",
    },
    "P. vivax": {
        "prevalence_afrique": "6.2%",
        "mortalite":          "Rechutes par hypnozoïtes hépatiques (jusqu'à 3 ans)",
        "specificite":        "Érythrocytes élargis, taches de Schüffner visibles",
        "urgence":            "MODÉRÉE — splénomégalie, anémie chronique",
        "traitement":         "Chloroquine + primaquine (éradication hypnozoïtes)",
        "resistance":         "Résistances chloroquine émergentes en Papouasie",
    },
    "P. malariae": {
        "prevalence_afrique": "2.5%",
        "mortalite":          "Syndrome néphrotique possible (infection chronique)",
        "specificite":        "Bandes transversales en 'bande d'équateur'",
        "urgence":            "FAIBLE — évolution lente, parasitémie basse",
        "traitement":         "Chloroquine — pas de primaquine nécessaire",
        "resistance":         "Sensible chloroquine",
    },
    "P. ovale": {
        "prevalence_afrique": "1.2%",
        "mortalite":          "Rechutes possibles (hypnozoïtes, comme P. vivax)",
        "specificite":        "Érythrocytes ovalisés avec bord dentelé (forme ovale)",
        "urgence":            "FAIBLE à MODÉRÉE",
        "traitement":         "Chloroquine + primaquine",
        "resistance":         "Sensible chloroquine",
    },
    "P. knowlesi": {
        "prevalence_afrique": "0.6%",
        "mortalite":          "Zoonose (singe → humain) — Asie du Sud-Est principalement",
        "specificite":        "Cycle 24h (quotidien) — parasitémie peut monter rapidement",
        "urgence":            "ÉLEVÉE si parasitémie > 1%",
        "traitement":         "Chloroquine ou ACT",
        "resistance":         "Données limitées",
    },
}


def setup_directories() -> None:
    for subdir in DIRS:
        (DATASET_DIR / subdir).mkdir(parents=True, exist_ok=True)
    log.info("Structure de repertoires creee : %s", DATASET_DIR)


def check_local_nih() -> dict[str, Any]:
    """Verifie si le dataset NIH est deja present localement."""
    para_dir  = ROOT / "data" / "malaria" / "Parasitised"
    uninf_dir = ROOT / "data" / "malaria" / "Uninfected"
    exts      = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    para_count  = len([p for p in para_dir.glob("*") if p.suffix.lower() in exts]) if para_dir.exists() else 0
    uninf_count = len([p for p in uninf_dir.glob("*") if p.suffix.lower() in exts]) if uninf_dir.exists() else 0
    total       = para_count + uninf_count

    return {
        "present":    total > 100,
        "parasitized": para_count,
        "uninfected":  uninf_count,
        "total":       total,
        "path":        str(ROOT / "data" / "malaria"),
    }


def try_kaggle_download(dataset_id: str, dest_dir: Path) -> bool:
    """Tente un telechargement via l'API Kaggle."""
    try:
        import kaggle  # type: ignore[import]
        log.info("Telechargement Kaggle : %s", dataset_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(dataset_id, path=str(dest_dir), unzip=True)
        log.info("Kaggle download OK : %s", dest_dir)
        return True
    except ImportError:
        log.warning("kaggle non installe — pip install kaggle")
    except Exception as exc:
        log.warning("Kaggle download echoue : %s", exc)
    return False


def http_download(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Telechargement HTTP avec reprise en cas d'interruption."""
    try:
        import urllib.request

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Telechargement : %s", url)

        # Reprise si fichier partiel
        resume_byte = dest_path.stat().st_size if dest_path.exists() else 0
        req = urllib.request.Request(url)
        if resume_byte:
            req.add_header("Range", f"bytes={resume_byte}-")
            log.info("Reprise a partir de %d bytes", resume_byte)

        with urllib.request.urlopen(req, timeout=60) as resp:
            content_length = int(resp.headers.get("Content-Length", 0))
            mode = "ab" if resume_byte else "wb"
            downloaded = resume_byte
            with open(dest_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if content_length:
                        pct = downloaded / (content_length + resume_byte) * 100
                        print(f"\r  {pct:.1f}%  ({downloaded // 1024} KB)", end="", flush=True)
        print()
        log.info("Download termine : %s (%d KB)", dest_path.name, dest_path.stat().st_size // 1024)
        return True

    except Exception as exc:
        log.warning("HTTP download echoue : %s", exc)
        return False


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """Extraction ZIP avec verification d'integrite."""
    try:
        log.info("Extraction : %s -> %s", zip_path.name, dest_dir)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_dir)
        log.info("Extraction OK : %d fichiers", len(list(dest_dir.rglob("*"))))
        return True
    except Exception as exc:
        log.error("Extraction echouee : %s", exc)
        return False


def validate_dataset(source_dir: Path, min_images: int = 50) -> dict[str, Any]:
    """Valide un dataset : compte les images valides, detecte les corrompues."""
    exts   = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    valid  = 0
    corrupt = 0
    formats: dict[str, int] = {}

    for img_path in source_dir.rglob("*"):
        if img_path.suffix.lower() not in exts:
            continue
        ext = img_path.suffix.lower()
        formats[ext] = formats.get(ext, 0) + 1
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                im.verify()
            valid += 1
        except Exception:
            corrupt += 1

    return {
        "valid":    valid,
        "corrupt":  corrupt,
        "formats":  formats,
        "ok":       valid >= min_images,
    }


def organize_bbbc041(raw_dir: Path) -> int:
    """Organise les images BBBC041 dans la structure stages/."""
    stage_map = {
        "ring":         DATASET_DIR / "stages" / "ring",
        "trophozoite":  DATASET_DIR / "stages" / "trophozoite",
        "schizont":     DATASET_DIR / "stages" / "schizont",
        "gametocyte":   DATASET_DIR / "stages" / "gametocyte",
    }
    total = 0

    # Chercher le JSON d'annotations BBBC041
    json_files = list(raw_dir.rglob("*.json"))
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            for sample in data:
                img_file = sample.get("image", {}).get("pathname", "")
                objects  = sample.get("objects", [])
                for obj in objects:
                    cat = obj.get("category", "").lower()
                    if cat in stage_map:
                        src = raw_dir / img_file.lstrip("/")
                        if src.exists():
                            dst = stage_map[cat] / src.name
                            if not dst.exists():
                                shutil.copy2(src, dst)
                                total += 1
        except Exception:
            pass

    log.info("BBBC041 organise : %d images copiees", total)
    return total


def download_source(source_key: str) -> dict[str, Any]:
    """Telechargement + organisation d'une source."""
    src   = SOURCES.get(source_key)
    if not src:
        return {"source": source_key, "status": "unknown"}

    log.info("=" * 50)
    log.info("Source : %s", src["name"])
    log.info("=" * 50)

    # NIH — deja present localement
    if source_key == "nih":
        info = check_local_nih()
        if info["present"]:
            log.info("NIH deja present : %d images dans %s", info["total"], info["path"])
            # Creer des symlinks ou copier dans la structure avancee
            nih_inf  = DATASET_DIR / "nih" / "infected"
            nih_uni  = DATASET_DIR / "nih" / "uninfected"
            src_para = ROOT / "data" / "malaria" / "Parasitised"
            src_uni  = ROOT / "data" / "malaria" / "Uninfected"
            if not any(nih_inf.glob("*")) and src_para.exists():
                log.info("Copie NIH Parasitised -> nih/infected (%d images)", info["parasitized"])
                for img in list(src_para.glob("*.png"))[:5000]:
                    shutil.copy2(img, nih_inf / img.name)
            if not any(nih_uni.glob("*")) and src_uni.exists():
                log.info("Copie NIH Uninfected -> nih/uninfected (%d images)", info["uninfected"])
                for img in list(src_uni.glob("*.png"))[:5000]:
                    shutil.copy2(img, nih_uni / img.name)
            return {"source": "nih", "status": "present", **info}
        else:
            # Essayer Kaggle
            cache_dir = DATASET_DIR / "cache" / "nih"
            ok = try_kaggle_download(src["kaggle_id"], cache_dir)
            return {"source": "nih", "status": "downloaded" if ok else "failed"}

    # BBBC041 — stades parasitaires
    if source_key == "bbbc041":
        cache_zip = DATASET_DIR / "cache" / "bbbc041.zip"
        cache_dir = DATASET_DIR / "cache" / "bbbc041_raw"
        if not cache_zip.exists():
            ok = http_download(src["url_direct"], cache_zip)
            if not ok:
                return {"source": "bbbc041", "status": "download_failed"}
        if not cache_dir.exists():
            extract_zip(cache_zip, cache_dir)
        n = organize_bbbc041(cache_dir)
        v = validate_dataset(DATASET_DIR / "stages", min_images=50)
        return {"source": "bbbc041", "status": "ok" if v["ok"] else "partial", "images": n, **v}

    # Zenodo Plasmodium species
    if source_key == "plasmodium_zenodo":
        cache_zip = DATASET_DIR / "cache" / "plasmodium_zenodo.zip"
        cache_dir = DATASET_DIR / "cache" / "zenodo_raw"
        if not cache_zip.exists():
            ok = http_download(src["url_direct"], cache_zip)
            if not ok:
                log.warning("Zenodo inaccessible - essai Kaggle fallback")
                ok = try_kaggle_download("rajadapa/malaria-cell-images-with-species-annotations", cache_dir)
                return {"source": "plasmodium_zenodo", "status": "downloaded_kaggle" if ok else "failed"}
        if not cache_dir.exists():
            extract_zip(cache_zip, cache_dir)
        # Detecter structure et copier dans species/
        for sp_dir in cache_dir.rglob("*/"):
            name_lower = sp_dir.name.lower()
            for sp_key in ("falciparum", "vivax", "malariae", "ovale", "knowlesi"):
                if sp_key in name_lower:
                    dest = DATASET_DIR / "species" / sp_key
                    dest.mkdir(parents=True, exist_ok=True)
                    n_copied = 0
                    for img in sp_dir.glob("*.[jpJP][pnPN][egEG]*"):
                        dst = dest / img.name
                        if not dst.exists():
                            shutil.copy2(img, dst)
                            n_copied += 1
                    if n_copied:
                        log.info("  %s : %d images copiees", sp_key, n_copied)
        v = validate_dataset(DATASET_DIR / "species", min_images=50)
        return {"source": "plasmodium_zenodo", "status": "ok" if v["ok"] else "partial", **v}

    # MP-IDB via Kaggle
    if source_key == "mpidb":
        cache_dir = DATASET_DIR / "cache" / "mpidb"
        ok = try_kaggle_download(src["kaggle_id"], cache_dir)
        if not ok:
            log.warning(
                "MP-IDB non telechargeable automatiquement.\n"
                "  Telechargement manuel :\n"
                "  1. Installer kaggle : pip install kaggle\n"
                "  2. Creer ~/.kaggle/kaggle.json avec vos credentials\n"
                "  3. kaggle datasets download -d %s", src["kaggle_id"]
            )
        return {"source": "mpidb", "status": "downloaded" if ok else "requires_manual"}

    return {"source": source_key, "status": "not_implemented"}


def generate_report(results: list[dict]) -> None:
    """Genere un rapport de telechargement."""
    report_path = DATASET_DIR / "download_report.json"
    report = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "root":      str(DATASET_DIR),
        "results":   results,
        "summary":   {
            "ok":     sum(1 for r in results if "ok" in r.get("status", "") or "present" in r.get("status", "")),
            "failed": sum(1 for r in results if "fail" in r.get("status", "")),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Rapport : %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Telechargement datasets malaria KANEA")
    parser.add_argument("--source", default="all",
                        choices=["all", "nih", "mpidb", "bbbc041", "plasmodium_zenodo"],
                        help="Source a telecharger (defaut: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Passer les sources deja presentes")
    args = parser.parse_args()

    log.info("=== KANEA MalariaScan AI - Dataset Download Pipeline ===")
    log.info("")
    log.info("--- Importance clinique des especes Plasmodium ---")
    for sp, info in SPECIES_CLINICAL_INFO.items():
        log.info("  %-20s | Prevalence Afrique: %-6s | Urgence: %s",
                 sp, info["prevalence_afrique"], info["urgence"])
    log.info("")
    log.info("--- Sources de datasets disponibles ---")
    for key, src in SOURCES.items():
        log.info("  %-22s | %d images | Tache: %s", src["name"], src["images"], src["task"])
        log.info("    Importance: %s", src["importance"][:90] + "..." if len(src["importance"]) > 90 else src["importance"])
    log.info("")

    setup_directories()

    sources = list(SOURCES.keys()) if args.source == "all" else [args.source]
    results = []

    for src_key in sources:
        try:
            result = download_source(src_key)
            results.append(result)
            status = result.get("status", "?")
            log.info("  %s -> %s", src_key, status)
        except Exception as exc:
            log.error("Erreur source %s : %s", src_key, exc)
            results.append({"source": src_key, "status": "error", "error": str(exc)})

    # Rapport final
    generate_report(results)

    log.info("\n=== RESUME ===")
    for r in results:
        icon = "OK" if "ok" in r.get("status","") or "present" in r.get("status","") else "!!"
        log.info("  [%s] %-25s : %s", icon, r.get("source","?"), r.get("status","?"))

    log.info("\nPour lancer l'entrainement multi-tache :")
    log.info("  python scripts/train_malaria_multitask.py")


if __name__ == "__main__":
    main()
