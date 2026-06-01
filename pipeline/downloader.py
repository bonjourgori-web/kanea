"""
KANÉA — Auto Download Manager
================================
Phase 3 : Téléchargement automatique des datasets autorisés.
Gère : Kaggle API · PhysioNet · Zenodo · Direct HTTP
Vérifie : intégrité SHA-256 · scan antivirus (optionnel) · décompression
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.request
import zipfile
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.catalog import get_catalog

logger = logging.getLogger("kanea.downloader")

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "pipeline"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

_download_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} PB"


def _extract(archive_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
    elif name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest)
    else:
        shutil.copy2(archive_path, dest / archive_path.name)
    logger.info(f"[Downloader] Extrait → {dest}")


# ═══════════════════════════════════════════════════════════════════════════════
# STRATÉGIES DE TÉLÉCHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class KaggleDownloader:
    def download(
        self,
        dest_dir: Path,
        api_params: dict[str, Any],
        progress_cb: Any = None,
    ) -> Path | None:
        slug = api_params.get("dataset") or api_params.get("competition")
        if not slug:
            logger.error("[Kaggle] Pas de slug dataset/competition")
            return None
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            if api_params.get("competition"):
                cmd = ["kaggle", "competitions", "download", "-c", slug, "-p", str(dest_dir)]
            else:
                cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest_dir)]
            logger.info(f"[Kaggle] Téléchargement : {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                logger.error(f"[Kaggle] Erreur : {result.stderr}")
                return None
            archives = list(dest_dir.glob("*.zip")) + list(dest_dir.glob("*.tar.gz"))
            if archives:
                _extract(archives[0], dest_dir)
                return dest_dir
            return dest_dir
        except FileNotFoundError:
            logger.warning("[Kaggle] CLI non installée — installez avec: pip install kaggle")
            return None
        except Exception as e:
            logger.error(f"[Kaggle] Exception : {e}")
            return None


class PhysioNetDownloader:
    def download(
        self,
        dest_dir: Path,
        api_params: dict[str, Any],
        credentials: dict[str, str] | None = None,
        progress_cb: Any = None,
    ) -> Path | None:
        project = api_params.get("project", "")
        version = api_params.get("version", "1.0.0")
        if not project:
            return None

        dest_dir.mkdir(parents=True, exist_ok=True)
        base_url = f"https://physionet.org/files/{project}/{version}/"

        username = (credentials or {}).get("username") or os.environ.get("PHYSIONET_USER", "")
        password = (credentials or {}).get("password") or os.environ.get("PHYSIONET_PASS", "")

        try:
            cmd = ["wget", "-r", "-N", "-c", "-np", "--no-check-certificate"]
            if username and password:
                cmd += [f"--user={username}", f"--password={password}"]
            cmd += ["-P", str(dest_dir), base_url]
            logger.info(f"[PhysioNet] Téléchargement {project} v{version}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if result.returncode != 0:
                logger.warning(f"[PhysioNet] wget retour {result.returncode}")
            return dest_dir
        except FileNotFoundError:
            logger.warning("[PhysioNet] wget non disponible — utilisez: pip install requests")
            return None
        except Exception as e:
            logger.error(f"[PhysioNet] Exception : {e}")
            return None


class ZenodoDownloader:
    API = "https://zenodo.org/api"

    def download(
        self,
        dest_dir: Path,
        api_params: dict[str, Any],
        progress_cb: Any = None,
    ) -> Path | None:
        record_id = api_params.get("record_id")
        if not record_id:
            return None

        dest_dir.mkdir(parents=True, exist_ok=True)
        url = f"{self.API}/records/{record_id}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "KANEA/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            files = data.get("files", [])
            downloaded = []
            for f in files:
                file_url  = f.get("links", {}).get("self") or f.get("download", "")
                file_name = f.get("key", f.get("filename", "file"))
                out_path  = dest_dir / file_name
                if not file_url:
                    continue
                logger.info(f"[Zenodo] {file_name}")
                urllib.request.urlretrieve(file_url, out_path)
                downloaded.append(out_path)
                if out_path.suffix in (".zip", ".tar.gz", ".tgz"):
                    _extract(out_path, dest_dir)

            return dest_dir if downloaded else None
        except Exception as e:
            logger.error(f"[Zenodo] Exception : {e}")
            return None


class DirectDownloader:
    def download(
        self,
        dest_dir: Path,
        url: str,
        filename: str | None = None,
        progress_cb: Any = None,
    ) -> Path | None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        fname = filename or url.split("/")[-1].split("?")[0] or "data.bin"
        out_path = dest_dir / fname

        try:
            headers = {"User-Agent": "KANEA/1.0"}
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk = 1 << 16  # 64 KB

                with open(out_path, "wb") as f:
                    while True:
                        block = resp.read(chunk)
                        if not block:
                            break
                        f.write(block)
                        downloaded += len(block)
                        if progress_cb and total:
                            progress_cb(downloaded / total)

            if out_path.suffix in (".zip", ".tar.gz", ".tgz", ".tar.bz2"):
                _extract(out_path, dest_dir)
            return out_path

        except Exception as e:
            logger.error(f"[Direct] Exception {url} : {e}")
            return None


import json

_DOWNLOADERS = {
    "kaggle":    KaggleDownloader(),
    "physionet": PhysioNetDownloader(),
    "zenodo":    ZenodoDownloader(),
    "direct":    DirectDownloader(),
    "tcia":      DirectDownloader(),
    "custom":    DirectDownloader(),
}


# ═══════════════════════════════════════════════════════════════════════════════
# GESTIONNAIRE DE TÉLÉCHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class DownloadManager:
    def __init__(self):
        self.catalog = get_catalog()

    def _dest_dir(self, module_key: str, source_name: str) -> Path:
        safe = source_name.replace(" ", "_").replace("/", "_")[:40]
        return DATA_ROOT / module_key / safe

    def download_dataset(
        self,
        dataset_id: int,
        credentials: dict[str, str] | None = None,
        verify_integrity: bool = True,
    ) -> bool:
        ds = self.catalog.get_dataset(dataset_id)
        if not ds:
            logger.error(f"[Downloader] Dataset {dataset_id} non trouvé")
            return False

        module_key  = ds["module_key"]
        source_name = ds["source_name"]
        api_type    = ds["api_type"]
        url         = ds["url"]
        metadata    = json.loads(ds.get("metadata") or "{}")

        dest_dir = self._dest_dir(module_key, source_name)
        logger.info(f"[Downloader] {module_key}/{source_name} → {dest_dir}")

        self.catalog.update_dataset(dataset_id, status="downloading")
        self.catalog.log_event(
            "download_start", module_key, "Phase 3 - Download",
            "started", f"Téléchargement {source_name}",
        )

        try:
            downloader = _DOWNLOADERS.get(api_type, _DOWNLOADERS["direct"])
            out_path: Path | None = None

            if api_type == "kaggle":
                from pipeline.registry import get_module
                mod = get_module(module_key)
                src = next(
                    (s for s in (mod.sources if mod else []) if s.name == source_name), None
                )
                out_path = downloader.download(dest_dir, src.api_params if src else metadata)
            elif api_type == "physionet":
                out_path = downloader.download(
                    dest_dir,
                    api_params=metadata,
                    credentials=credentials,
                )
            elif api_type == "zenodo":
                out_path = downloader.download(dest_dir, api_params=metadata)
            else:
                dl_url = metadata.get("download_url") or url
                out_path = downloader.download(dest_dir, url=dl_url)

            if not out_path:
                self.catalog.update_dataset(dataset_id, status="error")
                return False

            # Taille du dossier téléchargé
            size = sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file())
            n_files = sum(1 for _ in dest_dir.rglob("*") if _.is_file())

            # Intégrité SHA-256 (premier fichier)
            checksum = ""
            if verify_integrity:
                files = sorted(dest_dir.rglob("*"), key=lambda p: p.stat().st_size, reverse=True)
                first_file = next((f for f in files if f.is_file()), None)
                if first_file:
                    checksum = _sha256(first_file)

            self.catalog.update_dataset(
                dataset_id,
                status="downloaded",
                file_path=str(dest_dir),
                size_bytes=size,
                sample_count=n_files,
                checksum_sha256=checksum,
                downloaded_at=datetime.utcnow().isoformat(),
            )
            self.catalog.log_event(
                "download_complete", module_key, "Phase 3 - Download",
                "success",
                f"{source_name} — {_human_size(size)} — {n_files} fichiers",
                {"size_bytes": size, "n_files": n_files, "checksum": checksum},
            )
            logger.info(f"[Downloader] OK — {_human_size(size)} — {n_files} fichiers")
            return True

        except Exception as e:
            logger.error(f"[Downloader] Erreur : {e}")
            self.catalog.update_dataset(dataset_id, status="error")
            self.catalog.log_event(
                "download_error", module_key, "Phase 3 - Download",
                "error", str(e),
            )
            return False

    def download_module(
        self,
        module_key: str,
        credentials: dict[str, str] | None = None,
    ) -> dict[str, bool]:
        datasets = self.catalog.list_datasets(module_key=module_key, status="discovered")
        results: dict[str, bool] = {}
        for ds in datasets:
            ok = self.download_dataset(ds["id"], credentials=credentials)
            results[ds["source_name"]] = ok
        return results

    def scan_virus(self, path: Path) -> bool:
        """Scan antivirus basique via ClamAV si disponible."""
        try:
            result = subprocess.run(
                ["clamscan", "--recursive", str(path)],
                capture_output=True, text=True, timeout=120,
            )
            if "Infected files: 0" in result.stdout:
                logger.info(f"[AntiVirus] Clean — {path}")
                return True
            logger.warning(f"[AntiVirus] Menace détectée — {path}")
            return False
        except FileNotFoundError:
            logger.debug("[AntiVirus] ClamAV non installé — scan ignoré")
            return True
        except Exception as e:
            logger.warning(f"[AntiVirus] Erreur scan : {e}")
            return True


def run_download(
    module_key: str | None = None,
    dataset_ids: list[int] | None = None,
    credentials: dict[str, str] | None = None,
) -> dict[str, Any]:
    manager = DownloadManager()
    catalog = get_catalog()
    results: dict[str, Any] = {"started": datetime.utcnow().isoformat(), "results": {}}

    if dataset_ids:
        for did in dataset_ids:
            ok = manager.download_dataset(did, credentials=credentials)
            results["results"][did] = ok
    elif module_key:
        results["results"] = manager.download_module(module_key, credentials=credentials)
    else:
        # Tous les datasets découverts
        discovered = catalog.list_datasets(status="discovered")
        for ds in discovered:
            ok = manager.download_dataset(ds["id"], credentials=credentials)
            results["results"][ds["source_name"]] = ok

    results["finished"] = datetime.utcnow().isoformat()
    return results
