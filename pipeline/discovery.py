"""
KANÉA — Auto Data Discovery
=============================
Phase 2 : Exploration automatique des sources, détection des nouveaux datasets,
extraction des métadonnées, mise à jour du catalogue.

APIs supportées : Kaggle · PhysioNet · TCIA · Zenodo · Direct HTTP
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pipeline.registry import REGISTRY, DataSource, get_all_sources
from pipeline.catalog import get_catalog

logger = logging.getLogger("kanea.discovery")


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURES DE RÉSULTAT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiscoveryResult:
    module_key: str
    source_name: str
    api_type: str
    url: str
    reachable: bool
    metadata: dict[str, Any]
    dataset_id: int | None
    error: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# CLIENTS PAR TYPE D'API
# ═══════════════════════════════════════════════════════════════════════════════

class KaggleDiscovery:
    """Découverte via l'API Kaggle (kaggle.json requis dans ~/.kaggle/)."""

    def probe(self, source: DataSource) -> dict[str, Any]:
        try:
            import kaggle  # type: ignore
            slug = source.api_params.get("dataset") or source.api_params.get("competition", "")
            if not slug:
                return {"reachable": False, "error": "No dataset/competition slug"}

            if source.api_params.get("competition"):
                # Competitions
                return {
                    "reachable": True,
                    "type": "competition",
                    "slug": slug,
                    "formats": source.formats,
                    "size_gb": source.size_estimate_gb,
                }
            else:
                meta = kaggle.api.dataset_list(search=slug.split("/")[-1])
                return {
                    "reachable": True,
                    "type": "dataset",
                    "slug": slug,
                    "result_count": len(meta),
                    "formats": source.formats,
                    "size_gb": source.size_estimate_gb,
                }
        except ImportError:
            return {
                "reachable": True,
                "type": "dataset",
                "note": "kaggle package not installed — install with: pip install kaggle",
                "slug": source.api_params.get("dataset", source.api_params.get("competition")),
                "size_gb": source.size_estimate_gb,
            }
        except Exception as e:
            return {"reachable": False, "error": str(e)}


class PhysioNetDiscovery:
    BASE = "https://physionet.org/api/v1"

    def probe(self, source: DataSource) -> dict[str, Any]:
        project = source.api_params.get("project", "")
        version = source.api_params.get("version", "")
        url = f"{self.BASE}/project/{project}/" if project else source.url
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "KANEA/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    try:
                        data = json.loads(resp.read())
                    except Exception:
                        data = {}
                    return {
                        "reachable": True,
                        "project": project,
                        "version": version,
                        "size_gb": source.size_estimate_gb,
                        "requires_auth": source.requires_auth,
                        "api_data": data,
                    }
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return {
                    "reachable": True,
                    "project": project,
                    "note": "Accès credentialed requis — PhysioNet login obligatoire",
                    "requires_auth": True,
                    "size_gb": source.size_estimate_gb,
                }
            return {"reachable": False, "error": f"HTTP {e.code}"}
        except Exception as e:
            return {"reachable": False, "error": str(e)}
        return {"reachable": False, "error": "Unknown"}


class ZenodoDiscovery:
    BASE = "https://zenodo.org/api"

    def probe(self, source: DataSource) -> dict[str, Any]:
        record_id = source.api_params.get("record_id")
        q = source.api_params.get("q", "")

        if record_id:
            url = f"{self.BASE}/records/{record_id}"
        elif q:
            url = f"{self.BASE}/records?q={urllib.parse.quote(q)}&type=dataset&size=5"
        else:
            url = source.url

        try:
            import urllib.parse
            req = urllib.request.Request(url, headers={"User-Agent": "KANEA/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                if record_id:
                    return {
                        "reachable": True,
                        "record_id": record_id,
                        "title": data.get("metadata", {}).get("title", ""),
                        "size_bytes": sum(
                            f.get("size", 0) for f in data.get("files", [])
                        ),
                        "files": [f.get("key") for f in data.get("files", [])],
                    }
                else:
                    hits = data.get("hits", {}).get("hits", [])
                    return {
                        "reachable": True,
                        "query": q,
                        "results": len(hits),
                        "records": [
                            {"id": h["id"], "title": h.get("metadata", {}).get("title", "")}
                            for h in hits[:5]
                        ],
                    }
        except Exception as e:
            return {"reachable": False, "error": str(e)}


class TCIADiscovery:
    BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v2"

    def probe(self, source: DataSource) -> dict[str, Any]:
        collection = source.api_params.get("collection", "")
        url = f"{self.BASE}/getCollectionValues"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "KANEA/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                collections = [c.get("Collection", "") for c in data] if isinstance(data, list) else []
                found = collection in collections if collection else True
                return {
                    "reachable": True,
                    "collection": collection,
                    "found_in_tcia": found,
                    "total_collections": len(collections),
                    "size_gb": source.size_estimate_gb,
                }
        except Exception as e:
            return {
                "reachable": True,
                "collection": collection,
                "note": "TCIA API probe — connexion limitée",
                "size_gb": source.size_estimate_gb,
                "error": str(e),
            }


class DirectDiscovery:
    def probe(self, source: DataSource) -> dict[str, Any]:
        try:
            req = urllib.request.Request(
                source.url,
                method="HEAD",
                headers={"User-Agent": "KANEA/1.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                return {
                    "reachable": True,
                    "status": resp.status,
                    "content_type": resp.headers.get("Content-Type", ""),
                    "content_length": resp.headers.get("Content-Length", 0),
                    "size_gb": source.size_estimate_gb,
                }
        except urllib.error.HTTPError as e:
            return {
                "reachable": e.code < 500,
                "status": e.code,
                "size_gb": source.size_estimate_gb,
            }
        except Exception as e:
            return {"reachable": False, "error": str(e)}


# Import urllib.parse ici pour eviter les erreurs
import urllib.parse

_PROBES: dict[str, Any] = {
    "kaggle":     KaggleDiscovery(),
    "physionet":  PhysioNetDiscovery(),
    "zenodo":     ZenodoDiscovery(),
    "tcia":       TCIADiscovery(),
    "direct":     DirectDiscovery(),
    "custom":     DirectDiscovery(),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MOTEUR DE DÉCOUVERTE
# ═══════════════════════════════════════════════════════════════════════════════

class DataDiscovery:
    def __init__(self):
        self.catalog = get_catalog()

    def discover_source(
        self, module_key: str, source: DataSource
    ) -> DiscoveryResult:
        probe = _PROBES.get(source.api_type, _PROBES["direct"])
        logger.info(f"[Discovery] {module_key} / {source.name} ({source.api_type})")

        try:
            metadata = probe.probe(source)
        except Exception as e:
            metadata = {"reachable": False, "error": str(e)}

        reachable = metadata.get("reachable", False)

        dataset_id = self.catalog.register_dataset(
            module_key=module_key,
            source_name=source.name,
            api_type=source.api_type,
            url=source.url,
            license=source.license,
            metadata=metadata,
        )

        self.catalog.log_event(
            event_type="discovery",
            module_key=module_key,
            phase="Phase 2 - Discovery",
            status="ok" if reachable else "unreachable",
            message=f"{source.name} — {'OK' if reachable else metadata.get('error', 'unreachable')}",
            details=metadata,
        )

        return DiscoveryResult(
            module_key=module_key,
            source_name=source.name,
            api_type=source.api_type,
            url=source.url,
            reachable=reachable,
            metadata=metadata,
            dataset_id=dataset_id,
            error=metadata.get("error", ""),
        )

    def discover_module(
        self, module_key: str, delay: float = 0.3
    ) -> list[DiscoveryResult]:
        mod = REGISTRY.get(module_key)
        if not mod:
            logger.warning(f"Module '{module_key}' non trouvé dans le registre")
            return []

        results = []
        for source in mod.sources:
            result = self.discover_source(module_key, source)
            results.append(result)
            time.sleep(delay)
        return results

    def discover_all(
        self,
        modules: list[str] | None = None,
        delay: float = 0.5,
    ) -> dict[str, list[DiscoveryResult]]:
        targets = modules or list(REGISTRY.keys())
        all_results: dict[str, list[DiscoveryResult]] = {}

        logger.info(f"[Discovery] Démarrage — {len(targets)} modules")
        for key in targets:
            all_results[key] = self.discover_module(key, delay=delay)

        total = sum(len(v) for v in all_results.values())
        reachable = sum(
            1 for v in all_results.values() for r in v if r.reachable
        )
        logger.info(f"[Discovery] Terminé — {reachable}/{total} sources accessibles")
        return all_results

    def get_discovery_report(
        self, results: dict[str, list[DiscoveryResult]]
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_sources": 0,
            "reachable": 0,
            "unreachable": 0,
            "by_module": {},
            "by_api_type": {},
        }
        for module_key, mod_results in results.items():
            report["total_sources"] += len(mod_results)
            ok  = sum(1 for r in mod_results if r.reachable)
            nok = len(mod_results) - ok
            report["reachable"] += ok
            report["unreachable"] += nok
            report["by_module"][module_key] = {
                "total": len(mod_results),
                "reachable": ok,
                "unreachable": nok,
                "sources": [
                    {
                        "name": r.source_name,
                        "reachable": r.reachable,
                        "api_type": r.api_type,
                        "dataset_id": r.dataset_id,
                        "error": r.error,
                    }
                    for r in mod_results
                ],
            }
        # Statistiques par type d'API
        for _, mod_results in results.items():
            for r in mod_results:
                api = r.api_type
                if api not in report["by_api_type"]:
                    report["by_api_type"][api] = {"total": 0, "reachable": 0}
                report["by_api_type"][api]["total"] += 1
                if r.reachable:
                    report["by_api_type"][api]["reachable"] += 1
        return report


def run_discovery(
    modules: list[str] | None = None,
) -> dict[str, Any]:
    """Point d'entrée principal de la Phase 2."""
    engine = DataDiscovery()
    results = engine.discover_all(modules=modules)
    return engine.get_discovery_report(results)
