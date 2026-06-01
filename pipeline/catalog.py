"""
KANÉA — Data Catalog Manager
=============================
Gestion centralisée des datasets via SQLite.
Enregistre : découverte, téléchargement, qualité, entraînement, versions modèles.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "catalog.db"
_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHÉMA DE BASE DE DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    module_key      TEXT    NOT NULL,
    source_name     TEXT    NOT NULL,
    api_type        TEXT    NOT NULL,
    url             TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'discovered',
    -- discovered | downloading | downloaded | preprocessing | ready | error
    file_path       TEXT,
    size_bytes      INTEGER DEFAULT 0,
    sample_count    INTEGER DEFAULT 0,
    quality_score   REAL    DEFAULT 0.0,
    checksum_sha256 TEXT,
    license         TEXT,
    discovered_at   TEXT    NOT NULL,
    downloaded_at   TEXT,
    preprocessed_at TEXT,
    metadata        TEXT    DEFAULT '{}',
    UNIQUE(module_key, source_name)
);

CREATE TABLE IF NOT EXISTS training_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    module_key      TEXT    NOT NULL,
    run_id          TEXT    NOT NULL UNIQUE,
    status          TEXT    NOT NULL DEFAULT 'pending',
    -- pending | running | completed | failed | validated
    started_at      TEXT,
    finished_at     TEXT,
    dataset_ids     TEXT    DEFAULT '[]',
    framework       TEXT,
    architecture    TEXT,
    epochs          INTEGER DEFAULT 0,
    best_epoch      INTEGER DEFAULT 0,
    metrics         TEXT    DEFAULT '{}',
    model_path      TEXT,
    model_version   TEXT,
    triggered_by    TEXT    DEFAULT 'manual',
    -- manual | new_data | performance_drop | scheduled
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS knowledge_updates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL,
    -- pubmed | who | nccn | asco | esmo | fda | cdc
    title           TEXT    NOT NULL,
    url             TEXT,
    published_date  TEXT,
    modules_affected TEXT   DEFAULT '[]',
    summary         TEXT,
    applied         INTEGER DEFAULT 0,
    fetched_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS quality_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER REFERENCES datasets(id),
    module_key      TEXT    NOT NULL,
    score           REAL    NOT NULL,
    n_total         INTEGER DEFAULT 0,
    n_valid         INTEGER DEFAULT 0,
    n_corrupted     INTEGER DEFAULT 0,
    n_duplicates    INTEGER DEFAULT 0,
    n_missing_labels INTEGER DEFAULT 0,
    issues          TEXT    DEFAULT '[]',
    created_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT    NOT NULL,
    module_key  TEXT,
    phase       TEXT,
    status      TEXT    NOT NULL,
    message     TEXT,
    details     TEXT    DEFAULT '{}',
    created_at  TEXT    NOT NULL
);
"""


# ═══════════════════════════════════════════════════════════════════════════════
# GESTIONNAIRE DE CATALOGUE
# ═══════════════════════════════════════════════════════════════════════════════

class DataCatalog:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with _lock, self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ── DATASETS ─────────────────────────────────────────────────────────────

    def register_dataset(
        self,
        module_key: str,
        source_name: str,
        api_type: str,
        url: str,
        license: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = datetime.utcnow().isoformat()
        with _lock, self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO datasets
                   (module_key, source_name, api_type, url, license, discovered_at, metadata)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(module_key, source_name) DO UPDATE SET
                   url=excluded.url, discovered_at=excluded.discovered_at
                """,
                (module_key, source_name, api_type, url, license, now,
                 json.dumps(metadata or {})),
            )
            return cur.lastrowid

    def update_dataset(self, dataset_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [dataset_id]
        with _lock, self._conn() as conn:
            conn.execute(f"UPDATE datasets SET {cols} WHERE id=?", vals)

    def get_dataset(self, dataset_id: int) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
            return dict(row) if row else None

    def list_datasets(
        self,
        module_key: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM datasets WHERE 1=1"
        params: list[Any] = []
        if module_key:
            q += " AND module_key=?"; params.append(module_key)
        if status:
            q += " AND status=?"; params.append(status)
        q += " ORDER BY discovered_at DESC"
        with self._conn() as conn:
            rows = conn.execute(q, params).fetchall()
            return [dict(r) for r in rows]

    def get_ready_datasets(self, module_key: str) -> list[dict[str, Any]]:
        return self.list_datasets(module_key=module_key, status="ready")

    # ── ENTRAÎNEMENTS ─────────────────────────────────────────────────────────

    def create_training_run(
        self,
        module_key: str,
        run_id: str,
        framework: str,
        architecture: str,
        dataset_ids: list[int],
        triggered_by: str = "manual",
    ) -> int:
        now = datetime.utcnow().isoformat()
        with _lock, self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO training_runs
                   (module_key, run_id, framework, architecture, dataset_ids,
                    triggered_by, started_at, status)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (module_key, run_id, framework, architecture,
                 json.dumps(dataset_ids), triggered_by, now, "pending"),
            )
            return cur.lastrowid

    def update_training_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [run_id]
        with _lock, self._conn() as conn:
            conn.execute(f"UPDATE training_runs SET {cols} WHERE run_id=?", vals)

    def get_best_run(self, module_key: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM training_runs
                   WHERE module_key=? AND status='validated'
                   ORDER BY finished_at DESC LIMIT 1""",
                (module_key,),
            ).fetchone()
            return dict(row) if row else None

    def list_runs(self, module_key: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM training_runs"
        params: list[Any] = []
        if module_key:
            q += " WHERE module_key=?"; params.append(module_key)
        q += " ORDER BY started_at DESC"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(q, params).fetchall()]

    # ── QUALITÉ ──────────────────────────────────────────────────────────────

    def save_quality_report(
        self,
        dataset_id: int,
        module_key: str,
        score: float,
        n_total: int,
        n_valid: int,
        n_corrupted: int,
        n_duplicates: int,
        n_missing_labels: int,
        issues: list[str],
    ) -> None:
        now = datetime.utcnow().isoformat()
        with _lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO quality_reports
                   (dataset_id, module_key, score, n_total, n_valid, n_corrupted,
                    n_duplicates, n_missing_labels, issues, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (dataset_id, module_key, score, n_total, n_valid, n_corrupted,
                 n_duplicates, n_missing_labels, json.dumps(issues), now),
            )

    # ── CONNAISSANCES ─────────────────────────────────────────────────────────

    def save_knowledge_update(
        self,
        source: str,
        title: str,
        url: str,
        published_date: str,
        modules_affected: list[str],
        summary: str,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with _lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO knowledge_updates
                   (source, title, url, published_date, modules_affected, summary, fetched_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (source, title, url, published_date,
                 json.dumps(modules_affected), summary, now),
            )

    def list_knowledge_updates(
        self, applied: bool | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM knowledge_updates"
        params: list[Any] = []
        if applied is not None:
            q += " WHERE applied=?"; params.append(1 if applied else 0)
        q += f" ORDER BY fetched_at DESC LIMIT {limit}"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(q, params).fetchall()]

    # ── ÉVÉNEMENTS ────────────────────────────────────────────────────────────

    def log_event(
        self,
        event_type: str,
        status: str,
        message: str = "",
        module_key: str = "",
        phase: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with _lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO pipeline_events
                   (event_type, module_key, phase, status, message, details, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (event_type, module_key, phase, status, message,
                 json.dumps(details or {}), now),
            )

    def list_events(
        self,
        module_key: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM pipeline_events"
        params: list[Any] = []
        if module_key:
            q += " WHERE module_key=?"; params.append(module_key)
        q += f" ORDER BY created_at DESC LIMIT {limit}"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(q, params).fetchall()]

    # ── STATISTIQUES ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        with self._conn() as conn:
            total_ds = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
            ready_ds = conn.execute("SELECT COUNT(*) FROM datasets WHERE status='ready'").fetchone()[0]
            total_runs = conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
            validated = conn.execute(
                "SELECT COUNT(*) FROM training_runs WHERE status='validated'"
            ).fetchone()[0]
            knowledge = conn.execute("SELECT COUNT(*) FROM knowledge_updates").fetchone()[0]
            events = conn.execute("SELECT COUNT(*) FROM pipeline_events").fetchone()[0]
        return {
            "datasets_total": total_ds,
            "datasets_ready": ready_ds,
            "training_runs_total": total_runs,
            "training_runs_validated": validated,
            "knowledge_updates": knowledge,
            "pipeline_events": events,
        }


# Singleton
_catalog: DataCatalog | None = None


def get_catalog() -> DataCatalog:
    global _catalog
    if _catalog is None:
        _catalog = DataCatalog()
    return _catalog
