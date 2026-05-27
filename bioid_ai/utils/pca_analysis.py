"""
pca_analysis.py — Analyse PCA sur les marqueurs ancestraux (AIMs).

Fonctions :
- fit_aims_pca() : entraîne PCA + StandardScaler sur les données AIMs brutes
- transform_aims() : projette de nouvelles valeurs brutes en coordonnées PCA
- plot_pca_variance() : graphique de la variance expliquée
- plot_pca_scatter() : scatter plot des échantillons dans l'espace PCA
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    log.warning("scikit-learn non disponible — fonctions PCA désactivées")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    log.warning("matplotlib/seaborn non disponibles — graphiques PCA désactivés")


# Palette médicale professionnelle
_COLORS = ["#20B2AA", "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]


def fit_aims_pca(
    aims_data: np.ndarray,
    n_components: int = 3,
) -> "tuple[PCA, StandardScaler]":
    """
    Entraîne PCA + StandardScaler sur les données AIMs brutes.

    Paramètres
    ----------
    aims_data : np.ndarray, shape (n_samples, n_aims_markers)
        Données brutes des marqueurs AIMs.
    n_components : int
        Nombre de composantes PCA (défaut 3).

    Retourne
    --------
    tuple(PCA, StandardScaler) — les deux objets ajustés.
    """
    if not _SKLEARN_OK:
        raise ImportError("scikit-learn est requis pour fit_aims_pca()")

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(aims_data)

    n_comp = min(n_components, aims_data.shape[1], aims_data.shape[0])
    pca = PCA(n_components=n_comp)
    pca.fit(data_scaled)

    log.info(
        "fit_aims_pca → n_components=%d, variance expliquée=%s",
        n_comp,
        np.round(pca.explained_variance_ratio_, 3),
    )
    return pca, scaler


def transform_aims(
    aims_raw: list[float],
    pca: "PCA",
    scaler: "StandardScaler",
) -> dict[str, float]:
    """
    Projette les marqueurs AIMs bruts en coordonnées PCA.

    Paramètres
    ----------
    aims_raw : list[float]
        Valeurs brutes des marqueurs AIMs (même ordre qu'à l'entraînement).
    pca : PCA
        Objet PCA ajusté.
    scaler : StandardScaler
        Scaleur ajusté.

    Retourne
    --------
    dict avec clés AIM_PC1, AIM_PC2, AIM_PC3 (et plus si n_components > 3).
    """
    if not _SKLEARN_OK:
        raise ImportError("scikit-learn est requis pour transform_aims()")

    try:
        arr = np.array(aims_raw, dtype=float).reshape(1, -1)
        scaled = scaler.transform(arr)
        coords = pca.transform(scaled)[0]

        result: dict[str, float] = {}
        for i, val in enumerate(coords):
            result[f"AIM_PC{i + 1}"] = float(round(val, 6))

        log.debug("transform_aims → %s", result)
        return result

    except Exception as exc:
        log.error("transform_aims error: %s", exc)
        return {"AIM_PC1": 0.0, "AIM_PC2": 0.0, "AIM_PC3": 0.0}


def plot_pca_variance(
    pca: "PCA",
    save_path: Path,
    title: str = "Variance expliquée — AIM PCA",
) -> str:
    """
    Génère un graphique de la variance expliquée par chaque composante PCA.

    Retourne le chemin du fichier PNG sauvegardé, ou "" si erreur.
    """
    if not _MPL_OK or not _SKLEARN_OK:
        log.warning("plot_pca_variance : dépendances manquantes")
        return ""

    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        variance = pca.explained_variance_ratio_
        cumvar = np.cumsum(variance)
        x = np.arange(1, len(variance) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, variance * 100, color=_COLORS[0], alpha=0.8, label="Variance par composante")
        ax.plot(x, cumvar * 100, "o-", color=_COLORS[1], linewidth=2, label="Variance cumulée")
        ax.axhline(y=80, color="gray", linestyle="--", alpha=0.5, label="Seuil 80%")

        ax.set_xlabel("Composante PCA", fontsize=12)
        ax.set_ylabel("Variance expliquée (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"PC{i}" for i in x])
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info("plot_pca_variance sauvegardé : %s", save_path)
        return str(save_path)

    except Exception as exc:
        log.error("plot_pca_variance error: %s", exc)
        return ""


def plot_pca_scatter(
    pca_coords: np.ndarray,
    labels: list[str],
    save_path: Path,
    title: str = "Scatter PCA — Marqueurs AIM",
    pc_x: int = 0,
    pc_y: int = 1,
) -> str:
    """
    Génère un scatter plot des échantillons dans l'espace PCA.

    Paramètres
    ----------
    pca_coords : np.ndarray, shape (n_samples, n_components)
    labels : list[str]
        Labels de classe pour coloriser les points.
    save_path : Path
    title : str
    pc_x, pc_y : int
        Indices des composantes à afficher sur x et y.

    Retourne le chemin du fichier sauvegardé, ou "" si erreur.
    """
    if not _MPL_OK:
        log.warning("plot_pca_scatter : matplotlib non disponible")
        return ""

    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        unique_labels = sorted(set(labels))
        color_map = {lbl: _COLORS[i % len(_COLORS)] for i, lbl in enumerate(unique_labels)}
        colors_pts = [color_map[lbl] for lbl in labels]

        fig, ax = plt.subplots(figsize=(9, 7))
        scatter = ax.scatter(
            pca_coords[:, pc_x],
            pca_coords[:, pc_y],
            c=colors_pts,
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidths=0.5,
        )

        # Légende manuelle
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=color_map[lbl], markersize=10, label=lbl
            )
            for lbl in unique_labels
        ]
        ax.legend(handles=handles, title="Groupe", fontsize=10)

        ax.set_xlabel(f"PC{pc_x + 1}", fontsize=12)
        ax.set_ylabel(f"PC{pc_y + 1}", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info("plot_pca_scatter sauvegardé : %s", save_path)
        return str(save_path)

    except Exception as exc:
        log.error("plot_pca_scatter error: %s", exc)
        return ""
