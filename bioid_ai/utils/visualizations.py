"""
visualizations.py — Fonctions de visualisation médicale professionnelle BioID AI.

Toutes les fonctions :
- Sauvegardent en PNG HD (300 dpi)
- Utilisent une palette médicale professionnelle (#20B2AA teal, etc.)
- Gèrent les exceptions et retournent le chemin du fichier sauvegardé
- Retournent "" si erreur
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    import seaborn as sns
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    log.warning("matplotlib/seaborn non disponibles — visualisations désactivées")

try:
    import shap as _shap_module
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False

# ── Palette médicale professionnelle ─────────────────────────────────────────
PALETTE = {
    "teal":       "#20B2AA",
    "coral":      "#FF6B6B",
    "cyan":       "#4ECDC4",
    "blue":       "#45B7D1",
    "green":      "#96CEB4",
    "yellow":     "#FFEAA7",
    "purple":     "#6C5CE7",
    "orange":     "#FDCB6E",
    "dark_teal":  "#148F85",
    "dark_gray":  "#2D3436",
    "light_gray": "#DFE6E9",
    "white":      "#FFFFFF",
}

COLOR_LIST = [
    PALETTE["teal"], PALETTE["coral"], PALETTE["cyan"],
    PALETTE["blue"], PALETTE["green"], PALETTE["purple"],
    PALETTE["orange"],
]


def _setup_style() -> None:
    """Configure le style matplotlib pour tous les graphiques."""
    if not _MPL_OK:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.facecolor": PALETTE["white"],
        "figure.facecolor": PALETTE["white"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "grid.color": PALETTE["light_gray"],
    })


def _save_and_close(fig: Any, save_path: Path) -> str:
    """Sauvegarde la figure et la ferme proprement."""
    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info("Graphique sauvegardé : %s", save_path)
        return str(save_path)
    except Exception as exc:
        log.error("_save_and_close error: %s", exc)
        plt.close("all")
        return ""


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    labels: list[str],
    title: str = "Matrice de Confusion",
    save_path: Optional[Path] = None,
) -> str:
    """
    Génère une matrice de confusion annotée et colorée.

    Retourne le chemin PNG sauvegardé, ou "" si erreur.
    """
    if not _MPL_OK:
        return ""
    if save_path is None:
        save_path = Path("visualizations/confusion_matrix.png")

    try:
        from sklearn.metrics import confusion_matrix
        _setup_style()

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_norm, annot=cm, fmt="d",
            cmap=sns.light_palette(PALETTE["teal"], as_cmap=True),
            xticklabels=labels, yticklabels=labels,
            ax=ax, linewidths=0.5, linecolor=PALETTE["light_gray"],
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, pad=15)
        ax.set_xlabel("Prédiction", fontsize=12)
        ax.set_ylabel("Vérité terrain", fontsize=12)
        plt.tight_layout()
        return _save_and_close(fig, save_path)

    except Exception as exc:
        log.error("plot_confusion_matrix error: %s", exc)
        plt.close("all")
        return ""


def plot_roc_curves(
    y_true: list,
    y_proba: Any,
    class_names: list[str],
    title: str = "Courbes ROC",
    save_path: Optional[Path] = None,
) -> str:
    """
    Génère les courbes ROC multi-classes (One-vs-Rest).

    Retourne le chemin PNG sauvegardé, ou "" si erreur.
    """
    if not _MPL_OK:
        return ""
    if save_path is None:
        save_path = Path("visualizations/roc_curves.png")

    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        _setup_style()
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        y_proba_arr = np.array(y_proba)

        fig, ax = plt.subplots(figsize=(9, 7))

        for i, cls_name in enumerate(class_names):
            if y_bin.shape[1] > 1:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba_arr[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=COLOR_LIST[i % len(COLOR_LIST)],
                        lw=2, label=f"{cls_name} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Aléatoire")
        ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.02, color=PALETTE["teal"])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
        ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=12)
        ax.set_title(title, pad=15)
        ax.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        return _save_and_close(fig, save_path)

    except Exception as exc:
        log.error("plot_roc_curves error: %s", exc)
        plt.close("all")
        return ""


def plot_feature_importance(
    importances: dict[str, float],
    title: str = "Importance des Features",
    save_path: Optional[Path] = None,
    top_n: int = 20,
) -> str:
    """
    Génère un graphique d'importance des features (barh).

    Paramètres
    ----------
    importances : dict {feature_name: importance_value}
    top_n : int — affiche les N features les plus importantes

    Retourne le chemin PNG sauvegardé, ou "" si erreur.
    """
    if not _MPL_OK:
        return ""
    if save_path is None:
        save_path = Path("visualizations/feature_importance.png")

    try:
        _setup_style()

        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names = [x[0] for x in sorted_items]
        vals  = [x[1] for x in sorted_items]

        fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.35 + 2)))

        # Dégradé de couleur
        colors = [
            PALETTE["teal"] if v >= np.percentile(vals, 70) else
            (PALETTE["cyan"] if v >= np.percentile(vals, 40) else PALETTE["blue"])
            for v in vals
        ]

        bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="white", height=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (Gini / MDI)", fontsize=12)
        ax.set_title(title, pad=15)

        # Valeurs annotées
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_width() + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9, color=PALETTE["dark_gray"],
            )

        plt.tight_layout()
        return _save_and_close(fig, save_path)

    except Exception as exc:
        log.error("plot_feature_importance error: %s", exc)
        plt.close("all")
        return ""


def plot_shap_summary(
    shap_values: Any,
    feature_names: list[str],
    save_path: Optional[Path] = None,
    title: str = "SHAP — Résumé des contributions",
) -> str:
    """
    Génère un graphique de résumé SHAP (beeswarm ou barplot fallback).

    Retourne le chemin PNG sauvegardé, ou "" si erreur.
    """
    if not _MPL_OK:
        return ""
    if save_path is None:
        save_path = Path("visualizations/shap_summary.png")

    try:
        _setup_style()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if _SHAP_OK and hasattr(_shap_module, "summary_plot"):
            fig, ax = plt.subplots(figsize=(10, 7))
            _shap_module.summary_plot(
                shap_values,
                feature_names=feature_names,
                show=False,
                plot_type="bar",
                color=PALETTE["teal"],
            )
            fig = plt.gcf()
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        else:
            # Fallback : barplot manuel si shap non disponible
            if isinstance(shap_values, np.ndarray) and shap_values.ndim >= 2:
                mean_abs = np.abs(shap_values).mean(axis=0)
            elif isinstance(shap_values, np.ndarray):
                mean_abs = np.abs(shap_values)
            else:
                mean_abs = np.ones(len(feature_names))

            sorted_idx = np.argsort(mean_abs)[-20:]
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(
                [feature_names[i] for i in sorted_idx],
                mean_abs[sorted_idx],
                color=PALETTE["teal"], alpha=0.85,
            )
            ax.set_xlabel("|Valeur SHAP| moyenne", fontsize=12)
            ax.set_title(title, pad=15)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close("all")
        log.info("plot_shap_summary sauvegardé : %s", save_path)
        return str(save_path)

    except Exception as exc:
        log.error("plot_shap_summary error: %s", exc)
        plt.close("all")
        return ""


def plot_biological_radar(
    predictions: dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Profil Biologique — Radar",
) -> str:
    """
    Génère un radar chart du profil biologique forensique.

    predictions doit contenir des scores normalisés [0–1] pour chaque dimension.
    Retourne le chemin PNG sauvegardé, ou "" si erreur.
    """
    if not _MPL_OK:
        return ""
    if save_path is None:
        save_path = Path("visualizations/biological_radar.png")

    try:
        _setup_style()

        # Dimensions et valeurs (normalisées 0-1)
        dims_raw = {
            "Sexe\n(confiance)":        predictions.get("sex_confidence", 0.75),
            "Âge\n(précision)":         1.0 - min(predictions.get("age_uncertainty", 0.3), 1.0),
            "Ascendance\n(confiance)":  predictions.get("ancestry_confidence", 0.70),
            "Stature\n(disponible)":    1.0 if predictions.get("stature_cm") else 0.0,
            "Données\ncrâniales":       predictions.get("cranial_completeness", 0.80),
            "Données\npost-crâniales":  predictions.get("postcranial_completeness", 0.60),
        }

        labels = list(dims_raw.keys())
        values = [max(0.0, min(1.0, float(v))) for v in dims_raw.values()]

        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles_plot = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.plot(angles_plot, values_plot, "o-", linewidth=2.5, color=PALETTE["teal"])
        ax.fill(angles_plot, values_plot, alpha=0.25, color=PALETTE["teal"])

        # Cercles de référence
        for r in [0.25, 0.5, 0.75, 1.0]:
            ax.plot(
                np.linspace(0, 2 * np.pi, 100),
                [r] * 100,
                "-", color=PALETTE["light_gray"], linewidth=0.8, alpha=0.5,
            )

        ax.set_xticks(angles)
        ax.set_xticklabels(labels, size=11, fontweight="bold")
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=9)
        ax.set_ylim(0, 1)
        ax.set_title(title, y=1.08, fontsize=15, fontweight="bold", color=PALETTE["dark_gray"])
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        # Points sur le radar
        for angle, val, lbl in zip(angles, values, labels):
            ax.plot(angle, val, "o", color=PALETTE["teal"], markersize=10, zorder=5)

        plt.tight_layout()
        return _save_and_close(fig, save_path)

    except Exception as exc:
        log.error("plot_biological_radar error: %s", exc)
        plt.close("all")
        return ""


def plot_prediction_summary(
    predictions: dict[str, Any],
    confidences: dict[str, float],
    save_path: Optional[Path] = None,
    title: str = "Résumé des Prédictions BioID",
) -> str:
    """
    Génère un graphique résumé visuel du profil biologique complet.

    Inclut : sexe, âge, ascendance, stature avec barres de confiance.
    Retourne le chemin PNG sauvegardé, ou "" si erreur.
    """
    if not _MPL_OK:
        return ""
    if save_path is None:
        save_path = Path("visualizations/prediction_summary.png")

    try:
        _setup_style()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold",
                     color=PALETTE["dark_gray"], y=1.01)

        # ── 1. Sexe biologique ──────────────────────────────────────────────
        ax1 = axes[0, 0]
        sex = str(predictions.get("biological_sex", "Inconnu"))
        sex_conf = float(confidences.get("sex_confidence", 0.0))
        categories = ["Masculin", "Féminin"]
        if "male" in sex.lower() and "fe" not in sex.lower():
            values_sex = [sex_conf, 1 - sex_conf]
        elif "fem" in sex.lower() or "woman" in sex.lower():
            values_sex = [1 - sex_conf, sex_conf]
        else:
            values_sex = [0.5, 0.5]
        bars1 = ax1.bar(categories, values_sex,
                        color=[PALETTE["blue"], PALETTE["coral"]], edgecolor="white", width=0.5)
        for bar, v in zip(bars1, values_sex):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:.1%}", ha="center", fontsize=12, fontweight="bold")
        ax1.set_ylim(0, 1.15)
        ax1.set_title("Sexe Biologique", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Probabilité")
        ax1.tick_params(axis="x", labelsize=12)

        # ── 2. Âge au décès ────────────────────────────────────────────────
        ax2 = axes[0, 1]
        age = float(predictions.get("age_at_death", 0))
        age_min = max(0, age - 8)
        age_max = age + 8
        ax2.barh(["Estimation"], [age_max - age_min], left=[age_min],
                 color=PALETTE["teal"], alpha=0.35, edgecolor=PALETTE["teal"], height=0.4)
        ax2.axvline(age, color=PALETTE["teal"], linewidth=3, label=f"Médiane: {age:.0f} ans")
        ax2.scatter([age], [0], color=PALETTE["teal"], s=150, zorder=5)
        ax2.set_xlim(max(0, age_min - 10), age_max + 10)
        ax2.set_title("Âge au Décès (années)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Âge (ans)")
        ax2.legend(fontsize=11)
        ax2.set_facecolor("white")
        # Annotation intervalle
        ax2.text(age, 0.22, f"[{age_min:.0f} – {age_max:.0f}]",
                 ha="center", fontsize=11, color=PALETTE["dark_gray"])

        # ── 3. Ascendance ──────────────────────────────────────────────────
        ax3 = axes[1, 0]
        ancestry_probs = predictions.get("ancestry_probabilities", {})
        if not ancestry_probs:
            anc = str(predictions.get("ancestry", "Inconnu"))
            anc_conf = float(confidences.get("ancestry_confidence", 0.0))
            ancestry_probs = {anc: anc_conf}
            if anc_conf < 1.0:
                ancestry_probs["Autre"] = round(1.0 - anc_conf, 3)

        anc_labels = list(ancestry_probs.keys())
        anc_vals   = [float(v) for v in ancestry_probs.values()]
        bars3 = ax3.bar(anc_labels, anc_vals,
                        color=COLOR_LIST[:len(anc_labels)], edgecolor="white", width=0.5)
        for bar, v in zip(bars3, anc_vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")
        ax3.set_ylim(0, 1.15)
        ax3.set_title("Ascendance Biogéographique", fontsize=13, fontweight="bold")
        ax3.set_ylabel("Probabilité")

        # ── 4. Stature ─────────────────────────────────────────────────────
        ax4 = axes[1, 1]
        stature = float(predictions.get("stature_cm", 0))
        method  = str(predictions.get("stature_method", "Trotter & Gleser"))
        if stature > 0:
            stature_range = np.linspace(stature - 10, stature + 10, 200)
            from scipy.stats import norm as _norm
            density = _norm.pdf(stature_range, loc=stature, scale=3.5)
            ax4.fill_between(stature_range, density, alpha=0.35, color=PALETTE["green"])
            ax4.plot(stature_range, density, color=PALETTE["green"], linewidth=2)
            ax4.axvline(stature, color=PALETTE["dark_teal"], linewidth=2.5,
                        label=f"Stature: {stature:.1f} cm")
            ax4.set_xlim(stature - 15, stature + 15)
            ax4.legend(fontsize=11)
        else:
            ax4.text(0.5, 0.5, "Données insuffisantes",
                     ha="center", va="center", transform=ax4.transAxes,
                     fontsize=13, color=PALETTE["coral"])
        ax4.set_title(f"Stature Estimée ({method})", fontsize=13, fontweight="bold")
        ax4.set_xlabel("Taille (cm)")

        for ax in axes.flat:
            ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        plt.tight_layout(pad=2.5)
        return _save_and_close(fig, save_path)

    except Exception as exc:
        log.error("plot_prediction_summary error: %s", exc)
        plt.close("all")
        return ""
