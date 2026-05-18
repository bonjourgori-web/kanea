"""
KANÉA — Tableau de bord épidémiologique · Côte d'Ivoire
═════════════════════════════════════════════════════════
Fonctions :
  load_health_data()   → DataFrame synthétique réaliste (CIV, 2021-2023)
  render_stats_tab()   → onglet Streamlit complet

Données : synthétiques mais épidémiologiquement cohérentes avec les patterns
          OMS / PNLP Côte d'Ivoire (paludisme endémique, saisonnalité tropicale).

⚠️ Outil d'aide à la décision — pas un diagnostic officiel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─── Constantes géographiques CIV ──────────────────────────────────────────────

REGIONS = {
    # ── Zone Sud ────────────────────────────────────────────────────────────────
    "Abidjan":              {"zone": "sud",    "population": 5_500_000},
    "Bas-Sassandra":        {"zone": "sud",    "population": 1_800_000},
    "Lagunes":              {"zone": "sud",    "population": 700_000},
    "Agnéby":               {"zone": "sud",    "population": 850_000},
    # ── Zone Centre ─────────────────────────────────────────────────────────────
    "Vallée du Bandama":    {"zone": "centre", "population": 2_100_000},
    "Sassandra-Marahoué":   {"zone": "centre", "population": 1_600_000},
    "Haut Sassandra":       {"zone": "centre", "population": 1_300_000},
    "Gôh-Djiboua":          {"zone": "centre", "population": 1_100_000},
    "Marahoué":             {"zone": "centre", "population": 800_000},
    "Lacs":                 {"zone": "centre", "population": 800_000},
    "N'Zi Comoé":           {"zone": "centre", "population": 700_000},
    # ── Zone Nord ───────────────────────────────────────────────────────────────
    "Savanes":              {"zone": "nord",   "population": 1_400_000},
    "Worodougou":           {"zone": "nord",   "population": 700_000},
    "Denguélé":             {"zone": "nord",   "population": 600_000},
    # ── Zone Ouest ──────────────────────────────────────────────────────────────
    "Montagnes":            {"zone": "ouest",  "population": 1_200_000},
    "Moyen Cavally":        {"zone": "ouest",  "population": 600_000},
    # ── Zone Est ────────────────────────────────────────────────────────────────
    "Comoé":                {"zone": "est",    "population": 900_000},
    "Moyen Comoé":          {"zone": "est",    "population": 500_000},
    "Sud Comoé":            {"zone": "est",    "population": 700_000},
    "Zanzan":               {"zone": "est",    "population": 600_000},
}

MALADIES = ["Paludisme", "Malnutrition (MAS)", "Malnutrition (MAM)", "Tuberculose", "Diarrhée infantile"]

# Couleurs KANÉA
COLORS = {
    "Paludisme":            "#E74C3C",
    "Malnutrition (MAS)":   "#F39C12",
    "Malnutrition (MAM)":   "#F7DC6F",
    "Tuberculose":          "#8E44AD",
    "Diarrhée infantile":   "#3498DB",
}

RISK_COLORS = {
    "Très élevé": "#C0392B",
    "Élevé":      "#E67E22",
    "Modéré":     "#F1C40F",
    "Faible":     "#27AE60",
}


# ═══════════════════════════════════════════════════════════════════════════════
# GÉNÉRATION DE DONNÉES SYNTHÉTIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def load_health_data() -> pd.DataFrame:
    """
    Génère un DataFrame épidémiologique synthétique pour la Côte d'Ivoire.

    Cohérence épidémiologique :
      ▸ Paludisme endémique — pic saisons des pluies (avr-juil, sept-nov)
      ▸ Plus intense dans le Sud (zones humides)
      ▸ MAS/MAM plus élevés dans le Nord (soudano-sahélien)
      ▸ Incidence calculée sur population réelle par région
      ▸ 3 ans de données mensuelles (2021-2023) → 36 mois × 10 régions × 5 maladies
    """
    rng  = np.random.default_rng(42)
    rows = []

    dates = pd.date_range("2021-01", "2023-12", freq="MS")

    for date in dates:
        mois = date.month

        # Saison des pluies CIV : avr-jul (grande) et sept-nov (petite)
        saison_pluie = mois in (4, 5, 6, 7, 9, 10, 11)
        facteur_saison = 1.6 if saison_pluie else 0.7

        for region, meta in REGIONS.items():
            pop      = meta["population"]
            zone     = meta["zone"]

            # ── Paludisme ────────────────────────────────────────────────────
            base_pal = {
                "sud": 280, "centre": 230, "ouest": 210,
                "nord": 160, "est": 190,
            }[zone]
            cas_pal = int(
                rng.negative_binomial(
                    n=10,
                    p=0.035,
                    size=1
                )[0] * base_pal * facteur_saison
            )
            cas_pal = max(10, cas_pal)

            # ── Malnutrition (MAS) ───────────────────────────────────────────
            base_mas = {"nord": 180, "est": 140, "ouest": 120, "centre": 90, "sud": 60}[zone]
            facteur_mas = 1.3 if mois in (1, 2, 3, 11, 12) else 0.9  # pic saison sèche
            cas_mas = max(2, int(rng.poisson(base_mas * facteur_mas)))

            # ── Malnutrition (MAM) ───────────────────────────────────────────
            cas_mam = max(5, int(cas_mas * rng.uniform(2.5, 3.5)))

            # ── Tuberculose ──────────────────────────────────────────────────
            cas_tb = max(1, int(rng.poisson(35 if zone in ("sud", "centre") else 20)))

            # ── Diarrhée infantile ────────────────────────────────────────────
            cas_dia = max(5, int(rng.poisson(120 * facteur_saison)))

            for maladie, cas in [
                ("Paludisme",          cas_pal),
                ("Malnutrition (MAS)", cas_mas),
                ("Malnutrition (MAM)", cas_mam),
                ("Tuberculose",        cas_tb),
                ("Diarrhée infantile", cas_dia),
            ]:
                incidence = round(cas / pop * 100_000, 2)
                deces     = int(rng.binomial(cas, {"Paludisme": 0.005, "Malnutrition (MAS)": 0.04,
                                                    "Malnutrition (MAM)": 0.01, "Tuberculose": 0.08,
                                                    "Diarrhée infantile": 0.003}.get(maladie, 0.005)))
                rows.append({
                    "date":            date,
                    "annee":           date.year,
                    "mois":            date.month,
                    "mois_label":      date.strftime("%b %Y"),
                    "region":          region,
                    "zone":            zone,
                    "maladie":         maladie,
                    "cas":             cas,
                    "deces":           deces,
                    "incidence":       incidence,  # pour 100 000 habitants
                    "population":      pop,
                    "saison_pluie":    saison_pluie,
                })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSANTS GRAPHIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def _kpis(df: pd.DataFrame) -> None:
    """Affiche les indicateurs clés en haut du tableau de bord."""
    total_cas    = df["cas"].sum()
    total_deces  = df["deces"].sum()
    incidence_moy = df["incidence"].mean()
    region_max   = df.groupby("region")["cas"].sum().idxmax()
    maladie_dom  = df.groupby("maladie")["cas"].sum().idxmax()
    taux_letalite = round(total_deces / max(total_cas, 1) * 100, 2)

    cols = st.columns(5)
    with cols[0]:
        st.metric("🦠 Total cas", f"{total_cas:,}".replace(",", " "))
    with cols[1]:
        st.metric("💀 Décès", f"{total_deces:,}".replace(",", " "), f"létalité {taux_letalite}%")
    with cols[2]:
        st.metric("📈 Incidence moy.", f"{incidence_moy:.1f} / 100k")
    with cols[3]:
        st.metric("📍 Région la + touchée", region_max)
    with cols[4]:
        st.metric("🔴 Maladie dominante", maladie_dom)


def _chart_temporal(df: pd.DataFrame, maladie: str, region: str) -> go.Figure:
    """Courbe temporelle cas vs temps."""
    mask = pd.Series([True] * len(df))
    if maladie != "Toutes":
        mask &= df["maladie"] == maladie
    if region != "Toutes":
        mask &= df["region"] == region

    agg = df[mask].groupby(["date", "maladie"])["cas"].sum().reset_index()

    fig = px.line(
        agg, x="date", y="cas", color="maladie",
        title=f"Évolution temporelle des cas — {maladie} · {region}",
        labels={"cas": "Nombre de cas", "date": "Date", "maladie": "Maladie"},
        color_discrete_map=COLORS,
        markers=True,
    )
    # Zones saison des pluies
    for annee in [2021, 2022, 2023]:
        for debut, fin in [
            (f"{annee}-04-01", f"{annee}-07-31"),
            (f"{annee}-09-01", f"{annee}-11-30"),
        ]:
            fig.add_vrect(
                x0=debut, x1=fin,
                fillcolor="lightblue", opacity=0.12,
                annotation_text="🌧️ Pluies", annotation_position="top left",
                line_width=0,
            )
    fig.update_layout(hovermode="x unified", template="plotly_white")
    return fig


def _chart_regions(df: pd.DataFrame, maladie: str) -> go.Figure:
    """Histogramme cas par région."""
    mask = df["maladie"] == maladie if maladie != "Toutes" else pd.Series([True] * len(df))
    agg  = df[mask].groupby("region")[["cas", "deces", "incidence"]].sum().reset_index()
    agg  = agg.sort_values("cas", ascending=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=agg["cas"], y=agg["region"],
            orientation="h", name="Cas totaux",
            marker_color="#E74C3C", opacity=0.85,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=agg["incidence"] / agg["incidence"].max() * agg["cas"].max(),
            y=agg["region"],
            mode="markers+lines", name="Incidence (normalisée)",
            marker=dict(color="#3498DB", size=10),
        ),
        secondary_y=False,
    )
    fig.update_layout(
        title=f"Cas et incidence par région — {maladie}",
        xaxis_title="Cas",
        template="plotly_white",
        barmode="overlay",
    )
    return fig


def _chart_comparatif(df: pd.DataFrame) -> go.Figure:
    """Graphique comparatif des maladies par région (heatmap)."""
    pivot = df.groupby(["region", "maladie"])["incidence"].mean().unstack(fill_value=0)

    fig = px.imshow(
        pivot,
        title="Incidence moyenne (/ 100k hab.) par région et maladie",
        labels=dict(x="Maladie", y="Région", color="Incidence / 100k"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
        text_auto=".1f",
    )
    fig.update_layout(template="plotly_white")
    return fig


def _chart_repartition(df: pd.DataFrame, region: str) -> go.Figure:
    """Camembert répartition des maladies."""
    mask = df["region"] == region if region != "Toutes" else pd.Series([True] * len(df))
    agg  = df[mask].groupby("maladie")["cas"].sum().reset_index()

    fig = px.pie(
        agg, names="maladie", values="cas",
        title=f"Répartition des cas — {region}",
        color="maladie", color_discrete_map=COLORS,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template="plotly_white")
    return fig


def _chart_zones_risque(df: pd.DataFrame, maladie: str) -> go.Figure:
    """Classement des régions par niveau de risque."""
    mask   = df["maladie"] == maladie if maladie != "Toutes" else pd.Series([True] * len(df))
    agg    = df[mask].groupby("region")["incidence"].mean().reset_index()
    p75    = agg["incidence"].quantile(0.75)
    p50    = agg["incidence"].quantile(0.50)
    p25    = agg["incidence"].quantile(0.25)

    def _niveau(v: float) -> str:
        if v >= p75: return "Très élevé"
        if v >= p50: return "Élevé"
        if v >= p25: return "Modéré"
        return "Faible"

    agg["risque"] = agg["incidence"].apply(_niveau)
    agg = agg.sort_values("incidence", ascending=False)

    fig = px.bar(
        agg, x="region", y="incidence",
        color="risque",
        color_discrete_map=RISK_COLORS,
        title=f"Zones à risque — {maladie} (incidence / 100k hab.)",
        labels={"incidence": "Incidence moy. / 100k", "region": "Région"},
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=-30)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# RENDU STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600)
def _cached_health_data() -> pd.DataFrame:
    return load_health_data()


def render_stats_tab() -> None:
    """
    Onglet complet « Tableau de bord épidémiologique ».
    Appeler depuis dashboard/app.py à l'intérieur d'un st.tab().
    """
    st.markdown(
        "### 📊 Tableau de bord épidémiologique — Côte d'Ivoire\n"
        "> Données synthétiques réalistes · Sources : OMS / PNLP CIV / UNICEF  \n"
        "> ⚠️ Outil d'aide à la décision — pas un diagnostic officiel."
    )

    df = _cached_health_data()

    # ── Filtres ───────────────────────────────────────────────────────────────
    st.markdown("#### Filtres")
    col1, col2, col3 = st.columns(3)
    with col1:
        maladie = st.selectbox("🦠 Maladie", ["Toutes"] + MALADIES)
    with col2:
        region  = st.selectbox("📍 Région", ["Toutes"] + sorted(REGIONS.keys()))
    with col3:
        annees  = sorted(df["annee"].unique().tolist())
        annee   = st.select_slider("📅 Année", options=["Toutes"] + annees)

    if annee != "Toutes":
        df = df[df["annee"] == int(annee)]
    mask = pd.Series([True] * len(df))
    if maladie != "Toutes":
        mask &= df["maladie"] == maladie
    if region != "Toutes":
        mask &= df["region"] == region
    df_f = df[mask]

    st.divider()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.markdown("#### Indicateurs clés")
    _kpis(df_f)

    st.divider()

    # ── Graphiques ────────────────────────────────────────────────────────────
    st.markdown("#### Évolution temporelle")
    st.plotly_chart(_chart_temporal(df, maladie, region), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Cas par région")
        st.plotly_chart(_chart_regions(df, maladie), use_container_width=True)
    with col_b:
        st.markdown("#### Répartition des maladies")
        st.plotly_chart(_chart_repartition(df, region), use_container_width=True)

    st.markdown("#### Heatmap — Incidence par région et maladie")
    st.plotly_chart(_chart_comparatif(df), use_container_width=True)

    st.markdown("#### Zones à risque")
    st.plotly_chart(_chart_zones_risque(df, maladie), use_container_width=True)

    # ── Tableau détaillé ─────────────────────────────────────────────────────
    with st.expander("📋 Données détaillées"):
        agg = (
            df_f.groupby(["region", "maladie"])
            .agg(cas=("cas", "sum"), deces=("deces", "sum"), incidence=("incidence", "mean"))
            .reset_index()
            .sort_values("cas", ascending=False)
        )
        agg["incidence"] = agg["incidence"].round(2)
        agg["taux_letalite"] = (agg["deces"] / agg["cas"] * 100).round(2)
        st.dataframe(agg, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Télécharger CSV",
            agg.to_csv(index=False).encode("utf-8"),
            "kanea_epidemio_civ.csv", "text/csv",
        )

    st.caption(
        "📌 Sources de référence : OMS · PNLP Côte d'Ivoire · UNICEF · "
        "Bulletin épidémiologique hebdomadaire (INTS). "
        "Données synthétiques — structure prête pour intégration de données réelles."
    )
