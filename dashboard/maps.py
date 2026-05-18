"""
KANÉA — Carte sanitaire interactive · Côte d'Ivoire
═════════════════════════════════════════════════════
Fonctions :
  get_civ_cities()      → dict coordonnées + populations
  generate_civ_map()    → figure Plotly scatter_mapbox
  render_map_tab()      → onglet Streamlit complet

Technologie : Plotly scatter_mapbox (OpenStreetMap, sans token requis)
              Heatmap + cercles proportionnels + tooltips détaillés

⚠️ Données synthétiques — outil d'aide à la décision uniquement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# ─── Géographie CIV ────────────────────────────────────────────────────────────

CIV_CITIES: dict[str, dict] = {
    "Abidjan":          {"lat": 5.3544,  "lon": -4.0083, "region": "Abidjan",            "pop": 5_500_000},
    "Bouaké":           {"lat": 7.6900,  "lon": -5.0300, "region": "Vallée du Bandama",   "pop": 800_000},
    "Yamoussoukro":     {"lat": 6.8277,  "lon": -5.2893, "region": "Lacs",                "pop": 350_000},
    "San-Pédro":        {"lat": 4.7400,  "lon": -6.6369, "region": "Bas-Sassandra",       "pop": 320_000},
    "Korhogo":          {"lat": 9.4585,  "lon": -5.6296, "region": "Savanes",             "pop": 450_000},
    "Man":              {"lat": 7.4121,  "lon": -7.5534, "region": "Montagnes",           "pop": 240_000},
    "Daloa":            {"lat": 6.8770,  "lon": -6.4502, "region": "Sassandra-Marahoué",  "pop": 310_000},
    "Abengourou":       {"lat": 6.7294,  "lon": -3.4964, "region": "Comoé",              "pop": 220_000},
    "Divo":             {"lat": 5.8365,  "lon": -5.3575, "region": "Gôh-Djiboua",        "pop": 180_000},
    "Gagnoa":           {"lat": 6.1319,  "lon": -5.9508, "region": "Gôh-Djiboua",        "pop": 160_000},
    "Soubré":           {"lat": 5.7833,  "lon": -6.6000, "region": "Bas-Sassandra",       "pop": 140_000},
    "Bondoukou":        {"lat": 8.0333,  "lon": -2.8000, "region": "Hambol",              "pop": 130_000},
    "Odienné":          {"lat": 9.5093,  "lon": -7.5671, "region": "Denguélé",           "pop": 90_000},
    "Séguéla":          {"lat": 7.9614,  "lon": -6.6675, "region": "Woroba",             "pop": 110_000},
    "Grand-Bassam":     {"lat": 5.1961,  "lon": -3.7411, "region": "Lagunes",            "pop": 85_000},
    "Ferkessédougou":   {"lat": 9.5955,  "lon": -5.1953, "region": "Savanes",            "pop": 100_000},
    "Katiola":          {"lat": 8.1333,  "lon": -5.1000, "region": "Hambol",             "pop": 75_000},
    "Toumodi":          {"lat": 6.5617,  "lon": -5.0183, "region": "Lacs",               "pop": 60_000},
    "Agboville":        {"lat": 5.9248,  "lon": -4.2152, "region": "Lagunes",            "pop": 95_000},
    "Sassandra":        {"lat": 4.9500,  "lon": -6.0833, "region": "Bas-Sassandra",      "pop": 55_000},
    "Dimbokro":         {"lat": 6.6500,  "lon": -4.7000, "region": "Lacs",                "pop": 80_000},
    "Issia":            {"lat": 6.4930,  "lon": -6.5830, "region": "Sassandra-Marahoué",  "pop": 70_000},
    "Boundiali":        {"lat": 9.5230,  "lon": -6.4860, "region": "Savanes",             "pop": 65_000},
    "Dabakala":         {"lat": 8.3630,  "lon": -4.4300, "region": "Hambol",              "pop": 60_000},
    "Aboisso":          {"lat": 5.4670,  "lon": -3.2000, "region": "Comoé",               "pop": 85_000},
    "Adzopé":           {"lat": 6.1070,  "lon": -3.8610, "region": "Lagunes",             "pop": 72_000},
}

MALADIES = ["Paludisme", "Malnutrition (MAS)", "Malnutrition (MAM)", "Tuberculose", "Diarrhée infantile"]

RISK_SCALE = {
    "Très élevé": "#C0392B",
    "Élevé":      "#E67E22",
    "Modéré":     "#F1C40F",
    "Faible":     "#27AE60",
}

# Centre géographique CIV
CIV_CENTER = {"lat": 7.0, "lon": -5.7}


# ═══════════════════════════════════════════════════════════════════════════════
# DONNÉES VILLE PAR MALADIE
# ═══════════════════════════════════════════════════════════════════════════════

def get_civ_cities() -> dict:
    """Retourne le dictionnaire des villes CIV avec coordonnées."""
    return CIV_CITIES


def _generate_city_data(maladie: str, periode: str) -> pd.DataFrame:
    """
    Génère des données de cas par ville pour une maladie et période données.
    Cohérence géographique :
      ▸ Paludisme   : plus élevé au sud (humide), plus faible au nord
      ▸ Malnutrition: plus élevée au nord (soudano-sahélien)
      ▸ Tuberculose : plus élevée dans les grandes villes (densité)
    """
    rng  = np.random.default_rng(hash(f"{maladie}{periode}") % 2**32)
    rows = []

    for ville, meta in CIV_CITIES.items():
        lat = meta["lat"]
        pop = meta["pop"]

        # Facteur géographique par maladie
        if maladie == "Paludisme":
            # Facteur latitude : plus faible = plus de pluie = plus de paludisme
            geo_factor = max(0.4, 1.8 - (lat - 4.5) / 6.0)
        elif maladie in ("Malnutrition (MAS)", "Malnutrition (MAM)"):
            # Facteur inverse : nord plus touché
            geo_factor = max(0.3, (lat - 4.5) / 6.0 * 1.5)
        elif maladie == "Tuberculose":
            # Densité urbaine : Abidjan, Bouaké, Korhogo
            geo_factor = 1.5 if pop > 300_000 else (0.9 if pop > 100_000 else 0.6)
        else:
            geo_factor = rng.uniform(0.6, 1.2)

        # Base de cas / 100k pop
        base_rates = {
            "Paludisme":            250,
            "Malnutrition (MAS)":   80,
            "Malnutrition (MAM)":   200,
            "Tuberculose":          25,
            "Diarrhée infantile":   120,
        }
        rate_base = base_rates.get(maladie, 100)
        incidence = round(rate_base * geo_factor * rng.uniform(0.7, 1.3), 1)
        cas       = max(1, int(incidence * pop / 100_000))
        deces_rate = {
            "Paludisme": 0.004, "Malnutrition (MAS)": 0.04,
            "Malnutrition (MAM)": 0.008, "Tuberculose": 0.07,
            "Diarrhée infantile": 0.003,
        }.get(maladie, 0.005)
        deces = int(rng.binomial(cas, deces_rate))

        # Niveau de risque
        if incidence >= 300:
            risque = "Très élevé"
        elif incidence >= 200:
            risque = "Élevé"
        elif incidence >= 100:
            risque = "Modéré"
        else:
            risque = "Faible"

        rows.append({
            "ville":     ville,
            "region":    meta["region"],
            "lat":       meta["lat"],
            "lon":       meta["lon"],
            "population": pop,
            "cas":       cas,
            "deces":     deces,
            "incidence": incidence,
            "risque":    risque,
            "maladie":   maladie,
            "periode":   periode,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# CARTES PLOTLY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_civ_map(df_cities: pd.DataFrame, mode: str = "cercles") -> go.Figure:
    """
    Génère la carte interactive de la Côte d'Ivoire.

    Args:
        df_cities : DataFrame avec colonnes lat, lon, cas, incidence, risque, ville
        mode      : "cercles" | "heatmap" | "risque"

    Returns:
        Figure Plotly prête pour st.plotly_chart()
    """
    if mode == "cercles":
        fig = px.scatter_mapbox(
            df_cities,
            lat="lat", lon="lon",
            size="cas",
            color="risque",
            color_discrete_map=RISK_SCALE,
            hover_name="ville",
            hover_data={
                "region":     True,
                "cas":        ":,",
                "deces":      True,
                "incidence":  ":.1f",
                "population": ":,",
                "lat":        False,
                "lon":        False,
                "risque":     False,
            },
            size_max=50,
            zoom=5.8,
            center=CIV_CENTER,
            mapbox_style="open-street-map",
            title="🗺️ Carte sanitaire CIV — Cercles proportionnels aux cas",
        )
        fig.update_layout(height=600, margin=dict(t=40, b=0, l=0, r=0))

    elif mode == "heatmap":
        fig = px.density_mapbox(
            df_cities,
            lat="lat", lon="lon",
            z="incidence",
            radius=40,
            zoom=5.8,
            center=CIV_CENTER,
            mapbox_style="open-street-map",
            color_continuous_scale="YlOrRd",
            title="🌡️ Heatmap — Incidence (/ 100k hab.)",
            hover_name="ville",
        )
        fig.update_layout(height=600, margin=dict(t=40, b=0, l=0, r=0))

    else:  # risque
        fig = px.scatter_mapbox(
            df_cities,
            lat="lat", lon="lon",
            color="risque",
            color_discrete_map=RISK_SCALE,
            size=[30] * len(df_cities),
            size_max=30,
            hover_name="ville",
            hover_data={"region": True, "incidence": ":.1f", "risque": True, "lat": False, "lon": False},
            zoom=5.8,
            center=CIV_CENTER,
            mapbox_style="open-street-map",
            title="🎯 Zones de risque — Côte d'Ivoire",
        )
        fig.update_layout(height=600, margin=dict(t=40, b=0, l=0, r=0))

    return fig


def _chart_top_villes(df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Top N villes par nombre de cas."""
    top = df.nlargest(n, "cas")[["ville", "cas", "deces", "incidence", "risque"]]

    fig = go.Figure(go.Bar(
        x=top["cas"], y=top["ville"],
        orientation="h",
        marker_color=[RISK_SCALE.get(r, "#aaa") for r in top["risque"]],
        text=top["cas"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Cas : %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top {n} villes — cas totaux",
        xaxis_title="Cas",
        template="plotly_white",
        height=380,
        margin=dict(l=120, r=20, t=50, b=40),
    )
    return fig


def _chart_incidence_bubble(df: pd.DataFrame) -> go.Figure:
    """Bubble chart : incidence vs population, taille = décès."""
    fig = px.scatter(
        df,
        x="population", y="incidence",
        size="cas", color="risque",
        color_discrete_map=RISK_SCALE,
        hover_name="ville",
        hover_data={"deces": True, "region": True},
        title="Incidence vs Population — taille = nombre de cas",
        labels={"incidence": "Incidence / 100k", "population": "Population"},
        log_x=True,
    )
    fig.update_layout(template="plotly_white", height=380)
    return fig


def _slider_temporel(df_full: pd.DataFrame, maladie: str) -> None:
    """Slider temporel — animation évolution par mois."""
    from dashboard.stats import load_health_data

    health_df = load_health_data()
    mask = health_df["maladie"] == maladie if maladie != "Toutes" else pd.Series([True] * len(health_df))
    monthly   = (
        health_df[mask]
        .groupby(["date", "region"])["cas"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        monthly, x="region", y="cas",
        animation_frame=monthly["date"].dt.strftime("%b %Y"),
        color="region",
        title=f"Évolution mensuelle par région — {maladie}",
        labels={"cas": "Cas", "region": "Région"},
        height=420,
    )
    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        xaxis_tickangle=-30,
    )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RENDU STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600)
def _cached_city_data(maladie: str, periode: str) -> pd.DataFrame:
    return _generate_city_data(maladie, periode)


def render_map_tab() -> None:
    """
    Onglet « Carte sanitaire – Côte d'Ivoire ».
    Appeler depuis dashboard/app.py à l'intérieur d'un st.tab().
    """
    st.markdown(
        "### 🗺️ Carte sanitaire — Côte d'Ivoire\n"
        "> Visualisation spatiale des cas par ville et région  \n"
        "> ⚠️ Données synthétiques — outil d'aide à la décision uniquement."
    )

    # ── Filtres ────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        maladie = st.selectbox("🦠 Maladie", MALADIES, key="map_maladie")
    with col2:
        periode = st.selectbox("📅 Période", ["2023", "2022", "2021", "2021-2023"], key="map_periode")
    with col3:
        mode    = st.selectbox(
            "🗺️ Mode de visualisation",
            ["cercles", "heatmap", "risque"],
            format_func=lambda x: {"cercles": "Cercles (cas)", "heatmap": "Heatmap (incidence)", "risque": "Zones de risque"}[x],
            key="map_mode",
        )

    df = _cached_city_data(maladie, periode)

    st.divider()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    cols = st.columns(4)
    with cols[0]:
        st.metric("🏙️ Villes", len(df))
    with cols[1]:
        st.metric("🦠 Total cas", f"{df['cas'].sum():,}".replace(",", " "))
    with cols[2]:
        st.metric("📍 Ville la + touchée", df.loc[df["cas"].idxmax(), "ville"])
    with cols[3]:
        n_crit = (df["risque"] == "Très élevé").sum()
        st.metric("🔴 Zones très élevées", n_crit)

    # ── Carte principale ──────────────────────────────────────────────────────
    st.markdown("#### Carte interactive")
    st.info(
        "💡 Survolez un cercle pour voir les détails. "
        "Zoomez et déplacez la carte librement.",
        icon="ℹ️",
    )
    fig_map = generate_civ_map(df, mode=mode)
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Légende risque ────────────────────────────────────────────────────────
    st.markdown("**Niveaux de risque :**")
    cols_leg = st.columns(4)
    for col, (niveau, color) in zip(cols_leg, RISK_SCALE.items()):
        with col:
            st.markdown(
                f'<div style="background:{color};color:white;padding:6px 10px;'
                f'border-radius:6px;text-align:center;font-weight:bold">{niveau}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Graphiques complémentaires ─────────────────────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Top 10 villes")
        st.plotly_chart(_chart_top_villes(df, n=10), use_container_width=True)
    with col_right:
        st.markdown("#### Incidence vs Population")
        st.plotly_chart(_chart_incidence_bubble(df), use_container_width=True)

    st.divider()

    # ── Animation temporelle ──────────────────────────────────────────────────
    st.markdown("#### 📽️ Animation temporelle — évolution par région")
    st.info("▶️ Cliquez sur Play pour animer l'évolution mensuelle.", icon="▶️")
    _slider_temporel(df, maladie)

    # ── Tableau détaillé ──────────────────────────────────────────────────────
    with st.expander("📋 Données par ville"):
        disp = df[["ville", "region", "risque", "cas", "deces", "incidence", "population"]].copy()
        disp = disp.sort_values("cas", ascending=False).reset_index(drop=True)
        disp.columns = ["Ville", "Région", "Risque", "Cas", "Décès", "Incidence /100k", "Population"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Télécharger CSV",
            disp.to_csv(index=False).encode("utf-8"),
            f"kanea_carte_civ_{maladie.replace(' ', '_')}_{periode}.csv",
            "text/csv",
        )

    st.caption(
        "📌 Coordonnées géographiques : données officielles (IGT Côte d'Ivoire / OpenStreetMap). "
        "Données épidémiologiques synthétiques — structure prête pour données PNLP réelles."
    )
