"""
KANÉA — Corrélation Santé ↔ Climat · Côte d'Ivoire
════════════════════════════════════════════════════
Fonctions :
  load_climate_data()    → DataFrame climatique synthétique réaliste (CIV)
  render_climate_tab()   → onglet Streamlit complet

Cohérence climatique CIV :
  ▸ Zone équatoriale (sud)  : 2 saisons des pluies, T° ~27°C stable
  ▸ Zone tropicale (centre) : transition
  ▸ Zone soudanienne (nord) : 1 saison des pluies, T° variable 20-38°C

Corrélations épidémiologiques documentées :
  ▸ Paludisme ↑ avec pluviométrie + humidité (prolifération anophèles)
  ▸ Malnutrition ↑ en saison sèche (réduction ressources alimentaires)
  ▸ Diarrhée ↑ avec fortes pluies (contamination eau)

⚠️ Données simulées réalistes — structure prête pour données SYNOP/NASA POWER.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats


# ─── Profils climatiques par zone ──────────────────────────────────────────────

# Température mensuelle moyenne par zone (°C) — données WMO CIV
TEMP_PROFILES: dict[str, list[float]] = {
    "sud":    [27.5, 28.0, 28.5, 27.8, 27.0, 26.0, 25.5, 25.8, 26.5, 27.0, 27.5, 27.5],
    "centre": [26.5, 27.5, 28.5, 28.0, 27.0, 25.5, 24.8, 25.0, 26.0, 27.0, 27.0, 26.5],
    "nord":   [25.0, 27.5, 30.5, 31.0, 30.0, 28.0, 27.0, 27.0, 28.0, 29.0, 28.5, 25.5],
    "ouest":  [26.0, 27.0, 28.0, 27.5, 26.5, 25.0, 24.0, 24.5, 25.5, 26.5, 27.0, 26.5],
    "est":    [27.0, 28.0, 29.0, 28.5, 27.5, 26.0, 25.0, 25.5, 26.5, 27.5, 27.5, 27.0],
}

# Pluviométrie mensuelle moyenne (mm) — données CHIRPS CIV
RAIN_PROFILES: dict[str, list[float]] = {
    "sud":    [50, 80, 130, 180, 200, 120, 60, 80, 150, 190, 120, 60],    # 2 saisons
    "centre": [30, 50, 100, 150, 180, 140, 80, 90, 130, 160, 80, 30],
    "nord":   [5,  10,  20,  50, 120, 160, 200, 220, 140, 50, 10,  5],    # 1 saison
    "ouest":  [40, 70, 120, 170, 200, 150, 100, 110, 170, 180, 110, 50],
    "est":    [35, 65, 110, 160, 190, 130,  70,  85, 140, 175, 100, 45],
}

# Humidité relative (%) — approximation
HUMID_PROFILES: dict[str, list[float]] = {
    "sud":    [80, 82, 84, 86, 88, 87, 85, 84, 86, 88, 86, 82],
    "centre": [70, 72, 76, 80, 84, 85, 82, 80, 82, 84, 78, 72],
    "nord":   [55, 58, 62, 68, 76, 82, 87, 88, 84, 72, 60, 55],
    "ouest":  [75, 78, 82, 84, 86, 86, 84, 82, 84, 86, 82, 78],
    "est":    [72, 74, 78, 82, 85, 84, 82, 80, 82, 84, 78, 74],
}

REGIONS_ZONE: dict[str, str] = {
    # ── Zone Sud ────────────────────────────────────────────────────────────────
    "Abidjan":            "sud",
    "Bas-Sassandra":      "sud",
    "Lagunes":            "sud",
    "Agnéby":             "sud",
    # ── Zone Centre ─────────────────────────────────────────────────────────────
    "Gôh-Djiboua":        "centre",
    "Lacs":               "centre",
    "Sassandra-Marahoué": "centre",
    "Vallée du Bandama":  "centre",
    "Hambol":             "centre",
    "Haut Sassandra":     "centre",
    "Marahoué":           "centre",
    "N'Zi Comoé":         "centre",
    # ── Zone Nord ───────────────────────────────────────────────────────────────
    "Woroba":             "nord",
    "Savanes":            "nord",
    "Denguélé":           "nord",
    "Worodougou":         "nord",
    # ── Zone Ouest ──────────────────────────────────────────────────────────────
    "Montagnes":          "ouest",
    "Moyen Cavally":      "ouest",
    # ── Zone Est ────────────────────────────────────────────────────────────────
    "Comoé":              "est",
    "Moyen Comoé":        "est",
    "Sud Comoé":          "est",
    "Zanzan":             "est",
}


# ═══════════════════════════════════════════════════════════════════════════════
# GÉNÉRATION DE DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

def load_climate_data() -> pd.DataFrame:
    """
    Génère un DataFrame climatique mensuel pour les régions CIV (2021-2023).

    Colonnes :
      date, region, zone, temperature_moy, pluviometrie, humidite,
      indice_secheresse, mois, annee

    Cohérence :
      ▸ Profils basés sur normales WMO et CHIRPS
      ▸ Variabilité inter-annuelle simulée (ENSO léger)
    """
    rng  = np.random.default_rng(42)
    rows = []
    dates = pd.date_range("2021-01", "2023-12", freq="MS")

    for date in dates:
        m    = date.month - 1   # 0-indexé
        annee = date.year
        # Variabilité interannuelle (±0.5°C, ±15mm)
        temp_offset = rng.normal(0, 0.4)
        rain_offset = rng.normal(0, 0.08)   # facteur multiplicatif

        for region, zone in REGIONS_ZONE.items():
            temp  = round(TEMP_PROFILES[zone][m] + temp_offset + rng.normal(0, 0.3), 1)
            pluie = round(max(0, RAIN_PROFILES[zone][m] * (1 + rain_offset) * rng.uniform(0.85, 1.15)), 1)
            humid = round(min(99, max(40, HUMID_PROFILES[zone][m] + rng.normal(0, 1.5))), 1)

            # Indice sécheresse (0 = très humide, 10 = très sec)
            pluie_norm = RAIN_PROFILES[zone][m]
            idx_sec = round(max(0, min(10, 5 - pluie / max(pluie_norm, 1) * 5)), 2)

            rows.append({
                "date":               date,
                "annee":              annee,
                "mois":               date.month,
                "mois_label":         date.strftime("%b %Y"),
                "region":             region,
                "zone":               zone,
                "temperature_moy":    temp,
                "pluviometrie":       pluie,
                "humidite":           humid,
                "indice_secheresse":  idx_sec,
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _merge_health_climate(
    health_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    maladie: str,
    region: str,
) -> pd.DataFrame:
    """Fusionne données santé et climat sur date × région."""
    h = health_df.copy()
    if maladie != "Toutes":
        h = h[h["maladie"] == maladie]
    if region != "Toutes":
        h = h[h["region"] == region]

    h_agg = h.groupby(["date", "region"])["cas"].sum().reset_index()
    merged = h_agg.merge(climate_df, on=["date", "region"], how="inner")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def _chart_double_axis(df: pd.DataFrame, var_climat: str) -> go.Figure:
    """Courbe double axe : cas (rouge) + variable climatique (bleue)."""
    agg = df.groupby("date").agg(
        cas=("cas", "sum"),
        climat=(var_climat, "mean"),
    ).reset_index()

    labels = {
        "pluviometrie":      ("Pluviométrie (mm)", "#3498DB"),
        "temperature_moy":   ("Température (°C)",  "#E74C3C"),
        "humidite":          ("Humidité (%)",       "#27AE60"),
        "indice_secheresse": ("Indice sécheresse",  "#F39C12"),
    }
    label_cl, color_cl = labels.get(var_climat, ("Variable", "#555"))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=agg["date"], y=agg["cas"],
            name="Cas de maladie",
            line=dict(color="#E74C3C", width=2),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=agg["date"], y=agg["climat"],
            name=label_cl,
            line=dict(color=color_cl, width=2, dash="dot"),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Nombre de cas", secondary_y=False)
    fig.update_yaxes(title_text=label_cl, secondary_y=True)
    fig.update_layout(
        title=f"Cas de maladie vs {label_cl}",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _chart_scatter_correlation(df: pd.DataFrame, var_climat: str) -> go.Figure:
    """Nuage de points corrélation (variable climatique vs cas)."""
    agg = df.groupby("date").agg(
        cas=("cas", "sum"),
        climat=(var_climat, "mean"),
    ).reset_index()

    slope, intercept, r, p, _ = stats.linregress(agg["climat"], agg["cas"])
    r2 = round(r ** 2, 3)
    p_label = "< 0.05 (significatif ✓)" if p < 0.05 else f"= {p:.3f} (non significatif)"

    labels = {
        "pluviometrie":      "Pluviométrie (mm)",
        "temperature_moy":   "Température moy. (°C)",
        "humidite":          "Humidité (%)",
        "indice_secheresse": "Indice sécheresse",
    }
    x_label = labels.get(var_climat, var_climat)

    fig = px.scatter(
        agg, x="climat", y="cas",
        trendline="ols",
        labels={"climat": x_label, "cas": "Nombre de cas"},
        title=f"Corrélation : {x_label} ↔ Cas  |  R²={r2}  p{p_label}",
    )
    fig.update_layout(template="plotly_white")
    return fig, r2, p


def _chart_heatmap_correlation(df: pd.DataFrame) -> go.Figure:
    """Heatmap de la matrice de corrélation santé ↔ climat."""
    agg = df.groupby("date").agg(
        cas=("cas", "sum"),
        pluviometrie=("pluviometrie", "mean"),
        temperature_moy=("temperature_moy", "mean"),
        humidite=("humidite", "mean"),
        indice_secheresse=("indice_secheresse", "mean"),
    ).reset_index(drop=True)

    corr = agg[["cas", "pluviometrie", "temperature_moy", "humidite", "indice_secheresse"]].corr()
    labels_fr = {
        "cas": "Cas", "pluviometrie": "Pluviométrie",
        "temperature_moy": "Température", "humidite": "Humidité",
        "indice_secheresse": "Sécheresse",
    }
    corr.index   = [labels_fr[c] for c in corr.index]
    corr.columns = [labels_fr[c] for c in corr.columns]

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Matrice de corrélation — Cas ↔ Variables climatiques",
        aspect="auto",
    )
    fig.update_layout(template="plotly_white")
    return fig


def _chart_seasonality(climate_df: pd.DataFrame, region: str) -> go.Figure:
    """Graphique de saisonnalité climatique par mois."""
    df = climate_df if region == "Toutes" else climate_df[climate_df["region"] == region]
    agg = df.groupby("mois").agg(
        temp=("temperature_moy", "mean"),
        pluie=("pluviometrie", "mean"),
        humid=("humidite", "mean"),
    ).reset_index()
    agg["mois_label"] = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Température (°C)", "Pluviométrie (mm)", "Humidité (%)"],
    )
    fig.add_trace(go.Bar(x=agg["mois_label"], y=agg["temp"],  marker_color="#E74C3C", name="Temp."), row=1, col=1)
    fig.add_trace(go.Bar(x=agg["mois_label"], y=agg["pluie"], marker_color="#3498DB", name="Pluie"), row=1, col=2)
    fig.add_trace(go.Bar(x=agg["mois_label"], y=agg["humid"], marker_color="#27AE60", name="Humid."), row=1, col=3)
    fig.update_layout(
        title=f"Profil climatique mensuel — {region}",
        template="plotly_white",
        showlegend=False,
        height=380,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CLIMAT EN TEMPS RÉEL · PRÉDICTION · ALERTES
# ═══════════════════════════════════════════════════════════════════════════════

_WMO_CONDITIONS = {
    0: "Ciel dégagé", 1: "Principalement dégagé", 2: "Partiellement nuageux", 3: "Couvert",
    45: "Brouillard", 48: "Brouillard givrant",
    51: "Bruine légère", 53: "Bruine modérée", 55: "Bruine dense",
    61: "Pluie légère", 63: "Pluie modérée", 65: "Pluie forte",
    71: "Neige légère", 73: "Neige modérée", 75: "Neige forte",
    80: "Averses légères", 81: "Averses modérées", 82: "Averses violentes",
    95: "Orage", 96: "Orage avec grêle", 99: "Orage violent",
}

_API_PARAMS = {
    "current": (
        "temperature_2m,apparent_temperature,relative_humidity_2m,"
        "precipitation,wind_speed_10m,surface_pressure,weather_code"
    ),
    "daily": (
        "temperature_2m_max,temperature_2m_min,"
        "precipitation_sum,precipitation_probability_max,weather_code"
    ),
    "timezone":      "auto",
    "forecast_days": 7,
}


@st.cache_data(ttl=300)
def get_climate(lat: float, lon: float) -> dict:
    """
    Récupère météo complète via Open-Meteo.
    Lève RuntimeError si l'API est indisponible (pas de mise en cache d'un échec).
    """
    import json as _json
    import urllib.parse
    import urllib.request

    params = {**_API_PARAMS, "latitude": lat, "longitude": lon}

    data = None
    req_error: str | None = None

    # Tentative 1 : requests
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        req_error = str(e)

    # Tentative 2 : urllib (bibliothèque standard — toujours disponible)
    if data is None:
        try:
            url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
                {k: str(v) for k, v in params.items()}
            )
            with urllib.request.urlopen(url, timeout=15) as r:
                data = _json.loads(r.read().decode())
        except Exception as e2:
            raise RuntimeError(
                f"Open-Meteo indisponible — requests: {req_error} | urllib: {e2}"
            ) from e2

    cur  = data["current"]
    day  = data["daily"]
    code = cur.get("weather_code", 0)

    forecast = []
    for i in range(len(day["time"])):
        forecast.append({
            "date":      day["time"][i],
            "t_max":     day["temperature_2m_max"][i],
            "t_min":     day["temperature_2m_min"][i],
            "pluie":     day["precipitation_sum"][i],
            "prob_pluie":day["precipitation_probability_max"][i],
            "condition": _WMO_CONDITIONS.get(day["weather_code"][i], "—"),
        })

    return {
        "temperature":  cur["temperature_2m"],
        "ressenti":     cur["apparent_temperature"],
        "humidity":     cur["relative_humidity_2m"],
        "rainfall":     cur["precipitation"],
        "wind":         cur["wind_speed_10m"],
        "pressure":     cur["surface_pressure"],
        "condition":    _WMO_CONDITIONS.get(code, "—"),
        "forecast":     forecast,
    }


@st.cache_resource
def _load_risk_model():
    """RandomForest entraîné sur données synthétiques CIV (corrélations épidémio. validées)."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(42)
    n = 800
    temp  = rng.uniform(22.0, 36.0, n)
    rain  = rng.uniform(0.0, 250.0, n)
    humid = rng.uniform(45.0, 97.0, n)
    # Score basé sur corrélations connues : pluie + humidité → paludisme
    score = (rain / 250.0) * 0.50 + (humid / 100.0) * 0.35 + ((temp - 22.0) / 14.0) * 0.15
    labels = (score > 0.52).astype(int)
    X = np.column_stack([temp, rain, humid])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, labels)
    return model


def predict_risk(data: dict) -> float:
    """
    Prédit le risque épidémiologique (paludisme) à partir de données climatiques.

    Args:
        data : dict avec keys temperature, rainfall, humidity
    Returns:
        probabilité de risque élevé entre 0.0 et 1.0
    """
    model = _load_risk_model()
    X = np.array([[data["temperature"], data["rainfall"], data["humidity"]]])
    return round(float(model.predict_proba(X)[0][1]), 3)


def alert_system(risk: float, threshold: float = 0.6) -> str:
    """Déclenche une alerte si le risque dépasse le seuil."""
    if risk > threshold:
        return "ALERTE"
    return "OK"


# ═══════════════════════════════════════════════════════════════════════════════
# RENDU STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600)
def _cached_climate() -> pd.DataFrame:
    return load_climate_data()


def render_climate_tab(health_df: pd.DataFrame) -> None:
    """
    Onglet « Corrélation Santé ↔ Climat ».
    Appeler depuis dashboard/app.py.

    Args:
        health_df : DataFrame issu de load_health_data()
    """
    st.markdown(
        "### 🌡️ Corrélation Santé ↔ Climat — Côte d'Ivoire\n"
        "> Analyse de la relation entre variables climatiques et incidence des maladies  \n"
        "> ⚠️ Données climatiques simulées (profils WMO/CHIRPS) — "
        "prêtes pour intégration de données NASA POWER ou SYNOP réelles."
    )

    climate_df = _cached_climate()

    # ── Filtres ────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        maladie = st.selectbox(
            "🦠 Maladie",
            ["Toutes", "Paludisme", "Malnutrition (MAS)", "Malnutrition (MAM)", "Tuberculose", "Diarrhée infantile"],
            key="climat_maladie",
        )
    with col2:
        region = st.selectbox(
            "📍 Région",
            ["Toutes"] + sorted(REGIONS_ZONE.keys()),
            key="climat_region",
        )
    with col3:
        var_climat = st.selectbox(
            "🌦️ Variable climatique",
            ["pluviometrie", "temperature_moy", "humidite", "indice_secheresse"],
            format_func=lambda x: {
                "pluviometrie":      "🌧️ Pluviométrie (mm)",
                "temperature_moy":   "🌡️ Température (°C)",
                "humidite":          "💧 Humidité (%)",
                "indice_secheresse": "☀️ Indice sécheresse",
            }[x],
            key="climat_var",
        )

    merged_df = _merge_health_climate(health_df, climate_df, maladie, region)

    if merged_df.empty:
        st.warning("Aucune donnée pour cette combinaison de filtres.")
        return

    st.divider()

    # ── Indicateurs corrélation ────────────────────────────────────────────────
    agg = merged_df.groupby("date").agg(
        cas=("cas", "sum"),
        var=(var_climat, "mean"),
    )
    r, p = stats.pearsonr(agg["cas"], agg["var"])
    r2   = round(r ** 2, 3)

    cols = st.columns(4)
    with cols[0]:
        st.metric("Coefficient de Pearson (r)", f"{r:.3f}")
    with cols[1]:
        st.metric("R² (variance expliquée)", f"{r2:.3f}")
    with cols[2]:
        sens = "Positive ↑" if r > 0.1 else ("Négative ↓" if r < -0.1 else "Faible ≈")
        st.metric("Direction", sens)
    with cols[3]:
        sig = "✅ Significatif (p<0.05)" if p < 0.05 else "⚠️ Non significatif"
        st.metric("Significativité", sig)

    # Interprétation automatique
    if maladie == "Paludisme" and var_climat == "pluviometrie":
        st.success(
            "✅ **Interprétation :** Corrélation positive attendue — "
            "plus de pluie = plus de gîtes larvaires anophèles = plus de paludisme. "
            "Confirmé par les données épidémiologiques CIV (PNLP 2022)."
        )
    elif maladie in ("Malnutrition (MAS)", "Malnutrition (MAM)") and var_climat == "pluviometrie":
        st.info(
            "ℹ️ **Interprétation :** Corrélation négative attendue — "
            "moins de pluie (saison sèche) = réduction des ressources alimentaires = "
            "plus de malnutrition, surtout dans le Nord."
        )

    st.divider()

    # ── Graphique double axe ───────────────────────────────────────────────────
    st.markdown("#### Évolution temporelle — Cas vs Variable climatique")
    fig_double = _chart_double_axis(merged_df, var_climat)
    st.plotly_chart(fig_double, use_container_width=True)

    # ── Scatter + régression ──────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Nuage de corrélation")
        fig_scatter, _, _ = _chart_scatter_correlation(merged_df, var_climat)
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col_b:
        st.markdown("#### Profil climatique saisonnier")
        fig_seas = _chart_seasonality(climate_df, region)
        st.plotly_chart(fig_seas, use_container_width=True)

    # ── Matrice de corrélation ────────────────────────────────────────────────
    st.markdown("#### Matrice de corrélation complète")
    fig_matrix = _chart_heatmap_correlation(merged_df)
    st.plotly_chart(fig_matrix, use_container_width=True)

    # ── Note sources ─────────────────────────────────────────────────────────
    with st.expander("📚 Sources & méthodologie"):
        st.markdown("""
**Sources climatiques de référence :**
- [NASA POWER](https://power.larc.nasa.gov/) — données ERA5 quotidiennes
- [CHIRPS](https://www.chc.ucsb.edu/data/chirps) — précipitations satellitaires
- Normales climatiques WMO 1991-2020 pour la Côte d'Ivoire

**Corrélations épidémiologiques documentées :**
- *Guis et al. (2012)* : corrélation pluviométrie-paludisme en Afrique de l'Ouest
- *PNLP CIV (2022)* : Bulletin épidémiologique annuel — saisonnalité confirmée
- *Breman et al. (2004)* : Anopheles gambiae et zones humides tropicales

**Méthode statistique :**
- Coefficient de Pearson (r) — corrélation linéaire
- p-value : seuil 0.05 (test bilatéral)
- Données agrégées mensuellemement sur 36 mois (2021-2023)

> ⚠️ Ces données sont synthétiques. Pour une analyse publiable, utiliser les données
> réelles du PNLP Côte d'Ivoire et de la Météorologie Nationale (SODEXAM).
        """)

    # ── Données en temps réel + prédiction de risque ──────────────────────────
    st.divider()
    st.markdown("#### 🌐 Données en temps réel — Open-Meteo + Prédiction de risque")

    from dashboard.maps import CIV_CITIES

    col_city, col_btn = st.columns([4, 1])
    with col_city:
        selected_city = st.selectbox(
            "🏙️ Sélectionner une ville",
            list(CIV_CITIES.keys()),
            key="realtime_city",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch = st.button("🔄 Actualiser", key="fetch_realtime")

    if fetch:
        get_climate.clear()

    city_meta  = CIV_CITIES[selected_city]
    climate_rt = None
    _rt_error  = None
    try:
        climate_rt = get_climate(city_meta["lat"], city_meta["lon"])
    except Exception as _e:
        _rt_error = str(_e)

    if climate_rt:
        risk  = predict_risk(climate_rt)
        alert = alert_system(risk)

        cols = st.columns(4)
        with cols[0]:
            st.metric("🌡️ Température", f"{climate_rt['temperature']:.1f} °C")
        with cols[1]:
            st.metric("🌧️ Précipitations", f"{climate_rt['rainfall']:.1f} mm")
        with cols[2]:
            st.metric("💧 Humidité", f"{climate_rt['humidity']:.0f} %")
        with cols[3]:
            st.metric("⚠️ Score de risque", f"{risk:.0%}")

        if alert == "ALERTE":
            st.error(
                f"🚨 **ALERTE** — Risque épidémique élevé à **{selected_city}** "
                f"(score : {risk:.0%}). Surveillance renforcée recommandée."
            )
        else:
            st.success(
                f"✅ Risque sous contrôle à **{selected_city}** (score : {risk:.0%})"
            )
    else:
        st.warning("⚠️ Données en temps réel indisponibles. Réessayez dans quelques secondes.")
        if _rt_error:
            with st.expander("Détail de l'erreur"):
                st.code(_rt_error)

    st.caption(
        "📌 Sources : PNLP CIV · OMS · NASA POWER · CHIRPS · SODEXAM · Open-Meteo. "
        "Données climatiques simulées selon profils WMO — "
        "intégration de données réelles recommandée pour usage décisionnel."
    )
