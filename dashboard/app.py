"""
KANÉA — Knowledge Anthropology & Neural Engine for Africa
══════════════════════════════════════════════════════════
Dashboard principal · Streamlit v1.30+

Navigation : sidebar radio → Dashboard / Maladies / Carte / Paramètres
Design     : sidebar anthracite, accent teal #20B2AA, cards avec ombres
"""

from __future__ import annotations

import sys
import base64
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

# ── Compatibilité cloud : ajoute la racine du repo au PYTHONPATH ───────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ── Thème Plotly global : texte sombre lisible sur fond clair ─────────────────
pio.templates["kanea"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(255,255,255,0.88)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        font=dict(color="#1A2B3C", family="Inter, sans-serif", size=12),
        title_font=dict(color="#1A2B3C", size=14),
        xaxis=dict(
            tickfont=dict(color="#1A2B3C", size=12),
            title_font=dict(color="#1A2B3C"),
            linecolor="#DDE8EE",
            gridcolor="#EEF2F6",
        ),
        yaxis=dict(
            tickfont=dict(color="#1A2B3C", size=12),
            title_font=dict(color="#1A2B3C"),
            linecolor="#DDE8EE",
            gridcolor="#EEF2F6",
        ),
        legend=dict(font=dict(color="#1A2B3C", size=12)),
        coloraxis=dict(colorbar=dict(
            tickfont=dict(color="#1A2B3C"),
            title_font=dict(color="#1A2B3C"),
        )),
    )
)
pio.templates.default = "plotly_white+kanea"
import streamlit as st
import streamlit.components.v1 as components

# ─── Modules internes ──────────────────────────────────────────────────────────
try:
    from dashboard.stats import load_health_data, render_stats_tab
    from dashboard.climate import load_climate_data, render_climate_tab, get_climate, predict_risk, alert_system
    from dashboard.maps import CIV_CITIES, render_map_tab
    from modules.integration.engine import multibio_predict
    from dashboard.export_pdf import build_pdf_report
    _MODULES_OK = True
except Exception as _e:
    _MODULES_OK = False
    _MODULES_ERR = str(_e)
    # Fallbacks pour éviter NameError si l'import échoue
    def multibio_predict(**kw): return {"results": {}, "status": "module_unavailable"}
    def load_health_data(): import pandas as pd; return pd.DataFrame()
    def load_climate_data(): import pandas as pd; return pd.DataFrame()
    def render_stats_tab(*a, **kw): pass
    def render_climate_tab(*a, **kw): pass
    def render_map_tab(*a, **kw): pass
    def get_climate(*a, **kw): return None
    def predict_risk(*a, **kw): return 0.0
    def alert_system(*a, **kw): return "OK"
    CIV_CITIES = {}


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "branding" / "kanea_logo.png"

st.set_page_config(
    page_title="KANÉA — Diagnostic IA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# CSS GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════

def inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* ── Fonts ─────────────────────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* ── Variables ─────────────────────────────────────────────────────── */
        :root {
            --teal:        #20B2AA;
            --teal-dark:   #178C86;
            --teal-light:  #E0F7F5;
            --red:         #E74C3C;
            --red-light:   #FDECEA;
            --blue:        #2E86DE;
            --blue-light:  #EBF4FD;
            --green:       #27AE60;
            --green-light: #E9F7EF;
            --orange:      #E67E22;
            --bg:          #F7F9FC;
            --surface:     #FFFFFF;
            --ink:         #1A2B3C;
            --muted:       #5E7A8A;
            --border:      #DDE8EE;
            --shadow-sm:   0 2px 10px rgba(0,0,0,0.06);
            --shadow-md:   0 6px 28px rgba(0,0,0,0.09);
            --shadow-lg:   0 14px 50px rgba(0,0,0,0.12);
            --radius:      16px;
        }

        /* ── Reset global — on exclut span pour ne pas écraser              ──
           les polices d'icônes Material Icons de Streamlit (cause des
           textes bruts "arrow_right", "double_arrow_right" qui apparaissent
           quand font-family:Inter!important écrase le font d'icônes)         */
        html, body, .stApp,
        p, div, h1, h2, h3, h4, h5, h6,
        input, textarea, select, button, label,
        .stMarkdown, .stText, [data-testid="stMarkdownContainer"] {
            font-family: 'Inter', sans-serif !important;
            box-sizing: border-box;
        }
        /* Inter appliqué aux spans de contenu texte uniquement */
        p span, li span, td span, th span,
        .stMarkdown span, [data-testid="stMarkdownContainer"] span {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: transparent !important;
            color: var(--ink) !important;
        }

        /* ── Header/toolbar transparents ───────────────────────────────────── */
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        /* ── Cards en verre dépoli (frosted glass) ───────────────────────── */
        .section-card {
            background: rgba(255,255,255,0.97) !important;
            backdrop-filter: blur(6px) !important;
            -webkit-backdrop-filter: blur(6px) !important;
            border: 1px solid rgba(221,232,238,0.9) !important;
        }
        .kpi-card {
            background: rgba(255,255,255,0.96) !important;
            backdrop-filter: blur(6px) !important;
            -webkit-backdrop-filter: blur(6px) !important;
            border: 1px solid rgba(221,232,238,0.85) !important;
        }
        /* Zone principale bien opaque pour lisibilité */
        .block-container {
            background: rgba(247,249,252,0.88) !important;
            border-radius: 20px;
        }

        /* ── Expander header layout ──────────────────────────────────────── */
        [data-testid="stExpander"] summary {
            display: flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
        }
        [data-testid="stExpander"] summary p { margin: 0; line-height: 1.5; }

        .block-container {
            padding: 1.8rem 2.2rem 4rem !important;
            max-width: 1400px !important;
        }

        /* ── SIDEBAR ─────────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(175deg, #0D1B2A 0%, #112240 60%, #0A1628 100%) !important;
            border-right: 1px solid rgba(32,178,170,0.2) !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.4rem;
        }

        [data-testid="stSidebar"] * {
            color: #C8D8E8 !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #FFFFFF !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a {
            color: var(--teal) !important;
        }

        /* Radio nav items — style sur le div wrapper, pas le label lui-même */
        [data-testid="stSidebar"] .stRadio > div {
            display: flex !important;
            flex-direction: column !important;
            gap: 0.15rem !important;
        }

        [data-testid="stSidebar"] .stRadio label {
            color: #A8C0D0 !important;
            font-size: 0.94rem !important;
            font-weight: 500 !important;
            line-height: 1.5 !important;
            border-radius: 10px !important;
            cursor: pointer !important;
        }

        /* Texte du radio : marges/padding normalisés */
        [data-testid="stSidebar"] .stRadio label p,
        [data-testid="stSidebar"] .stRadio label span {
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.5 !important;
        }

        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSelectbox div {
            color: #C8D8E8 !important;
        }

        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
            background: rgba(255,255,255,0.06) !important;
            border-color: rgba(32,178,170,0.3) !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(32,178,170,0.2) !important;
        }

        /* ── BOUTONS ─────────────────────────────────────────────────────── */
        .stButton > button {
            background: linear-gradient(135deg, var(--teal), var(--teal-dark)) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            padding: 0.6rem 1.4rem !important;
            box-shadow: 0 6px 22px rgba(32,178,170,0.30) !important;
            transition: all 0.2s ease !important;
            min-height: 46px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            line-height: 1.4 !important;
        }

        /* Neutralise les marges du <p> que Streamlit injecte dans le bouton */
        .stButton > button p,
        .stButton > button div {
            margin: 0 !important;
            padding: 0 !important;
            line-height: inherit !important;
            color: inherit !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 28px rgba(32,178,170,0.42) !important;
        }

        /* ── METRICS ─────────────────────────────────────────────────────── */
        [data-testid="metric-container"] {
            background: var(--surface) !important;
            border-radius: var(--radius) !important;
            border: 1px solid var(--border) !important;
            padding: 1rem 1.2rem !important;
            box-shadow: var(--shadow-sm) !important;
        }

        /* ── TABS ────────────────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem !important;
            background: #EEF2F7 !important;
            border-radius: 14px !important;
            padding: 0.4rem !important;
            border: 1px solid var(--border) !important;
            box-shadow: var(--shadow-sm) !important;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            color: var(--muted) !important;
            min-height: 44px !important;
            background: #FFFFFF !important;
            border: 1px solid var(--border) !important;
            padding: 0 1.1rem !important;
            transition: all 0.18s ease !important;
        }

        .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
            background: var(--teal-light) !important;
            color: var(--teal-dark) !important;
            border-color: rgba(32,178,170,0.3) !important;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--teal), var(--teal-dark)) !important;
            color: white !important;
            border-color: transparent !important;
            box-shadow: 0 4px 14px rgba(32,178,170,0.3) !important;
        }

        /* ── EXPANDER ────────────────────────────────────────────────────── */
        [data-testid="stExpander"] {
            border-radius: var(--radius) !important;
            border: 1px solid var(--border) !important;
            box-shadow: var(--shadow-sm) !important;
        }

        /* ── FILE UPLOADER ───────────────────────────────────────────────── */
        [data-testid="stFileUploader"] {
            border-radius: var(--radius) !important;
        }

        /* ── DATAFRAME ───────────────────────────────────────────────────── */
        [data-testid="stDataFrame"] {
            border-radius: var(--radius) !important;
            border: 1px solid var(--border) !important;
        }

        /* ── SPINNER ─────────────────────────────────────────────────────── */
        .stSpinner > div {
            border-top-color: var(--teal) !important;
        }

        /* ── DIVIDER ─────────────────────────────────────────────────────── */
        hr { border-color: var(--border) !important; }

        /* ─── CUSTOM COMPONENTS ──────────────────────────────────────────── */

        /* Page header */
        .page-header {
            padding: 1.6rem 2rem;
            background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%);
            border-radius: 22px;
            color: white;
            margin-bottom: 1.8rem;
            box-shadow: 0 10px 40px rgba(32,178,170,0.28);
        }

        .page-header h1 {
            margin: 0 0 0.3rem 0;
            font-size: 1.9rem;
            font-weight: 800;
            color: white !important;
        }

        .page-header p {
            margin: 0;
            font-size: 0.97rem;
            opacity: 0.88;
            color: white !important;
        }

        /* KPI card */
        .kpi-card {
            background: var(--surface);
            border-radius: var(--radius);
            padding: 1.3rem 1.4rem;
            box-shadow: var(--shadow-md);
            border-top: 4px solid transparent;
            border: 1px solid var(--border);
            transition: transform 0.2s;
        }

        .kpi-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-lg); }
        .kpi-card.kpi-blue   { border-top: 4px solid var(--blue); }
        .kpi-card.kpi-teal   { border-top: 4px solid var(--teal); }
        .kpi-card.kpi-red    { border-top: 4px solid var(--red); }
        .kpi-card.kpi-green  { border-top: 4px solid var(--green); }
        .kpi-card.kpi-orange { border-top: 4px solid var(--orange); }

        .kpi-icon {
            font-size: 1.8rem;
            margin-bottom: 0.4rem;
            line-height: 1;
            display: block;
        }

        .kpi-label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--muted);
            margin: 0 0 0.25rem 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            display: block;
        }

        .kpi-value {
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.2;
            margin: 0 0 0.2rem 0;
            color: var(--ink);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            display: block;
        }

        .kpi-value.kpi-val-blue   { color: var(--blue); }
        .kpi-value.kpi-val-teal   { color: var(--teal); }
        .kpi-value.kpi-val-red    { color: var(--red); }
        .kpi-value.kpi-val-green  { color: var(--green); }
        .kpi-value.kpi-val-orange { color: var(--orange); }

        .kpi-card { overflow: hidden; }

        .kpi-sub {
            font-size: 0.8rem;
            color: var(--muted);
            font-weight: 500;
        }

        /* Section card */
        .section-card {
            background: var(--surface);
            border-radius: 20px;
            padding: 1.4rem 1.6rem;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
            margin-bottom: 1.2rem;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Alert banner */
        .alert-high {
            background: var(--red-light);
            border-left: 5px solid var(--red);
            border-radius: 0 14px 14px 0;
            padding: 1rem 1.2rem;
            margin: 0.8rem 0;
            color: #922B21;
            font-weight: 600;
        }

        .alert-ok {
            background: var(--green-light);
            border-left: 5px solid var(--green);
            border-radius: 0 14px 14px 0;
            padding: 1rem 1.2rem;
            margin: 0.8rem 0;
            color: #1E8449;
            font-weight: 600;
        }

        /* Module card (AI modules) */
        .module-card {
            background: linear-gradient(160deg, var(--surface), #F0FFFE);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.2rem;
            height: 100%;
            box-shadow: var(--shadow-sm);
            transition: all 0.2s;
        }

        .module-card:hover {
            border-color: var(--teal);
            box-shadow: 0 8px 30px rgba(32,178,170,0.15);
        }

        .module-card h4 {
            margin: 0 0 0.5rem 0;
            font-size: 1rem;
            font-weight: 700;
            color: var(--ink);
        }

        .module-card p {
            margin: 0;
            font-size: 0.88rem;
            color: var(--muted);
            line-height: 1.6;
        }

        .chip {
            display: inline-block;
            margin-top: 0.7rem;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            background: rgba(32,178,170,0.1);
            color: var(--teal-dark);
            font-size: 0.76rem;
            font-weight: 700;
            border: 1px solid rgba(32,178,170,0.25);
        }

        /* CTA button */
        .cta-container {
            text-align: center;
            padding: 3.5rem 1rem 2rem;
        }

        .cta-title {
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--ink);
            margin-bottom: 0.6rem;
        }

        .cta-sub {
            font-size: 1rem;
            color: var(--muted);
            margin-bottom: 2rem;
        }

        .cta-btn {
            display: inline-block;
            padding: 1rem 2.8rem;
            background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%);
            color: white !important;
            text-decoration: none !important;
            font-size: 1rem;
            font-weight: 800;
            border-radius: 14px;
            letter-spacing: 0.04em;
            box-shadow: 0 10px 36px rgba(32,178,170,0.38);
            transition: all 0.25s;
            cursor: pointer;
            border: none;
        }

        .cta-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 16px 44px rgba(32,178,170,0.48);
        }

        /* Disclaimer */
        .disclaimer {
            background: linear-gradient(135deg, #FFF8E1, #FFFDE7);
            border: 1px solid #FFD54F;
            border-radius: 14px;
            padding: 0.9rem 1.1rem;
            font-size: 0.85rem;
            color: #7D6608;
            margin-top: 1rem;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: #B2C8D4; border-radius: 6px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--teal); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FOND ANIMÉ — IMAGE SCIENTIFIQUE + MOLÉCULES ANTI-GRAVITÉ
# ═══════════════════════════════════════════════════════════════════════════════

def inject_science_background() -> None:
    """Injecte l'image bg_science.png comme fond fixe animé (zoom lent + shimmer)."""
    bg_path = Path(__file__).resolve().parents[1] / "assets" / "branding" / "bg_science.png"
    if not bg_path.exists():
        return
    bg_b64 = base64.b64encode(bg_path.read_bytes()).decode()
    components.html(
        f"""
        <script>
        (function() {{
          var doc = window.parent.document;
          if (doc.getElementById('kanea-science-bg')) return;

          /* ── Animations CSS ───────────────────────────────────────────── */
          var st = doc.createElement('style');
          st.id = 'kanea-science-style';
          st.textContent = `
            @keyframes kaneaZoom {{
              from {{ transform: scale(1)    translateZ(0); }}
              to   {{ transform: scale(1.10) translateZ(0); }}
            }}
            @keyframes kaneaPulse {{
              0%,100% {{ opacity: 0.28; }}
              50%      {{ opacity: 0.34; }}
            }}
            #kanea-science-bg {{
              position   : fixed;
              top        : -6%; left: -6%;
              width      : 112%; height: 112%;
              background : url('data:image/png;base64,{bg_b64}') center/cover no-repeat;
              z-index    : -10;
              animation  : kaneaZoom 30s ease-in-out infinite alternate,
                           kaneaPulse 10s ease-in-out infinite;
              will-change: transform, opacity;
            }}
            #kanea-overlay {{
              position  : fixed;
              top: 0; left: 0; width: 100%; height: 100%;
              background: linear-gradient(
                135deg,
                rgba(247,249,252,0.78) 0%,
                rgba(247,249,252,0.72) 50%,
                rgba(247,249,252,0.68) 100%
              );
              z-index: -9;
              pointer-events: none;
            }}
          `;
          doc.head.appendChild(st);

          /* ── Div image de fond ────────────────────────────────────────── */
          var bg = doc.createElement('div');
          bg.id = 'kanea-science-bg';
          doc.body.insertBefore(bg, doc.body.firstChild);

          /* ── Overlay de lisibilité ────────────────────────────────────── */
          var ov = doc.createElement('div');
          ov.id = 'kanea-overlay';
          doc.body.insertBefore(ov, doc.body.children[1] || null);
        }})();
        </script>
        """,
        height=0,
    )


def inject_molecule_background() -> None:
    """Canvas HTML5 fixe avec molécules 3D ball-and-stick en apesanteur.

    Injecte le canvas dans le document parent via window.parent (iframe Streamlit).
    Physique : anti-gravité, répulsion souris, rebond doux sur les bords.
    """
    components.html(
        """
        <script>
        (function () {
          var doc = window.parent.document;

          /* ── Évite la duplication lors des re-renders Streamlit ─────────── */
          var existing = doc.getElementById('kanea-bg-canvas');
          if (existing && doc.body.contains(existing)) return;
          if (existing) existing.remove();

          /* ── Rendre .stApp transparent (déjà fait en CSS, sécurité) ─────── */
          var st = doc.getElementById('kanea-bg-style');
          if (!st) {
            st = doc.createElement('style');
            st.id = 'kanea-bg-style';
            st.textContent = '.stApp{background:transparent!important}';
            doc.head.appendChild(st);
          }

          /* ── Créer le canvas fixe derrière tout ─────────────────────────── */
          var cv = doc.createElement('canvas');
          cv.id = 'kanea-bg-canvas';
          cv.style.cssText = [
            'position:fixed', 'top:0', 'left:0',
            'width:100vw', 'height:100vh',
            'z-index:-8', 'pointer-events:none'
          ].join(';');
          doc.body.insertBefore(cv, doc.body.firstChild);

          var ctx = cv.getContext('2d');

          function resize() {
            cv.width  = window.parent.innerWidth;
            cv.height = window.parent.innerHeight;
          }
          resize();
          window.parent.addEventListener('resize', resize);

          /* ── Suivi souris ────────────────────────────────────────────────── */
          var mx = -2000, my = -2000;
          doc.addEventListener('mousemove', function (e) { mx = e.clientX; my = e.clientY; });

          /* ── Palette des atomes (dégradé clair→foncé) ───────────────────── */
          var COLS = [
            ['#00FFFF', '#006699'],   /* cyan */
            ['#FF44FF', '#880088'],   /* magenta */
            ['#FF9900', '#AA4400'],   /* orange */
            ['#66CCFF', '#003388']    /* bleu */
          ];

          /* ── Gabarits de molécules ───────────────────────────────────────── */
          var T = [
            /* Courbé 3 atomes */
            { a: [{x:0,y:0,r:22,c:0},{x:42,y:-24,r:15,c:1},{x:-42,y:-24,r:15,c:2}],
              b: [[0,1],[0,2]] },
            /* Chaîne 4 atomes */
            { a: [{x:-54,y:0,r:16,c:1},{x:-18,y:-22,r:23,c:0},{x:18,y:9,r:16,c:2},{x:54,y:-15,r:14,c:3}],
              b: [[0,1],[1,2],[2,3]] },
            /* Triangle */
            { a: [{x:0,y:-44,r:20,c:0},{x:38,y:22,r:16,c:2},{x:-38,y:22,r:16,c:1}],
              b: [[0,1],[1,2],[2,0]] },
            /* Étoile 4 branches */
            { a: [{x:0,y:0,r:25,c:0},{x:46,y:-20,r:14,c:1},{x:-46,y:-20,r:14,c:2},{x:0,y:50,r:14,c:3}],
              b: [[0,1],[0,2],[0,3]] }
          ];

          /* ── Instancier 15 molécules ─────────────────────────────────────── */
          var W = cv.width, H = cv.height;
          var mols = [];
          for (var i = 0; i < 15; i++) {
            var tmpl = T[i % T.length];
            var sc = 0.42 + Math.random() * 0.9;
            mols.push({
              x:  Math.random() * W,
              y:  Math.random() * H,
              vx: (Math.random() - 0.5) * 0.8,
              vy: (Math.random() - 0.5) * 0.8 - 0.15,
              angle: Math.random() * Math.PI * 2,
              va:    (Math.random() - 0.5) * 0.014,
              scale: sc,
              t:     tmpl,
              alpha: 0.05 + sc * 0.07
            });
          }

          /* ── Particules bokeh flottantes ─────────────────────────────────── */
          var bokeh = [];
          for (var j = 0; j < 20; j++) {
            bokeh.push({
              x: Math.random() * W, y: Math.random() * H,
              r:  28 + Math.random() * 55,
              op: 0.012 + Math.random() * 0.022,
              vx: (Math.random() - 0.5) * 0.22,
              vy: -(0.07 + Math.random() * 0.18),
              col: Math.random() > 0.5 ? [0,200,255] : [140,0,255]
            });
          }

          /* ── Dégradé de fond (blanc/cyan → violet profond) ───────────────── */
          function drawBG() {
            var W = cv.width, H = cv.height;
            var g = ctx.createLinearGradient(0, 0, W, H);
            g.addColorStop(0.00, '#F0FFFE');
            g.addColorStop(0.22, '#D2F4F8');
            g.addColorStop(0.48, '#A8C4E8');
            g.addColorStop(0.70, '#6B3CAA');
            g.addColorStop(1.00, '#150030');
            ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
            /* ondulation teal */
            ctx.save(); ctx.globalAlpha = 0.11;
            var wg = ctx.createLinearGradient(0, H * 0.45, W * 0.65, H);
            wg.addColorStop(0, '#00D4FF');
            wg.addColorStop(1, 'rgba(0,212,255,0)');
            ctx.fillStyle = wg; ctx.fillRect(0, 0, W, H);
            ctx.restore();
          }

          /* ── Bokeh ───────────────────────────────────────────────────────── */
          function drawBokeh() {
            for (var k = 0; k < bokeh.length; k++) {
              var b = bokeh[k], c = b.col;
              var g = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.r);
              g.addColorStop(0, 'rgba('+c[0]+','+c[1]+','+c[2]+','+b.op+')');
              g.addColorStop(1, 'rgba('+c[0]+','+c[1]+','+c[2]+',0)');
              ctx.fillStyle = g;
              ctx.beginPath(); ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2); ctx.fill();
              b.x += b.vx; b.y += b.vy;
              if (b.y < -b.r) { b.y = cv.height + b.r; b.x = Math.random() * cv.width; }
            }
          }

          /* ── Atome sphérique 3D ──────────────────────────────────────────── */
          function drawAtom(x, y, r, cp) {
            var g = ctx.createRadialGradient(x - r*0.35, y - r*0.38, r*0.05, x, y, r);
            g.addColorStop(0, cp[0]); g.addColorStop(1, cp[1]);
            ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2);
            ctx.fillStyle = g; ctx.fill();
            /* reflet spéculaire */
            var hl = ctx.createRadialGradient(x-r*0.38, y-r*0.42, 0, x-r*0.2, y-r*0.28, r*0.52);
            hl.addColorStop(0, 'rgba(255,255,255,0.82)');
            hl.addColorStop(1, 'rgba(255,255,255,0)');
            ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2);
            ctx.fillStyle = hl; ctx.fill();
          }

          /* ── Molécule complète ───────────────────────────────────────────── */
          function drawMol(m) {
            ctx.save();
            ctx.translate(m.x, m.y);
            ctx.rotate(m.angle);
            ctx.scale(m.scale, m.scale);
            ctx.globalAlpha = m.alpha;

            /* liaisons bâtonnets */
            for (var bi = 0; bi < m.t.b.length; bi++) {
              var bond = m.t.b[bi];
              var a1 = m.t.a[bond[0]], a2 = m.t.a[bond[1]];
              ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(a2.x, a2.y);
              ctx.strokeStyle = 'rgba(140,210,255,0.52)';
              ctx.lineWidth = 9; ctx.lineCap = 'round'; ctx.stroke();
              ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(a2.x, a2.y);
              ctx.strokeStyle = 'rgba(225,242,255,0.88)';
              ctx.lineWidth = 3; ctx.stroke();
            }

            /* sphères */
            for (var ai = 0; ai < m.t.a.length; ai++) {
              var atom = m.t.a[ai];
              drawAtom(atom.x, atom.y, atom.r, COLS[atom.c]);
            }
            ctx.restore();
          }

          /* ── Physique anti-gravité ───────────────────────────────────────── */
          function updateMols() {
            var W = cv.width, H = cv.height;
            for (var i = 0; i < mols.length; i++) {
              var m = mols[i];

              /* légère poussée vers le haut (apesanteur) */
              m.vy -= 0.0009;
              /* micro-turbulence */
              m.vx += (Math.random() - 0.5) * 0.0045;
              m.vy += (Math.random() - 0.5) * 0.0045;

              /* répulsion souris */
              var dx = m.x - mx, dy = m.y - my;
              var d  = Math.sqrt(dx*dx + dy*dy);
              if (d < 190 && d > 1) {
                var f = (190 - d) / 190 * 0.7;
                m.vx += dx / d * f;
                m.vy += dy / d * f;
                m.va += f * 0.026 * (Math.random() > 0.5 ? 1 : -1);
              }

              /* amortissement & limite de vitesse */
              m.vx *= 0.989; m.vy *= 0.989; m.va *= 0.976;
              var sp = Math.sqrt(m.vx*m.vx + m.vy*m.vy);
              if (sp > 2.8) { m.vx = m.vx/sp*2.8; m.vy = m.vy/sp*2.8; }

              m.x += m.vx; m.y += m.vy; m.angle += m.va;

              /* rebond doux sur les bords */
              var mg = 95;
              if (m.x < mg)   m.vx += (mg - m.x) * 0.026;
              if (m.x > W-mg) m.vx -= (m.x - (W-mg)) * 0.026;
              if (m.y < mg)   m.vy += (mg - m.y) * 0.026;
              if (m.y > H-mg) m.vy -= (m.y - (H-mg)) * 0.026;
            }
          }

          /* ── Boucle d'animation (pas de fond — image réelle derrière) ────── */
          function frame() {
            ctx.clearRect(0, 0, cv.width, cv.height);
            updateMols();
            for (var i = 0; i < mols.length; i++) drawMol(mols[i]);
            requestAnimationFrame(frame);
          }

          frame();
        })();
        </script>
        """,
        height=0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les données épidémiologiques et climatiques (avec fallback vide)."""
    try:
        health_df = load_health_data()
    except Exception as e:
        st.warning(f"⚠️ Données santé non disponibles : {e}")
        health_df = pd.DataFrame()

    try:
        climate_df = load_climate_data()
    except Exception as e:
        st.warning(f"⚠️ Données climatiques non disponibles : {e}")
        climate_df = pd.DataFrame()

    return health_df, climate_df


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> tuple[str, str]:
    """Rendu sidebar et retourne (page_active, sous_page_maladie)."""
    with st.sidebar:
        # Logo
        if LOGO_PATH.exists():
            _logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
            _logo_html = (
                f'<img src="data:image/png;base64,{_logo_b64}" '
                'style="width:88%; max-height:130px; object-fit:contain; display:block; margin:0 auto 0.6rem; '
                'filter:drop-shadow(0 4px 18px rgba(32,178,170,0.55)) drop-shadow(0 2px 6px rgba(0,0,0,0.4));" />'
            )
        else:
            _logo_html = '<div style="font-size:2rem;font-weight:900;color:#7EECEA;text-align:center;">🧬 KANÉA</div>'

        st.markdown(
            f"""
            <div style="padding: 1rem 0.4rem 1rem; border-bottom: 1px solid rgba(32,178,170,0.25); margin-bottom: 1rem; text-align:center;">
                {_logo_html}
                <div style="font-size: 0.72rem; color: #8AABB8; line-height: 1.5; margin-top: 0.4rem;">
                    Knowledge Anthropology &amp;<br>Neural Engine for Africa
                </div>
                <div style="margin-top: 0.7rem; display: inline-block; padding: 0.25rem 0.7rem;
                     background: linear-gradient(135deg, rgba(32,178,170,0.2), rgba(32,178,170,0.08));
                     border: 1px solid rgba(32,178,170,0.35);
                     border-radius: 999px;
                     font-size: 0.7rem; color: #7EECEA; font-weight: 700; letter-spacing: 0.04em;">
                    IA Médicale · CIV
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation principale
        st.markdown(
            "<div style='font-size:0.7rem; font-weight:700; letter-spacing:0.1em; "
            "color:#5A7A8A; text-transform:uppercase; margin-bottom:0.4rem;'>Navigation</div>",
            unsafe_allow_html=True,
        )

        page = st.radio(
            label="nav",
            options=["📊 Dashboard", "🦠 Maladies", "🗺️ Carte", "⚙️ Paramètres"],
            label_visibility="collapsed",
            key="main_nav",
        )

        sous_page = "Paludisme"
        if page == "🦠 Maladies":
            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            sous_page = st.selectbox(
                "Sélectionner un module",
                ["Paludisme", "Nutrition", "Médico-légal", "Cancer sein"],
                key="sous_maladie",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <hr style='border-color:rgba(32,178,170,0.2); margin: 0.5rem 0;'/>
            <div style='font-size:0.7rem; font-weight:700; letter-spacing:0.1em;
                 color:#5A7A8A; text-transform:uppercase; margin: 0.8rem 0 0.5rem;'>
                Modules IA
            </div>
            """,
            unsafe_allow_html=True,
        )

        modules = [
            ("🔬", "MalariaScan", "ResNet34"),
            ("📊", "Biometry",    "RF + XGBoost"),
            ("🦴", "BioID",       "PCA + Régressions"),
            ("🩺", "BreastCancer","EfficientNet-B0"),
        ]
        for icon, name, tech in modules:
            st.markdown(
                f"""
                <div style='display:flex; align-items:center; gap:0.5rem;
                     padding:0.4rem 0.6rem; margin-bottom:0.2rem;
                     border-radius:10px; border:1px solid rgba(32,178,170,0.12);
                     background:rgba(32,178,170,0.05);'>
                    <span style='font-size:1.1rem;'>{icon}</span>
                    <div>
                        <div style='font-size:0.82rem; font-weight:600; color:#C8D8E8;'>{name}</div>
                        <div style='font-size:0.68rem; color:#5A7A8A;'>{tech}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.7rem; color:#3D5A6A; text-align:center;'>"
            "v1.0 · 2026 · MIT Licence</div>",
            unsafe_allow_html=True,
        )

    return page, sous_page


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def _kpi_card(icon: str, label: str, value: str, sub: str, color_class: str) -> str:
    return f"""
    <div class="kpi-card kpi-{color_class}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value kpi-val-{color_class}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def _chart_animated_maladies(health_df: pd.DataFrame) -> go.Figure:
    """Graphique animé : évolution mensuelle des cas par maladie (animation_frame)."""
    agg = (
        health_df.groupby(["mois_label", "maladie", "date"])["cas"]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    colors = {
        "Paludisme":            "#E74C3C",
        "Malnutrition (MAS)":   "#F39C12",
        "Malnutrition (MAM)":   "#F7DC6F",
        "Tuberculose":          "#8E44AD",
        "Diarrhée infantile":   "#2E86DE",
    }
    fig = px.bar(
        agg,
        x="maladie",
        y="cas",
        color="maladie",
        animation_frame="mois_label",
        range_y=[0, int(agg["cas"].max() * 1.15)],
        color_discrete_map=colors,
        labels={"cas": "Nombre de cas", "maladie": "Maladie"},
        title="📊 Évolution mensuelle des cas — Appuie sur ▶",
    )
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        height=420,
        title_font_size=15,
        paper_bgcolor="rgba(255,255,255,0.88)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        font=dict(color="#1A2B3C", family="Inter, sans-serif"),
        xaxis=dict(showgrid=False, tickfont=dict(color="#1A2B3C", size=12), title_font=dict(color="#1A2B3C")),
        yaxis=dict(gridcolor="#DDE8EE", tickfont=dict(color="#1A2B3C", size=12), title_font=dict(color="#1A2B3C")),
    )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 400
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200
    return fig


def _chart_scatter_climat_paludisme(
    health_df: pd.DataFrame,
    climate_df: pd.DataFrame,
) -> go.Figure:
    """Scatter interactif : Température vs Cas paludisme, taille=pluie, couleur=humidité."""
    pal = health_df[health_df["maladie"] == "Paludisme"]
    pal_agg = pal.groupby(["date", "region"])["cas"].sum().reset_index()

    merged = pal_agg.merge(climate_df, on=["date", "region"], how="inner")

    fig = px.scatter(
        merged,
        x="temperature_moy",
        y="cas",
        size="pluviometrie",
        color="humidite",
        hover_name="region",
        hover_data={"pluviometrie": True, "humidite": True, "date": "|%b %Y"},
        color_continuous_scale="Teal",
        size_max=28,
        labels={
            "temperature_moy": "Température (°C)",
            "cas":              "Cas de paludisme",
            "pluviometrie":     "Pluviométrie (mm)",
            "humidite":         "Humidité (%)",
        },
        title="🌡️ Climat ↔ Paludisme  |  Taille = Pluie  ·  Couleur = Humidité",
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        title_font_size=15,
        paper_bgcolor="rgba(255,255,255,0.88)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        font=dict(color="#1A2B3C", family="Inter, sans-serif"),
        xaxis=dict(tickfont=dict(color="#1A2B3C", size=12), title_font=dict(color="#1A2B3C")),
        yaxis=dict(tickfont=dict(color="#1A2B3C", size=12), title_font=dict(color="#1A2B3C")),
        coloraxis_colorbar=dict(title="Humidité %", len=0.6, tickfont=dict(color="#1A2B3C")),
    )
    return fig


def _inject_geolocation() -> None:
    """Demande la position GPS du navigateur et la stocke dans st.query_params."""
    components.html(
        """
        <script>
        (function() {
          if (!navigator.geolocation) return;
          navigator.geolocation.getCurrentPosition(function(pos) {
            var lat = pos.coords.latitude.toFixed(4);
            var lon = pos.coords.longitude.toFixed(4);
            var url = new URL(window.parent.location.href);
            if (url.searchParams.get('geo_lat') !== lat ||
                url.searchParams.get('geo_lon') !== lon) {
              url.searchParams.set('geo_lat', lat);
              url.searchParams.set('geo_lon', lon);
              window.parent.location.replace(url.toString());
            }
          }, function() {}, { timeout: 5000 });
        })();
        </script>
        """,
        height=0,
    )


def page_dashboard(health_df: pd.DataFrame, climate_df: pd.DataFrame) -> None:
    # ── Géolocalisation : utilise la position GPS du navigateur si dispo ───────
    _inject_geolocation()
    params  = st.query_params
    geo_lat = params.get("geo_lat")
    geo_lon = params.get("geo_lon")

    if geo_lat and geo_lon:
        try:
            lat, lon = float(geo_lat), float(geo_lon)
            location_label = f"📍 Position détectée ({lat:.2f}°, {lon:.2f}°)"
        except ValueError:
            lat, lon = 5.3204, -4.0161
            location_label = "Abidjan · CIV"
    else:
        lat, lon = 5.3204, -4.0161
        location_label = "Abidjan · CIV (par défaut)"

    # En-tête
    st.markdown(
        f"""
        <div class="page-header">
            <h1>📊 Dashboard KANÉA</h1>
            <p>Vue temps réel · {location_label} · Données épidémio &amp; climatiques</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Météo temps réel (Open-Meteo — même source que MSN Météo) ─────────────
    climate_rt  = None
    _meteo_err  = None
    try:
        climate_rt = get_climate(lat, lon)
    except Exception as _e:
        _meteo_err = str(_e)

    temp_val  = f"{climate_rt['temperature']:.0f}°C"   if climate_rt else "—"
    pluie_val = f"{climate_rt['rainfall']:.1f} mm"      if climate_rt else "—"
    risk_val  = "—"
    risk_sub  = "Données live non dispo"
    risk_color = "orange"

    if climate_rt:
        risk_score = predict_risk(climate_rt)
        alert      = alert_system(risk_score)
        risk_val   = f"{risk_score:.0%}"
        risk_sub   = "🚨 Surveillance requise" if alert == "ALERTE" else "✅ Sous contrôle"
        risk_color = "red" if alert == "ALERTE" else "green"

    # Cas paludisme total
    total_cas = (
        health_df[health_df["maladie"] == "Paludisme"]["cas"].sum()
        if not health_df.empty else 0
    )

    # ── KPI CARDS ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    ressenti  = f"Ressenti {climate_rt['ressenti']:.0f}°C" if climate_rt and "ressenti" in climate_rt else "live"
    condition = climate_rt.get("condition", "live") if climate_rt else "live"
    vent      = f"Vent {climate_rt['wind']:.0f} km/h" if climate_rt and "wind" in climate_rt else ""
    humidite  = f"Humidité {climate_rt['humidity']:.0f}%" if climate_rt else ""

    kpis = [
        (col1, "🌡️", "Température",  temp_val,  ressenti,                         "blue"),
        (col2, "🌧️", "Précipitations", pluie_val, condition,                       "blue"),
        (col3, "💨", "Vent · Humidité", vent,     humidite,                         "teal"),
        (col4, "⚠️", "Risque IA",     risk_val,  risk_sub,                          risk_color),
        (col5, "🦠", "Cas paludisme", f"{total_cas:,}".replace(",", " "), "2021-2023", "red"),
    ]
    for col, icon, label, value, sub, color in kpis:
        with col:
            st.markdown(_kpi_card(icon, label, value, sub, color), unsafe_allow_html=True)

    if _meteo_err:
        with st.expander("⚠️ Météo indisponible — détail de l'erreur"):
            st.code(_meteo_err)

    # ── Prévisions 7 jours ────────────────────────────────────────────────────
    if climate_rt and climate_rt.get("forecast"):
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        fcols = st.columns(7)
        for i, (fc, day) in enumerate(zip(fcols, climate_rt["forecast"])):
            import datetime as _dt
            try:
                d = _dt.date.fromisoformat(day["date"])
                label_j = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"][d.weekday()]
                label_j = "Auj." if i == 0 else label_j
            except Exception:
                label_j = f"J+{i}"
            with fc:
                st.markdown(
                    f"""<div style='text-align:center;background:rgba(255,255,255,0.82);
                    border-radius:14px;padding:0.6rem 0.3rem;border:1px solid rgba(255,255,255,0.6);
                    backdrop-filter:blur(8px);'>
                    <div style='font-size:0.75rem;font-weight:700;color:#5E7A8A;'>{label_j}</div>
                    <div style='font-size:1.1rem;'>{"🌧️" if day["prob_pluie"]>50 else "☀️" if day["prob_pluie"]<20 else "⛅"}</div>
                    <div style='font-size:0.9rem;font-weight:700;color:#1A2B3C;'>{day["t_max"]:.0f}°</div>
                    <div style='font-size:0.75rem;color:#5E7A8A;'>{day["t_min"]:.0f}°</div>
                    <div style='font-size:0.68rem;color:#2E86DE;'>{day["prob_pluie"]:.0f}%</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    if health_df.empty or climate_df.empty:
        st.warning("⚠️ Données insuffisantes pour afficher les graphiques.")
        return

    # ── GRAPHIQUES ────────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        # ⚠️ NE PAS splitter un <div> sur plusieurs st.markdown() — chaque appel
        # est indépendant. On utilise un titre natif Streamlit à la place.
        st.markdown(
            "<p style='font-weight:700;font-size:1rem;color:#1A2B3C;"
            "margin:0 0 0.5rem;'>📈 Évolution animée des maladies</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _chart_animated_maladies(health_df),
            use_container_width=True,
        )

    with col_r:
        st.markdown(
            "<p style='font-weight:700;font-size:1rem;color:#1A2B3C;"
            "margin:0 0 0.5rem;'>🌡️ Corrélation Climat ↔ Paludisme</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _chart_scatter_climat_paludisme(health_df, climate_df),
            use_container_width=True,
        )

    # ── Epidémio complète (onglets intégrés) ──────────────────────────────────
    with st.expander("📋 Tableau de bord épidémiologique complet", expanded=False):
        render_stats_tab()

    # ── Climat temps réel ─────────────────────────────────────────────────────
    with st.expander("🌐 Données climatiques & prédiction de risque en temps réel", expanded=False):
        render_climate_tab(health_df)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : MALADIES
# ═══════════════════════════════════════════════════════════════════════════════

def _render_paludisme() -> None:
    st.markdown(
        """
        <div class="page-header" style="background:linear-gradient(135deg,#C0392B,#922B21);">
            <h1>🦠 Paludisme — Module 1 · MalariaScan AI</h1>
            <p>Détection automatique sur images microscopiques de frottis sanguins · ResNet34</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_info, col_upload = st.columns([1, 1.4], gap="large")

    with col_info:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">🔬 Comment ça fonctionne</div>
                <p style="color:#5E7A8A; line-height:1.7; font-size:0.92rem;">
                    Le modèle ResNet34 analyse l'image microscopique et détecte
                    la présence de <strong>Plasmodium</strong> dans les globules rouges.
                    Entraîné sur le dataset NIH (27 560 images).
                </p>
                <div style="margin-top:1rem;">
                    <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                        <span style="background:#FDECEA; color:#E74C3C; padding:0.2rem 0.5rem; border-radius:8px; font-size:0.78rem; font-weight:700;">🔴 Parasitisé</span>
                        <span style="font-size:0.85rem; color:#5E7A8A;">Présence de Plasmodium détectée</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:0.5rem;">
                        <span style="background:#E9F7EF; color:#27AE60; padding:0.2rem 0.5rem; border-radius:8px; font-size:0.78rem; font-weight:700;">🟢 Non infecté</span>
                        <span style="font-size:0.85rem; color:#5E7A8A;">Globules rouges sains</span>
                    </div>
                </div>
            </div>

            <div class="section-card">
                <div class="section-title">📊 Facteurs climatiques de propagation</div>
                <table style="width:100%; font-size:0.86rem; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #DDE8EE;">
                        <td style="padding:0.5rem 0; color:#5E7A8A;">🌧️ Pluviométrie</td>
                        <td style="padding:0.5rem 0; font-weight:700; color:#2E86DE;">Très corrélée ↑</td>
                    </tr>
                    <tr style="border-bottom:1px solid #DDE8EE;">
                        <td style="padding:0.5rem 0; color:#5E7A8A;">💧 Humidité</td>
                        <td style="padding:0.5rem 0; font-weight:700; color:#2E86DE;">Fortement corrélée ↑</td>
                    </tr>
                    <tr>
                        <td style="padding:0.5rem 0; color:#5E7A8A;">🌡️ Température</td>
                        <td style="padding:0.5rem 0; font-weight:700; color:#E67E22;">Corrélation modérée</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_upload:
        st.markdown(
            "<div class='section-card'><div class='section-title'>📤 Analyser une image</div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Importer une image microscopique (PNG, JPG)",
            type=["png", "jpg", "jpeg"],
            key="malaria_upload",
        )

        if uploaded:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

            if st.button("🔬 Lancer l'analyse IA", use_container_width=True, key="btn_malaria"):
                with st.spinner("🧠 Analyse IA en cours — ResNet34 en traitement..."):
                    time.sleep(0.8)
                    suffix = Path(uploaded.name).suffix or ".png"
                    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.getbuffer())
                        temp_path = tmp.name
                    result = multibio_predict(image_path=temp_path)

                pred = result.get("results", {}).get("module_1_malaria", {})
                prediction = pred.get("prediction")
                confidence = pred.get("confidence")

                expl     = pred.get("explainability", {})
                gradcam  = expl.get("heatmap_b64")

                if prediction == "Parasitised":
                    st.markdown(
                        f"<div class='alert-high'>🔴 Résultat : <strong>PARASITISÉ</strong>"
                        f"{'  ·  Confiance : ' + f'{confidence:.1%}' if confidence else ''}"
                        "<br><small>Présence de Plasmodium détectée — consultation médicale recommandée.</small></div>",
                        unsafe_allow_html=True,
                    )
                elif prediction == "Uninfected":
                    st.markdown(
                        f"<div class='alert-ok'>🟢 Résultat : <strong>NON INFECTÉ</strong>"
                        f"{'  ·  Confiance : ' + f'{confidence:.1%}' if confidence else ''}"
                        "<br><small>Globules rouges sains — pas de Plasmodium détecté.</small></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("ℹ️ Modèle non encore entraîné — résultat placeholder. "
                            "Lancer `python scripts/train_malaria_pytorch.py`")

                if gradcam:
                    st.markdown("**🔥 Carte Grad-CAM — zones d'activation ResNet34**")
                    st.markdown(
                        f'<img src="data:image/png;base64,{gradcam}" '
                        'style="width:100%;border-radius:12px;border:1px solid #DDE8EE;" '
                        'alt="Grad-CAM heatmap"/>',
                        unsafe_allow_html=True,
                    )

                _download_report(pred, "malaria", "dl_malaria")
                with st.expander("Réponse JSON complète"):
                    st.json(result)
        else:
            st.markdown(
                """
                <div style="border:2px dashed #DDE8EE; border-radius:14px; padding:2.5rem;
                     text-align:center; color:#8AABB8;">
                    <div style="font-size:2.5rem; margin-bottom:0.5rem;">🔬</div>
                    <div style="font-size:0.92rem;">Importer une image de frottis sanguin microscopique</div>
                    <div style="font-size:0.78rem; margin-top:0.3rem; opacity:0.7;">PNG · JPG · JPEG</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='disclaimer'>⚠️ Outil d'aide à la décision — pas un diagnostic officiel. "
            "Interpréter sous supervision clinique.</div>",
            unsafe_allow_html=True,
        )


def _render_nutrition() -> None:
    st.markdown(
        """
        <div class="page-header" style="background:linear-gradient(135deg,#E67E22,#CA6F1E);">
            <h1>📊 Nutrition — Module 2 · Biometry AI</h1>
            <p>Prédiction nutritionnelle · RandomForest + XGBoost · z-scores OMS</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _NUTRI_INFO = {
        "severe_undernutrition":   ("🔴", "#E74C3C", "#FDECEA", "Malnutrition sévère (MAS)",   "Prise en charge thérapeutique urgente (CRENI/CRENAS)"),
        "moderate_undernutrition": ("🟠", "#E67E22", "#FEF0E7", "Malnutrition modérée (MAM)",  "Supplémentation nutritionnelle thérapeutique (CSPS)"),
        "normal":                  ("🟢", "#27AE60", "#E9F7EF", "Statut nutritionnel normal",  "Maintien de l'alimentation diversifiée"),
        "overweight":              ("🟡", "#F1C40F", "#FEFDE7", "Surpoids",                    "Suivi diététique et activité physique adaptée"),
        "obesity":                 ("🔴", "#C0392B", "#FDECEA", "Obésité",                     "Prise en charge nutritionnelle spécialisée"),
    }

    col_form, col_info = st.columns([1.5, 1], gap="large")

    with col_form:
        st.markdown("<div class='section-card'><div class='section-title'>📋 Données anthropométriques</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        age_months = col1.number_input("Âge (mois)", 0, 240, 24)
        weight_kg  = col2.number_input("Poids (kg)", 0.0, 150.0, 12.0, step=0.1)
        height_cm  = col3.number_input("Taille (cm)", 0.0, 250.0, 85.0, step=0.5)

        col4, col5 = st.columns(2)
        sex  = col4.selectbox("Sexe", ["F", "M"])
        muac = col5.number_input("MUAC (cm)", 0.0, 50.0, 13.5, step=0.1)

        st.markdown("**Z-scores OMS** *(facultatifs — calculés automatiquement si absents)*")
        col6, col7, col8 = st.columns(3)
        waz = col6.number_input("WAZ (P/A)", -6.0, 6.0, 0.0, step=0.1, help="Poids-pour-âge")
        haz = col7.number_input("HAZ (T/A)", -6.0, 6.0, 0.0, step=0.1, help="Taille-pour-âge")
        whz = col8.number_input("WHZ (P/T)", -6.0, 6.0, 0.0, step=0.1, help="Poids-pour-taille")

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("📊 Analyser le statut nutritionnel", use_container_width=True):
            if weight_kg <= 0 or height_cm <= 0:
                st.error("⚠️ Poids et taille doivent être supérieurs à 0.")
            else:
                with st.spinner("🧠 Analyse nutritionnelle en cours — RF + XGBoost..."):
                    time.sleep(0.6)
                    try:
                        result = multibio_predict(nutrition_data={
                            "age_months": age_months, "weight_kg": weight_kg,
                            "height_cm": height_cm,  "sex": sex,
                            "muac_cm": muac,         "waz": waz,
                            "haz": haz,              "whz": whz,
                        })
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse : {e}")
                        result = None

                if result:
                    pred_data  = result.get("results", {}).get("module_2_biometry", {})
                    prediction = pred_data.get("prediction")
                    confidence = pred_data.get("confidence")
                    derived    = pred_data.get("derived_features", {})
                    bmi        = derived.get("bmi")

                    if prediction and prediction in _NUTRI_INFO:
                        icon, color, bg, label, reco = _NUTRI_INFO[prediction]
                        st.markdown(
                            f"""<div style='background:{bg};border-left:5px solid {color};
                            border-radius:0 14px 14px 0;padding:1.2rem 1.4rem;margin:0.8rem 0;'>
                            <div style='font-size:1.3rem;font-weight:800;color:{color};'>
                                {icon} {label}
                            </div>
                            {"<div style='font-size:0.9rem;color:#5E7A8A;margin-top:0.3rem;'>Confiance : <b>" + f"{confidence:.1%}" + "</b></div>" if confidence else ""}
                            <div style='font-size:0.88rem;margin-top:0.5rem;color:#1A2B3C;'>
                                💡 <strong>Recommandation :</strong> {reco}
                            </div></div>""",
                            unsafe_allow_html=True,
                        )

                        # Z-scores gauge
                        cols_z = st.columns(4)
                        for col_z, (label_z, val) in zip(
                            cols_z,
                            [("WAZ", waz), ("HAZ", haz), ("WHZ", whz), ("BMI", bmi or 0)],
                        ):
                            z_color = "#E74C3C" if val < -2 else ("#F39C12" if val < -1 else "#27AE60")
                            with col_z:
                                st.markdown(
                                    f"""<div style='text-align:center;background:rgba(255,255,255,0.9);
                                    border-radius:12px;padding:0.7rem;border:1px solid #DDE8EE;'>
                                    <div style='font-size:0.72rem;font-weight:700;color:#5E7A8A;
                                    text-transform:uppercase;'>{label_z}</div>
                                    <div style='font-size:1.5rem;font-weight:800;color:{z_color};'>{val:.1f}</div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.info("ℹ️ Modèle non encore entraîné — résultat placeholder. "
                                "Lancer `python scripts/train_biometry_model.py`")

                    _download_report(pred_data, "nutrition", "dl_nutrition")
                    with st.expander("Réponse JSON complète"):
                        st.json(result)

    with col_info:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">📏 Référentiel OMS</div>
                <table style="width:100%; font-size:0.84rem; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #DDE8EE;">
                        <td style="padding:0.4rem 0; color:#E74C3C; font-weight:700;">🔴 MAS</td>
                        <td style="padding:0.4rem 0; color:#5E7A8A;">WHZ &lt; -3 ou MUAC &lt; 11.5</td>
                    </tr>
                    <tr style="border-bottom:1px solid #DDE8EE;">
                        <td style="padding:0.4rem 0; color:#E67E22; font-weight:700;">🟠 MAM</td>
                        <td style="padding:0.4rem 0; color:#5E7A8A;">WHZ -3 à -2 ou MUAC 11.5–12.5</td>
                    </tr>
                    <tr style="border-bottom:1px solid #DDE8EE;">
                        <td style="padding:0.4rem 0; color:#27AE60; font-weight:700;">🟢 Normal</td>
                        <td style="padding:0.4rem 0; color:#5E7A8A;">WHZ -2 à +2</td>
                    </tr>
                    <tr>
                        <td style="padding:0.4rem 0; color:#F1C40F; font-weight:700;">🟡 Surpoids</td>
                        <td style="padding:0.4rem 0; color:#5E7A8A;">WHZ &gt; +2</td>
                    </tr>
                </table>
            </div>
            <div class='disclaimer' style='margin-top:0.8rem;'>
                ⚠️ Outil d'aide à la décision — supervision médicale requise.
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_medicolegal() -> None:
    st.markdown(
        """
        <div class="page-header" style="background:linear-gradient(135deg,#8E44AD,#6C3483);">
            <h1>🦴 Médico-légal — Module 3 · BioID AI</h1>
            <p>Estimation du profil biologique · PCA · Régressions ostéométriques · AIMs</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_form, col_help = st.columns([1.6, 1], gap="large")

    with col_form:
        with st.expander("🦷 Mesures crâniennes", expanded=True):
            c1, c2, c3 = st.columns(3)
            mcl = c1.number_input("Longueur crânienne max (mm)", 0.0, 250.0, 180.0)
            mcb = c2.number_input("Largeur crânienne max (mm)",  0.0, 250.0, 140.0)
            bzb = c3.number_input("Diamètre bizygomatique (mm)", 0.0, 200.0, 128.0)
            c4, c5, c6 = st.columns(3)
            nh  = c4.number_input("Hauteur nasale (mm)",        0.0, 100.0, 52.0)
            nb  = c5.number_input("Largeur nasale (mm)",        0.0,  80.0, 25.0)
            bnl = c6.number_input("Longueur basion-nasion (mm)",0.0, 150.0, 98.0)

        with st.expander("🦴 Mesures post-crâniennes", expanded=True):
            c7, c8, c9, c10 = st.columns(4)
            fem = c7.number_input("Fémur (cm)",   0.0, 80.0, 45.0)
            tib = c8.number_input("Tibia (cm)",   0.0, 70.0, 37.0)
            hum = c9.number_input("Humérus (cm)", 0.0, 60.0, 32.0)
            rad = c10.number_input("Radius (cm)", 0.0, 50.0, 24.0)

        with st.expander("🧬 Marqueurs ancestraux (AIMs)", expanded=False):
            c11, c12, c13 = st.columns(3)
            pc1 = c11.number_input("AIM_PC1", -5.0, 5.0, 0.0, step=0.01)
            pc2 = c12.number_input("AIM_PC2", -5.0, 5.0, 0.0, step=0.01)
            pc3 = c13.number_input("AIM_PC3", -5.0, 5.0, 0.0, step=0.01)

        if st.button("🦴 Analyser le profil BioID", use_container_width=True):
            with st.spinner("🧠 Estimation du profil biologique en cours — PCA + régressions..."):
                time.sleep(0.6)
                try:
                    result = multibio_predict(bioid_data={
                        "cranial_measurements": {
                            "max_cranial_length_mm": mcl, "max_cranial_breadth_mm": mcb,
                            "bizygomatic_breadth_mm": bzb, "nasal_height_mm": nh,
                            "nasal_breadth_mm": nb, "basion_nasion_length_mm": bnl,
                        },
                        "postcranial_measurements": {
                            "femur_length_cm": fem, "tibia_length_cm": tib,
                            "humerus_length_cm": hum, "radius_length_cm": rad,
                        },
                        "aims_pcs": {"AIM_PC1": pc1, "AIM_PC2": pc2, "AIM_PC3": pc3},
                    })
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {e}")
                    result = None

            if result:
                pred_data   = result.get("results", {}).get("module_3_forensic", {})
                prediction  = pred_data.get("prediction", {})
                confidence  = pred_data.get("confidence", {})
                status      = pred_data.get("status", "scaffold_ready")

                if status == "model_loaded" and prediction:
                    bio_sex   = prediction.get("biological_sex")
                    age_death = prediction.get("age_at_death")
                    ancestry  = prediction.get("ancestry")
                    stature   = prediction.get("stature_cm")
                    conf_sex  = confidence.get("biological_sex") if confidence else None
                    conf_anc  = confidence.get("ancestry") if confidence else None

                    st.markdown(
                        f"""<div class='section-card'>
                        <div class='section-title'>🦴 Profil biologique estimé</div>
                        <div style='display:grid;grid-template-columns:1fr 1fr;gap:1rem;'>
                            <div style='background:#EBF4FD;border-radius:12px;padding:1rem;text-align:center;'>
                                <div style='font-size:0.72rem;font-weight:700;color:#5E7A8A;text-transform:uppercase;'>Sexe biologique</div>
                                <div style='font-size:1.6rem;font-weight:800;color:#2E86DE;'>{"♂ Masculin" if bio_sex == "M" else "♀ Féminin" if bio_sex == "F" else bio_sex or "—"}</div>
                                {f"<div style='font-size:0.78rem;color:#5E7A8A;'>Confiance : {conf_sex:.1%}</div>" if conf_sex else ""}
                            </div>
                            <div style='background:#EBF4FD;border-radius:12px;padding:1rem;text-align:center;'>
                                <div style='font-size:0.72rem;font-weight:700;color:#5E7A8A;text-transform:uppercase;'>Âge au décès</div>
                                <div style='font-size:1.6rem;font-weight:800;color:#2E86DE;'>{f"{age_death:.0f} ans" if age_death else "—"}</div>
                            </div>
                            <div style='background:#F5EEF8;border-radius:12px;padding:1rem;text-align:center;'>
                                <div style='font-size:0.72rem;font-weight:700;color:#5E7A8A;text-transform:uppercase;'>Ascendance</div>
                                <div style='font-size:1.3rem;font-weight:800;color:#8E44AD;'>{ancestry or "—"}</div>
                                {f"<div style='font-size:0.78rem;color:#5E7A8A;'>Confiance : {conf_anc:.1%}</div>" if conf_anc else ""}
                            </div>
                            <div style='background:#EAF5EA;border-radius:12px;padding:1rem;text-align:center;'>
                                <div style='font-size:0.72rem;font-weight:700;color:#5E7A8A;text-transform:uppercase;'>Stature estimée</div>
                                <div style='font-size:1.6rem;font-weight:800;color:#27AE60;'>{f"{stature:.1f} cm" if stature else "—"}</div>
                            </div>
                        </div></div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("ℹ️ Modèle non encore entraîné — résultat placeholder. "
                            "Lancer `python scripts/train_forensic_model.py`")

                _download_report(pred_data, "medicolegal", "dl_bioid")
                with st.expander("Réponse JSON complète"):
                    st.json(result)

    with col_help:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">📐 Repères anthropométriques</div>
                <table style="width:100%; font-size:0.82rem; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #DDE8EE;">
                        <td style="padding:0.4rem 0; color:#5E7A8A; font-weight:600;">Fémur moyen H</td>
                        <td style="padding:0.4rem 0; color:#1A2B3C;">45–48 cm</td>
                    </tr>
                    <tr style="border-bottom:1px solid #DDE8EE;">
                        <td style="padding:0.4rem 0; color:#5E7A8A; font-weight:600;">Fémur moyen F</td>
                        <td style="padding:0.4rem 0; color:#1A2B3C;">41–44 cm</td>
                    </tr>
                    <tr style="border-bottom:1px solid #DDE8EE;">
                        <td style="padding:0.4rem 0; color:#5E7A8A; font-weight:600;">Bizygomatique H</td>
                        <td style="padding:0.4rem 0; color:#1A2B3C;">130–140 mm</td>
                    </tr>
                    <tr>
                        <td style="padding:0.4rem 0; color:#5E7A8A; font-weight:600;">Bizygomatique F</td>
                        <td style="padding:0.4rem 0; color:#1A2B3C;">118–128 mm</td>
                    </tr>
                </table>
                <div style="margin-top:0.8rem;font-size:0.78rem;color:#5E7A8A;line-height:1.6;">
                    Méthodes : <strong>PCA craniométrique</strong>,
                    régressions ostéométriques Trotter & Gleser
                    adaptées aux populations africaines subsahariennes.
                </div>
            </div>
            <div class='disclaimer' style='margin-top:0.8rem;'>
                ⚠️ Usage médico-légal uniquement — supervision d'un anthropologue judiciaire requise.
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_breast_cancer() -> None:
    st.markdown(
        """
        <div class="page-header" style="background:linear-gradient(135deg,#D91E7A,#A91560);">
            <h1>🩺 Cancer du Sein — Module 4 · EfficientNet-B0</h1>
            <p>Classification mammographique · Normal / Bénin / Malin · Offline CPU</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _BC_INFO = {
        "Normal":    ("🟢", "#27AE60", "#E9F7EF", "Tissu mammaire normal",    "Poursuite du dépistage standard annuel."),
        "Benign":    ("🟡", "#E67E22", "#FEF0E7", "Lésion bénigne détectée",  "Surveillance rapprochée recommandée — IRM ou biopsie à discuter."),
        "Malignant": ("🔴", "#E74C3C", "#FDECEA", "Suspicion maligne",        "Avis oncologique urgent — biopsie confirmatoire requise."),
    }

    col_info, col_upload = st.columns([1, 1.4], gap="large")

    with col_info:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">🩺 Comment ça fonctionne</div>
                <p style="color:#5E7A8A; line-height:1.7; font-size:0.92rem;">
                    EfficientNet-B0 analyse la mammographie et classe le tissu en
                    <strong>Normal</strong>, <strong>Bénin</strong> ou <strong>Malin</strong>.
                    La carte Grad-CAM surligne les zones activatrices de la décision.
                </p>
                <div style="margin-top:0.8rem;">
                    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem;">
                        <span style="background:#E9F7EF;color:#27AE60;padding:0.2rem 0.5rem;border-radius:8px;font-size:0.78rem;font-weight:700;">🟢 Normal</span>
                        <span style="font-size:0.82rem;color:#5E7A8A;">Aucune anomalie détectée</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem;">
                        <span style="background:#FEF0E7;color:#E67E22;padding:0.2rem 0.5rem;border-radius:8px;font-size:0.78rem;font-weight:700;">🟡 Bénin</span>
                        <span style="font-size:0.82rem;color:#5E7A8A;">Lésion non maligne</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:0.5rem;">
                        <span style="background:#FDECEA;color:#E74C3C;padding:0.2rem 0.5rem;border-radius:8px;font-size:0.78rem;font-weight:700;">🔴 Malin</span>
                        <span style="font-size:0.82rem;color:#5E7A8A;">Suspicion maligne</span>
                    </div>
                </div>
            </div>
            <div class='disclaimer' style='margin-top:0.8rem;'>
                ⚠️ Outil de triage — ne remplace pas une biopsie ni un avis radiologique.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_upload:
        st.markdown("<div class='section-card'><div class='section-title'>📤 Analyser une mammographie</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Importer une mammographie (PNG, JPG)", type=["png", "jpg", "jpeg"], key="bc_upload")

        if uploaded:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

            if st.button("🩺 Analyser la mammographie", use_container_width=True, key="btn_bc"):
                with st.spinner("🧠 Classification EfficientNet-B0 en cours..."):
                    time.sleep(0.8)
                    try:
                        suffix = Path(uploaded.name).suffix or ".png"
                        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded.getbuffer())
                            temp_path = tmp.name
                        result = multibio_predict(breast_cancer_image_path=temp_path)
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse : {e}")
                        result = None

                if result:
                    pred_data  = result.get("results", {}).get("module_4_breast_cancer", {})
                    prediction = pred_data.get("prediction")
                    confidence = pred_data.get("confidence")
                    expl       = pred_data.get("explainability", {})
                    gradcam    = expl.get("heatmap_b64")

                    if prediction and prediction in _BC_INFO:
                        icon, color, bg, label, reco = _BC_INFO[prediction]
                        st.markdown(
                            f"""<div style='background:{bg};border-left:5px solid {color};
                            border-radius:0 14px 14px 0;padding:1.2rem 1.4rem;margin:0.8rem 0;'>
                            <div style='font-size:1.3rem;font-weight:800;color:{color};'>
                                {icon} {label}
                            </div>
                            {"<div style='font-size:0.9rem;color:#5E7A8A;margin-top:0.3rem;'>Confiance : <b>" + f"{confidence:.1%}" + "</b></div>" if confidence else ""}
                            <div style='font-size:0.88rem;margin-top:0.5rem;color:#1A2B3C;'>
                                💡 <strong>À faire :</strong> {reco}
                            </div></div>""",
                            unsafe_allow_html=True,
                        )

                        if gradcam:
                            st.markdown("**🔥 Carte Grad-CAM — zones d'activation**")
                            st.markdown(
                                f'<img src="data:image/png;base64,{gradcam}" '
                                'style="width:100%;border-radius:12px;border:1px solid #DDE8EE;" '
                                'alt="Grad-CAM heatmap"/>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("ℹ️ Modèle non encore entraîné — résultat placeholder. "
                                "Lancer `python scripts/train_breast_cancer_model.py`")

                    _download_report(pred_data, "breast_cancer", "dl_bc")
                    with st.expander("Réponse JSON complète"):
                        st.json(result)
        else:
            st.markdown(
                """
                <div style="border:2px dashed #DDE8EE; border-radius:14px; padding:2.5rem;
                     text-align:center; color:#8AABB8;">
                    <div style="font-size:2.5rem; margin-bottom:0.5rem;">🩺</div>
                    <div style="font-size:0.92rem;">Importer une mammographie pour activer l'analyse</div>
                    <div style="font-size:0.78rem; margin-top:0.3rem; opacity:0.7;">PNG · JPG · JPEG · Fonctionne hors ligne sur CPU</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def page_maladies(sous_page: str) -> None:
    dispatch = {
        "Paludisme":    _render_paludisme,
        "Nutrition":    _render_nutrition,
        "Médico-légal": _render_medicolegal,
        "Cancer sein":  _render_breast_cancer,
    }
    dispatch.get(sous_page, _render_paludisme)()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : CARTE
# ═══════════════════════════════════════════════════════════════════════════════

def page_carte(health_df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="page-header">
            <h1>🗺️ Carte sanitaire — Côte d'Ivoire</h1>
            <p>HeatMap épidémiologique interactive · 26 villes · Cas paludisme</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tentative avec Folium (streamlit-folium requis)
    try:
        import folium
        from folium.plugins import HeatMap
        from streamlit_folium import st_folium

        # Calcul intensité paludisme par ville
        pal = health_df[health_df["maladie"] == "Paludisme"]
        region_cas = pal.groupby("region")["cas"].sum().to_dict()

        heat_data = []
        for city, meta in CIV_CITIES.items():
            region = meta["region"]
            cas    = region_cas.get(region, 0)
            intensity = np.log1p(cas) / 15.0
            heat_data.append([meta["lat"], meta["lon"], min(intensity, 1.0)])

        m = folium.Map(
            location=[7.0, -5.7],
            zoom_start=6,
            tiles="CartoDB positron",
        )

        HeatMap(
            heat_data,
            radius=30,
            blur=20,
            min_opacity=0.3,
            gradient={0.3: "#27AE60", 0.6: "#F39C12", 0.85: "#E74C3C", 1.0: "#922B21"},
        ).add_to(m)

        # Marqueurs villes
        for city, meta in CIV_CITIES.items():
            region = meta["region"]
            cas    = region_cas.get(region, 0)
            folium.CircleMarker(
                location=[meta["lat"], meta["lon"]],
                radius=5,
                color="#20B2AA",
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b>{city}</b><br>Région: {region}<br>Cas paludisme: {cas:,}",
                    max_width=200,
                ),
                tooltip=city,
            ).add_to(m)

        col_map, col_legend = st.columns([3, 1])
        with col_map:
            st_folium(m, width="100%", height=560, returned_objects=[])
        with col_legend:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">Légende</div>
                    <div style="display:flex;flex-direction:column;gap:0.6rem;font-size:0.85rem;">
                        <div><span style="background:#922B21;padding:2px 10px;border-radius:4px;">&nbsp;</span> &nbsp;Très élevé</div>
                        <div><span style="background:#E74C3C;padding:2px 10px;border-radius:4px;">&nbsp;</span> &nbsp;Élevé</div>
                        <div><span style="background:#F39C12;padding:2px 10px;border-radius:4px;">&nbsp;</span> &nbsp;Modéré</div>
                        <div><span style="background:#27AE60;padding:2px 10px;border-radius:4px;">&nbsp;</span> &nbsp;Faible</div>
                    </div>
                    <hr/>
                    <div style="font-size:0.8rem;color:#5E7A8A;margin-top:0.5rem;">
                        🔵 Points bleus = villes<br>
                        Intensité = log(cas)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except ImportError:
        st.warning(
            "📦 `streamlit-folium` non installé.  \n"
            "Carte Plotly affichée en remplacement.  \n"
            "Installer avec : `pip install streamlit-folium`"
        )
        render_map_tab()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : PARAMÈTRES
# ═══════════════════════════════════════════════════════════════════════════════

def page_parametres() -> None:
    st.markdown(
        """
        <div class="page-header" style="background:linear-gradient(135deg,#34495E,#2C3E50);">
            <h1>⚙️ Paramètres & À propos</h1>
            <p>Configuration de l'application · Documentation · Statut des modules</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_about, tab_modules, tab_data = st.tabs(["À propos", "Statut modules", "Sources de données"])

    with tab_about:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">🧬 KANÉA</div>
                    <p style="color:#5E7A8A; line-height:1.8; font-size:0.92rem;">
                        <strong>Knowledge Anthropology & Neural Engine for Africa</strong><br><br>
                        Plateforme d'aide à la décision biomédicale et médico-légale,
                        conçue pour les environnements à ressources limitées en Afrique subsaharienne.
                        Fonctionne entièrement <strong>hors ligne</strong> sur CPU.
                    </p>
                    <div style="margin-top:1rem;">
                        <span class="chip">v1.0</span>
                        <span class="chip" style="margin-left:0.4rem;">MIT Licence</span>
                        <span class="chip" style="margin-left:0.4rem;">Python 3.10+</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">🛠️ Stack technique</div>
                    <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
                        <tr style="border-bottom:1px solid #DDE8EE;">
                            <td style="padding:0.45rem 0; color:#5E7A8A; font-weight:600;">UI</td>
                            <td style="padding:0.45rem 0; color:#1A2B3C;">Streamlit 1.30+</td>
                        </tr>
                        <tr style="border-bottom:1px solid #DDE8EE;">
                            <td style="padding:0.45rem 0; color:#5E7A8A; font-weight:600;">Deep Learning</td>
                            <td style="padding:0.45rem 0; color:#1A2B3C;">PyTorch · ResNet34 · EfficientNet-B0</td>
                        </tr>
                        <tr style="border-bottom:1px solid #DDE8EE;">
                            <td style="padding:0.45rem 0; color:#5E7A8A; font-weight:600;">ML</td>
                            <td style="padding:0.45rem 0; color:#1A2B3C;">scikit-learn · XGBoost · PCA</td>
                        </tr>
                        <tr style="border-bottom:1px solid #DDE8EE;">
                            <td style="padding:0.45rem 0; color:#5E7A8A; font-weight:600;">Cartes</td>
                            <td style="padding:0.45rem 0; color:#1A2B3C;">Folium · Plotly</td>
                        </tr>
                        <tr>
                            <td style="padding:0.45rem 0; color:#5E7A8A; font-weight:600;">API</td>
                            <td style="padding:0.45rem 0; color:#1A2B3C;">FastAPI · Uvicorn</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_modules:
        modules_status = [
            ("Module 1", "🔬 MalariaScan AI",   "ResNet34 · NIH 27k images",    Path("models/deep_learning/malaria_model.pth")),
            ("Module 2", "📊 Biometry AI",        "RF + XGBoost · z-scores OMS",  Path("models/machine_learning/nutrition_model.pkl")),
            ("Module 3", "🦴 BioID AI",           "PCA + régressions ostéo",      Path("models/machine_learning/forensic_model.pkl")),
            ("Module 4", "🩺 Breast Cancer AI",   "EfficientNet-B0 · 3 classes",  Path("models/deep_learning/breast_cancer_model.pth")),
        ]

        for mod, name, desc, model_path in modules_status:
            exists = (Path.cwd() / model_path).exists()
            badge_color = "#27AE60" if exists else "#E74C3C"
            badge_text  = "✅ Modèle chargé" if exists else "⚠️ Modèle manquant"
            badge_bg    = "#E9F7EF" if exists else "#FDECEA"
            st.markdown(
                f"""
                <div class="section-card" style="margin-bottom:0.8rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="font-weight:700; font-size:0.95rem;">{name}</div>
                            <div style="font-size:0.82rem; color:#5E7A8A; margin-top:0.2rem;">{mod} · {desc}</div>
                            <div style="font-size:0.75rem; color:#8AABB8; font-family:monospace;">{model_path}</div>
                        </div>
                        <span style="background:{badge_bg}; color:{badge_color}; padding:0.3rem 0.8rem;
                               border-radius:999px; font-size:0.8rem; font-weight:700; white-space:nowrap;">
                            {badge_text}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='disclaimer'>Pour entraîner le modèle malaria : "
            "<code>python scripts/train_malaria_pytorch.py</code></div>",
            unsafe_allow_html=True,
        )

    with tab_data:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">📚 Sources de données</div>
                <table style="width:100%; font-size:0.86rem; border-collapse:collapse;">
                    <thead>
                        <tr style="border-bottom:2px solid #DDE8EE;">
                            <th style="padding:0.6rem; text-align:left; color:#1A2B3C;">Dataset</th>
                            <th style="padding:0.6rem; text-align:left; color:#1A2B3C;">Source</th>
                            <th style="padding:0.6rem; text-align:left; color:#1A2B3C;">Taille</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid #DDE8EE;">
                            <td style="padding:0.55rem; color:#1A2B3C; font-weight:600;">Malaria Cell Images</td>
                            <td style="padding:0.55rem; color:#5E7A8A;">NIH · ceb.nlm.nih.gov</td>
                            <td style="padding:0.55rem; color:#5E7A8A;">27 560 images</td>
                        </tr>
                        <tr style="border-bottom:1px solid #DDE8EE;">
                            <td style="padding:0.55rem; color:#1A2B3C; font-weight:600;">Nutrition CIV</td>
                            <td style="padding:0.55rem; color:#5E7A8A;">UNICEF · OMS</td>
                            <td style="padding:0.55rem; color:#5E7A8A;">Synthétique ↔ réel</td>
                        </tr>
                        <tr style="border-bottom:1px solid #DDE8EE;">
                            <td style="padding:0.55rem; color:#1A2B3C; font-weight:600;">Épidémio CIV</td>
                            <td style="padding:0.55rem; color:#5E7A8A;">PNLP · INTS · OMS</td>
                            <td style="padding:0.55rem; color:#5E7A8A;">20 régions · 36 mois</td>
                        </tr>
                        <tr>
                            <td style="padding:0.55rem; color:#1A2B3C; font-weight:600;">Climatique</td>
                            <td style="padding:0.55rem; color:#5E7A8A;">NASA POWER · CHIRPS · Open-Meteo</td>
                            <td style="padding:0.55rem; color:#5E7A8A;">Temps réel + historique</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT PDF
# ═══════════════════════════════════════════════════════════════════════════════

def _download_report(result: dict, module: str, key: str) -> None:
    """Bouton de téléchargement du rapport PDF (ou HTML en fallback)."""
    try:
        data, mime = build_pdf_report(result, module=module)
        ext  = "pdf" if mime == "application/pdf" else "html"
        label = f"⬇️ Télécharger le rapport ({ext.upper()})"
        st.download_button(label, data, f"rapport_kanea_{module}.{ext}", mime, key=key)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# CTA
# ═══════════════════════════════════════════════════════════════════════════════

def render_cta() -> None:
    st.markdown(
        """
        <div class="cta-container">
            <div class="cta-title">Prêt à révolutionner le diagnostic médical en Afrique ?</div>
            <div class="cta-sub">
                Déployez KANÉA dans votre établissement de santé — entièrement offline, sur CPU standard.
            </div>
            <button class="cta-btn" onclick="window.open('mailto:contact@kanea.ai','_blank')">
                DEMANDER UNE DÉMONSTRATION
            </button>
            <div style="margin-top:2rem; font-size:0.8rem; color:#8AABB8;">
                🧬 KANÉA · MIT Licence · 2026 · Côte d'Ivoire
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_styles()
    inject_science_background()

    if not _MODULES_OK:
        st.error(
            f"⚠️ Erreur de chargement des modules internes : `{_MODULES_ERR}`  \n"
            "Vérifiez que toutes les dépendances sont installées : "
            "`pip install -r requirements.txt`"
        )
        st.stop()

    page, sous_page = render_sidebar()

    # Chargement des données (cached)
    health_df, climate_df = load_data()

    # Routage pages
    if page == "📊 Dashboard":
        page_dashboard(health_df, climate_df)

    elif page == "🦠 Maladies":
        page_maladies(sous_page)

    elif page == "🗺️ Carte":
        page_carte(health_df)

    elif page == "⚙️ Paramètres":
        page_parametres()

    # CTA en bas de chaque page
    st.markdown("<div style='margin-top:3rem'></div>", unsafe_allow_html=True)
    render_cta()


if __name__ == "__main__":
    main()
