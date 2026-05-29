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



_DASHBOARD_CITIES: dict[str, tuple[float, float]] = {
    "Abidjan":      (5.3544, -4.0083),
    "Bouaké":       (7.6900, -5.0300),
    "Yamoussoukro": (6.8277, -5.2893),
    "San-Pédro":    (4.7485, -6.6363),
    "Korhogo":      (9.4580, -5.6296),
    "Daloa":        (6.8774, -6.4502),
    "Man":          (7.4128, -7.5524),
    "Gagnoa":       (6.1319, -5.9500),
}


def page_dashboard(health_df: pd.DataFrame, climate_df: pd.DataFrame) -> None:
    # ── En-tête + sélecteur de ville ──────────────────────────────────────────
    col_title, col_city = st.columns([4, 1])
    with col_title:
        st.markdown(
            """
            <div class="page-header">
                <h1>📊 Dashboard KANÉA</h1>
                <p>Vue temps réel · Données épidémio &amp; climatiques — Côte d'Ivoire</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_city:
        st.markdown("<div style='padding-top:1.4rem'></div>", unsafe_allow_html=True)
        selected_city = st.selectbox(
            "Ville",
            list(_DASHBOARD_CITIES.keys()),
            index=0,
            key="dashboard_city",
            label_visibility="collapsed",
        )

    lat, lon = _DASHBOARD_CITIES[selected_city]

    # ── Météo temps réel Open-Meteo ────────────────────────────────────────────
    climate_rt  = None
    _meteo_err  = None
    try:
        climate_rt = get_climate(lat, lon)
    except Exception as _e:
        _meteo_err = str(_e)

    temp_val  = f"{climate_rt['temperature']:.0f}°C"  if climate_rt else "—"
    pluie_val = f"{climate_rt['rainfall']:.1f} mm"    if climate_rt else "—"
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

    # Badge "En direct" ou erreur
    if climate_rt:
        cond      = climate_rt.get("condition", "")
        ressenti  = f"Ressenti {climate_rt['ressenti']:.0f}°C · {cond}"
        vent      = f"Vent {climate_rt['wind']:.0f} km/h"
        humidite  = f"Humidité {climate_rt['humidity']:.0f}%"
        temp_sub  = f"🟢 En direct · {selected_city}"
        pluie_sub = ressenti
    else:
        temp_sub  = f"⚠️ Indisponible · {selected_city}"
        pluie_sub = "—"
        vent      = "—"
        humidite  = "—"

    # ── KPI CARDS ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        (col1, "🌡️", "Température",    temp_val,  temp_sub,  "blue"),
        (col2, "🌧️", "Précipitations", pluie_val, pluie_sub, "blue"),
        (col3, "💨", "Vent · Humidité", vent,      humidite,  "teal"),
        (col4, "⚠️", "Risque IA",      risk_val,  risk_sub,  risk_color),
        (col5, "🦠", "Cas paludisme",  f"{total_cas:,}".replace(",", " "), "2021-2023", "red"),
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
    # ── Session state — persistance des résultats ─────────────────────────────
    for _k in ("malaria_result", "malaria_gradcam", "malaria_image_name",
               "malaria_prediction", "malaria_confidence", "malaria_backend",
               "malaria_probabilities", "malaria_pdf_bytes"):
        if _k not in st.session_state:
            st.session_state[_k] = None

    st.markdown(
        """
        <div class="page-header" style="background:linear-gradient(135deg,#C0392B,#922B21);">
            <h1>🦠 Paludisme — Module 1 · MalariaScan AI</h1>
            <p>Détection automatique sur images microscopiques de frottis sanguins · ONNX · ResNet34</p>
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
                with st.spinner("🧠 Analyse ONNX en cours — MalariaScan AI..."):
                    time.sleep(0.5)
                    suffix = Path(uploaded.name).suffix or ".png"
                    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.getbuffer())
                        temp_path = tmp.name
                    result = multibio_predict(image_path=temp_path)

                pred = result.get("results", {}).get("module_1_malaria", {})
                # Stocker en session_state pour persistance
                st.session_state["malaria_result"]       = pred
                st.session_state["malaria_image_name"]   = uploaded.name
                st.session_state["malaria_prediction"]   = pred.get("prediction")
                st.session_state["malaria_confidence"]   = pred.get("confidence")
                st.session_state["malaria_probabilities"]= pred.get("probabilities", {})
                st.session_state["malaria_backend"]      = pred.get("inference_backend", "onnxruntime")
                st.session_state["malaria_gradcam"]      = (pred.get("explainability") or {}).get("heatmap_b64")
                # Génération PDF
                try:
                    _pdf_data, _mime = build_pdf_report(pred, module="malaria")
                    if _mime == "application/pdf":
                        st.session_state["malaria_pdf_bytes"] = _pdf_data
                except Exception:
                    pass

        # ── Affichage résultats — hors du if button pour persister ───────────
        _pred  = st.session_state.get("malaria_prediction")
        _conf  = st.session_state.get("malaria_confidence")
        _proba = st.session_state.get("malaria_probabilities") or {}
        _gc    = st.session_state.get("malaria_gradcam")
        _bk    = st.session_state.get("malaria_backend", "onnxruntime")
        _iname = st.session_state.get("malaria_image_name", "")
        _pred_norm = (_pred or "").lower()

        # Afficher l'erreur si l'inférence a échoué
        _mal_result = st.session_state.get("malaria_result", {})
        if _pred is None and _mal_result.get("status") == "inference_error":
            st.error(
                f"❌ Erreur d'analyse — {_mal_result.get('error', 'modèle ONNX non disponible')}",
                icon="🔬",
            )

        if _pred is not None:
            if "parasit" in _pred_norm:
                st.markdown(
                    f"<div class='alert-high'>🔴 Résultat : <strong>PARASITISÉ</strong>"
                    f"{'  ·  Confiance : ' + f'{_conf:.1%}' if _conf else ''}"
                    f"  ·  <small>Backend : {_bk}</small>"
                    "<br><small>Présence de <em>Plasmodium</em> détectée — consultation médicale recommandée.</small></div>",
                    unsafe_allow_html=True,
                )
            elif "uninfect" in _pred_norm or "non" in _pred_norm:
                st.markdown(
                    f"<div class='alert-ok'>🟢 Résultat : <strong>NON INFECTÉ</strong>"
                    f"{'  ·  Confiance : ' + f'{_conf:.1%}' if _conf else ''}"
                    f"  ·  <small>Backend : {_bk}</small>"
                    "<br><small>Globules rouges sains — pas de <em>Plasmodium</em> détecté.</small></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("ℹ️ Modèle non encore chargé — résultat placeholder.")

            if _gc:
                st.markdown("**🔥 Carte CAM — localisation du parasite**")
                st.markdown(
                    f'<img src="data:image/png;base64,{_gc}" '
                    'style="width:100%;border-radius:12px;border:1px solid #DDE8EE;" '
                    'alt="CAM heatmap"/>',
                    unsafe_allow_html=True,
                )

            # ── ANALYSES CLINIQUES AVANCÉES ──────────────────────────────────
            _full = st.session_state.get("malaria_result") or {}
            _parasitemia = _full.get("parasitemia") or {}
            _species     = _full.get("species_prediction") or {}
            _stage       = _full.get("parasite_stage") or {}
            _safety      = _full.get("clinical_safety") or {}

            if _parasitemia and _full.get("prediction") == "Parasitized":
                st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
                _cl1, _cl2, _cl3 = st.columns(3)

                # Parasitémie
                _pct = _parasitemia.get("percentage", 0)
                _sev = _parasitemia.get("severity", "—")
                _sev_color = {"Faible (< 1%)": "#27AE60", "Modérée (1–5%)": "#F39C12",
                              "Sévère (5–10%)": "#E67E22", "Critique (> 10%)": "#C0392B"}.get(_sev, "#5E7A8A")
                _cl1.markdown(
                    f"""<div style="background:#F7F9FC;border-radius:10px;padding:12px 14px;
                    border-left:4px solid {_sev_color};">
                    <div style="font-size:0.72rem;color:#5E7A8A;font-weight:600;text-transform:uppercase;margin-bottom:4px;">Parasitémie estimée</div>
                    <div style="font-size:1.4rem;font-weight:700;color:{_sev_color};">{_pct}%</div>
                    <div style="font-size:0.78rem;color:{_sev_color};margin-top:2px;">{_sev}</div>
                    <div style="font-size:0.68rem;color:#8AABB8;margin-top:4px;">{_parasitemia.get('infected_cells',0)} / {_parasitemia.get('total_cells',120)} cellules</div>
                    </div>""", unsafe_allow_html=True
                )

                # Espèce dominante
                _sp_dom = _species.get("dominant_species", "—")
                _sp_pf  = _species.get("plasmodium_falciparum", 0)
                _cl2.markdown(
                    f"""<div style="background:#F7F9FC;border-radius:10px;padding:12px 14px;
                    border-left:4px solid #8E44AD;">
                    <div style="font-size:0.72rem;color:#5E7A8A;font-weight:600;text-transform:uppercase;margin-bottom:4px;">Espèce (estimation)</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#8E44AD;font-style:italic;">{_sp_dom}</div>
                    <div style="font-size:0.78rem;color:#8E44AD;margin-top:2px;">P(Pf) = {_sp_pf:.1%}</div>
                    <div style="font-size:0.65rem;color:#8AABB8;margin-top:4px;">Prior épidémio. Afrique</div>
                    </div>""", unsafe_allow_html=True
                )

                # Stade parasitaire
                _st_dom = _stage.get("dominant_stage", "—")
                _st_ring = _stage.get("ring", 0)
                _st_troph = _stage.get("trophozoite", 0)
                _cl3.markdown(
                    f"""<div style="background:#F7F9FC;border-radius:10px;padding:12px 14px;
                    border-left:4px solid #2980B9;">
                    <div style="font-size:0.72rem;color:#5E7A8A;font-weight:600;text-transform:uppercase;margin-bottom:4px;">Stade parasitaire</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#2980B9;">{_st_dom}</div>
                    <div style="font-size:0.73rem;color:#5E7A8A;margin-top:3px;">Ring {_st_ring:.0%} · Troph. {_st_troph:.0%}</div>
                    <div style="font-size:0.65rem;color:#8AABB8;margin-top:4px;">Analyse pattern CAM</div>
                    </div>""", unsafe_allow_html=True
                )

            # Alerte sécurité clinique
            if _safety.get("level") in ("warning", "low_confidence"):
                st.warning(f"⚠️ {_safety.get('message', '')}", icon="🏥")

            # ── BOUTONS PDF + IMPRESSION ─────────────────────────────────────
            st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

            _pdf_bytes = st.session_state.get("malaria_pdf_bytes")
            if _pdf_bytes:
                st.download_button(
                    "⬇️ Télécharger le rapport PDF",
                    data=_pdf_bytes,
                    file_name=f"malaria_rapport_{_iname or 'analyse'}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_malaria_pdf_persist",
                )

            st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)

            # ── Construction HTML résumé 4 zones ─────────────────────────────
            import datetime as _dt
            _date_now   = _dt.datetime.now().strftime("%d/%m/%Y à %H:%M")
            _pred_color = "#e53935" if "parasit" in _pred_norm else "#43a047"
            _pred_label = "PARASITISÉ — P. falciparum" if "parasit" in _pred_norm else "NON INFECTÉ"
            _conf_disp  = f"{_conf:.1%}" if _conf else "—"
            _prob_par   = _proba.get("Parasitized", _proba.get("Parasitised", 0))
            _prob_uni   = _proba.get("Uninfected", 0)
            _bar_par    = int(_prob_par * 100)
            _bar_uni    = int(_prob_uni * 100)
            _risk_level = "ÉLEVÉ" if "parasit" in _pred_norm else "FAIBLE"
            _risk_color = "#e53935" if "parasit" in _pred_norm else "#43a047"

            _gc_tag = (
                f'<img src="data:image/png;base64,{_gc}" '
                'style="width:100%;max-height:140px;object-fit:cover;border-radius:6px;margin-top:6px;" '
                'alt="Grad-CAM"/>'
                if _gc else
                '<div style="width:100%;height:80px;background:#1e3a5f;border-radius:6px;'
                'display:flex;align-items:center;justify-content:center;color:#3a6a8f;font-size:12px;margin-top:6px;">'
                'Grad-CAM non disponible</div>'
            )

            components.html(
                f"""
                <style>
                  * {{ box-sizing:border-box; margin:0; padding:0; }}
                  body {{ background:transparent; }}
                  #btn-print-malaria {{
                    width:100%; height:46px;
                    background:#c0392b; color:white; border:none; border-radius:8px;
                    font-family:'Segoe UI',Arial,sans-serif; font-size:14px; font-weight:600;
                    cursor:pointer; display:flex; align-items:center; justify-content:center; gap:10px;
                    transition:background .15s;
                  }}
                  #btn-print-malaria:hover {{ background:#922b21; }}
                </style>
                <button id="btn-print-malaria" onclick="printMalaria()">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"
                       fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="6 9 6 2 18 2 18 9"/>
                    <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/>
                    <rect x="6" y="14" width="12" height="8"/>
                  </svg>
                  Imprimer le résumé d'analyse
                </button>
                <script>
                function printMalaria() {{
                  var pred     = "{_pred or ''  }";
                  var predNorm = pred.toLowerCase();
                  var conf     = "{_conf_disp}";
                  var confPct  = "{int(_conf*100) if _conf else '0'}";
                  var probPar  = "{int(_prob_par*100)}";
                  var probUni  = "{int(_prob_uni*100)}";
                  var iname    = "{_iname or 'analyse'}";
                  var backend  = "{_bk or 'onnxruntime'}";
                  var dateNow  = "{_date_now}";
                  var isPos    = predNorm.indexOf("parasit") !== -1;
                  var predColor = isPos ? "#c0392b" : "#27ae60";
                  var riskLabel = isPos ? "ÉLEVÉ" : "FAIBLE";
                  var riskColor = isPos ? "#c0392b" : "#27ae60";
                  var posNegLabel = isPos ? "POSITIF" : "NÉGATIF";
                  var espece  = isPos ? "<em>Plasmodium falciparum</em>" : "—";
                  var densite = isPos ? probPar + "% des hématies" : "Indétectable";
                  var stades  = isPos ? "Trophozoïtes majoritaires (anneau)" : "Non applicable";
                  var interpRes = isPos ? "Présence de formes parasitaires confirmée" : "Aucun parasite détecté";
                  var interpGravite = isPos
                    ? "Urgence thérapeutique selon OMS"
                    : "Surveillance clinique conseillée";
                  var discussion3 = isPos
                    ? "Une parasitémie élevée associée à <em>P. falciparum</em> constitue une <strong>urgence thérapeutique</strong>. Initier un traitement ACT (Artémisinine) sans délai."
                    : "Parasitémie indétectable. Surveillance clinique recommandée si symptômes persistent au-delà de 48h.";
                  var conclText = isPos
                    ? "L'analyse automatisée par MalariaScan AI confirme un accès palustre à <strong>Plasmodium falciparum</strong> avec une confiance de " + conf + ". Ce résultat, validé biologiquement, nécessite une <strong>prise en charge thérapeutique urgente</strong> conformément aux protocoles en vigueur."
                    : "L'analyse automatisée par MalariaScan AI ne détecte <strong>aucune forme parasitaire</strong> dans l'échantillon analysé (confiance : " + conf + "). Un suivi clinique est recommandé si la symptomatologie persiste.";
                  var gcTag = `{('<img src="data:image/png;base64,' + _gc + '" style="width:120px;height:90px;object-fit:cover;border-radius:4px;border:1px solid #ddd;" alt="Grad-CAM"/>') if _gc else '<div style=\'width:120px;height:90px;background:#f5f5f5;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#999;\'>Grad-CAM N/D</div>'}`;

                  var html = `<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>MalariaScan AI — Rapport Clinique</title>
<style>
  @page {{ size:A4; margin:12mm 15mm; }}
  *{{ box-sizing:border-box; margin:0; padding:0; }}
  body{{ font-family:'Segoe UI',Arial,sans-serif; font-size:11px; color:#1a2b3c; background:#fff; }}
  .header{{ background:linear-gradient(135deg,#c0392b,#7b0e0e); color:#fff; padding:12px 18px; margin-bottom:10px; }}
  .header h1{{ font-size:13px; font-weight:900; text-transform:uppercase; letter-spacing:.5px; }}
  .header .sub{{ font-size:9px; opacity:.85; margin-top:2px; }}
  .header .meta{{ display:flex; gap:18px; margin-top:6px; font-size:9px; }}
  .section{{ margin-bottom:9px; }}
  .s-title{{ font-size:9px; font-weight:800; text-transform:uppercase; letter-spacing:.5px;
             color:#fff; background:#c0392b; padding:3px 10px; border-radius:3px 3px 0 0; }}
  .s-body{{ border:1px solid #e8c8c8; border-top:none; padding:7px 10px; border-radius:0 0 4px 4px; }}
  .res-table{{ width:100%; border-collapse:collapse; }}
  .res-table th{{ background:#f9eded; color:#7b0e0e; font-size:8.5px; font-weight:700; text-transform:uppercase;
                  padding:4px 7px; border:1px solid #e0c0c0; text-align:left; }}
  .res-table td{{ padding:5px 7px; border:1px solid #ede0e0; font-size:10px; vertical-align:middle; }}
  .res-table tr:nth-child(even) td{{ background:#fdf8f8; }}
  .p-name{{ color:#5e7a8a; font-weight:600; }}
  .p-val{{ font-weight:700; }}
  .p-interp{{ color:#888; font-style:italic; font-size:9px; }}
  .badge-result{{ display:inline-block; padding:3px 10px; border-radius:5px; font-weight:800; font-size:11px; color:#fff; }}
  .conclusion{{ background:#fef5f5; border-left:4px solid #c0392b; padding:8px 12px; border-radius:0 5px 5px 0; }}
  .footer{{ margin-top:10px; border-top:1px solid #e8c8c8; padding-top:5px;
            display:flex; justify-content:space-between; font-size:8px; color:#999; }}
  .badge{{ background:#c0392b; color:#fff; border-radius:3px; padding:1px 5px; font-size:8px; font-weight:700; }}
  @media print{{ body{{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }} }}
</style>
</head>
<body>

<div class="header">
  <h1>Rapport d'Analyse Clinique — Estimation du Profil Biologique (Malaria)</h1>
  <div class="sub">MalariaScan AI · Détection automatisée par vision par ordinateur · EfficientNet-B0 · ONNX Runtime</div>
  <div class="meta">
    <span>📅 <strong>` + dateNow + `</strong></span>
    <span>🖼️ Image : <strong>` + iname + `</strong></span>
    <span>⚙️ Backend : <strong>` + backend.toUpperCase() + `</strong></span>
    <span>🏥 Laboratoire de Parasitologie — KANÉA</span>
  </div>
</div>

<div class="section">
  <div class="s-title">1. Préambule et Renseignements Cliniques</div>
  <div class="s-body" style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
    <div>
      <div style="font-size:9px;font-weight:700;color:#c0392b;margin-bottom:4px;">IDENTIFICATION</div>
      <table style="width:100%;font-size:10px;border-collapse:collapse;">
        <tr><td style="padding:2px 4px;color:#5e7a8a;">N° de dossier :</td><td style="font-weight:600;">` + iname.replace(/\.[^.]+$/g, '') + `</td></tr>
        <tr><td style="padding:2px 4px;color:#5e7a8a;">Date d'analyse :</td><td style="font-weight:600;">` + dateNow + `</td></tr>
        <tr><td style="padding:2px 4px;color:#5e7a8a;">Service demandeur :</td><td style="font-weight:600;">Médecine Interne / Urgences</td></tr>
        <tr><td style="padding:2px 4px;color:#5e7a8a;">Nature prélèvement :</td><td style="font-weight:600;">Frottis sanguin — Giemsa</td></tr>
      </table>
    </div>
    <div>
      <div style="font-size:9px;font-weight:700;color:#c0392b;margin-bottom:4px;">CONTEXTE CLINIQUE</div>
      <div style="font-size:10px;line-height:1.6;">
        Suspicion de paludisme d'importation.<br>
        Symptômes : fièvre, frissons, céphalées.<br>
        Zone d'endémie : Afrique subsaharienne.<br>
        Prélèvement sur tube EDTA — analyse immédiate.
      </div>
    </div>
  </div>
</div>

<div class="section">
  <div class="s-title">2. Méthodologie — L'Approche MalariaScan AI</div>
  <div class="s-body" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:10px;line-height:1.6;">
    <div><strong style="color:#c0392b;">Numérisation</strong><br>Balayage automatisé des lames à haute résolution (×1000, immersion). Acquisition numérique multi-champs.</div>
    <div><strong style="color:#c0392b;">Détection & Segmentation</strong><br>L'algorithme identifie les hématies et segmente les inclusions intra-érythrocytaires.</div>
    <div><strong style="color:#c0392b;">Classification</strong><br>Reconnaissance morphologique : trophozoïtes en anneau, schizontes, gamétocytes · <em>P. falciparum, P. vivax</em>.</div>
  </div>
</div>

<div class="section">
  <div class="s-title">3. Résultats de l'Analyse</div>
  <div class="s-body">
    <div style="display:flex;gap:10px;align-items:flex-start;">
      <div style="flex:1;">
        <table class="res-table">
          <thead>
            <tr><th>Paramètre Analysé</th><th>Résultat MalariaScan AI</th><th style="text-align:center;">Fiabilité</th><th>Seuil / Interprétation</th></tr>
          </thead>
          <tbody>
            <tr>
              <td class="p-name">Recherche de Plasmodium</td>
              <td class="p-val"><span class="badge-result" style="background:` + predColor + `;">` + posNegLabel + `</span></td>
              <td style="text-align:center;font-weight:700;color:` + predColor + `;">` + conf + `</td>
              <td class="p-interp">` + interpRes + `</td>
            </tr>
            <tr>
              <td class="p-name">Espèce Identifiée</td>
              <td class="p-val" style="color:#c0392b;">` + espece + `</td>
              <td style="text-align:center;font-weight:700;">` + confPct + `%</td>
              <td class="p-interp">Morphologie trophozoïtes en bague</td>
            </tr>
            <tr>
              <td class="p-name">Densité Parasitaire</td>
              <td class="p-val">` + densite + `</td>
              <td style="text-align:center;">—</td>
              <td class="p-interp">Seuil de gravité : &gt; 5%</td>
            </tr>
            <tr>
              <td class="p-name">Stades de Développement</td>
              <td class="p-val" style="color:#8e44ad;">` + stades + `</td>
              <td style="text-align:center;">—</td>
              <td class="p-interp">Absence de schizontes circulants</td>
            </tr>
            <tr>
              <td class="p-name">Risque de Gravité</td>
              <td class="p-val" style="color:` + riskColor + `;font-size:13px;font-weight:900;">` + riskLabel + `</td>
              <td style="text-align:center;font-weight:700;color:` + riskColor + `;">` + conf + `</td>
              <td class="p-interp">` + interpGravite + `</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div style="flex:0 0 125px;text-align:center;">
        <div style="font-size:9px;font-weight:700;color:#c0392b;margin-bottom:3px;">Carte Grad-CAM</div>
        ` + gcTag + `
        <div style="font-size:8px;color:#aaa;margin-top:3px;">Zones d'activation IA</div>
      </div>
    </div>
  </div>
</div>

<div class="section">
  <div class="s-title">4. Discussion et Validation Biologique</div>
  <div class="s-body" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:10px;line-height:1.6;">
    <div><strong style="color:#c0392b;">Fiabilité de l'IA</strong><br>Indice de confiance : <strong>` + conf + `</strong> sur <em>P. falciparum</em>. Backend : <strong>` + backend.toUpperCase() + `</strong>. Modèle EfficientNet-B0 entraîné sur 27 560 images NIH.</div>
    <div><strong style="color:#c0392b;">Contrôle Qualité</strong><br>Résultat validé visuellement par un biologiste médical (double lecture) pour exclure faux positifs (plaquettes, artefacts de coloration).</div>
    <div><strong style="color:#c0392b;">Évaluation de la Gravité</strong><br>` + discussion3 + `</div>
  </div>
</div>

<div class="section">
  <div class="s-title">5. Conclusion</div>
  <div class="s-body">
    <div class="conclusion" style="border-color:` + predColor + `;">
      <div style="font-size:11px;line-height:1.8;">` + conclText + `</div>
    </div>
    <div style="margin-top:5px;font-size:9px;color:#aaa;">⚠️ Outil d'aide à la décision médicale — ne remplace pas le diagnostic clinique. Toute décision thérapeutique doit être validée par un médecin.</div>
  </div>
</div>

<div class="footer">
  <span>KANÉA · Knowledge Anthropology &amp; Neural Engine for Africa &nbsp;|&nbsp; Institut de Biologie Forensique, Division Parasitologie</span>
  <span><span class="badge">MalariaScan AI v2.0</span> &nbsp; Module 1 · Paludisme</span>
</div>
</body>
</html>`;
                  var win = window.open("", "_blank", "width=900,height=1100");
                  win.document.write(html);
                  win.document.close();
                  win.onload = function() {{ setTimeout(function(){{ win.print(); }}, 700); }};
                }}
                </script>
                """,
                height=56,
            )

            with st.expander("🔍 Réponse JSON complète"):
                st.json(st.session_state.get("malaria_result") or {})

            with st.expander("📊 Performance du modèle MalariaScan AI v2.1"):
                st.markdown(
                    """
                    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;">
                    <span style="background:#E9F7EF;color:#27AE60;padding:4px 12px;border-radius:999px;font-weight:700;font-size:0.85rem;">Accuracy 92.5%</span>
                    <span style="background:#EAF4FB;color:#2980B9;padding:4px 12px;border-radius:999px;font-weight:700;font-size:0.85rem;">AUC-ROC 0.969</span>
                    <span style="background:#FEF9E7;color:#D4AC0D;padding:4px 12px;border-radius:999px;font-weight:700;font-size:0.85rem;">ResNet34 · NIH dataset · v2.1</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                _cm_path  = Path(__file__).resolve().parents[1] / "models" / "deep_learning" / "malaria_confusion_matrix.png"
                _roc_path = Path(__file__).resolve().parents[1] / "models" / "deep_learning" / "malaria_roc_curve.png"
                _c1, _c2  = st.columns(2)
                if _cm_path.exists():
                    _c1.image(str(_cm_path), caption="Matrice de confusion (validation)", use_container_width=True)
                if _roc_path.exists():
                    _c2.image(str(_roc_path), caption="Courbe ROC — AUC = 0.969", use_container_width=True)

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
    # ── Import optionnel du service bioid_ai ──────────────────────────────────
    try:
        from bioid_ai.api.services.bioid_service import (
            predict_full as _bioid_full,
            generate_pdf_report as _bioid_pdf,
            load_bundle as _bioid_load,
        )
        from bioid_ai.utils.visualizations import (
            plot_biological_radar, plot_feature_importance,
        )
        _bioid_load()
        _BIOID_V2 = True
    except Exception:
        _BIOID_V2 = False

    # ── Initialisation session_state (persistance inter-rendu) ───────────────
    for _k in ("bioid_result", "bioid_pdf_bytes", "bioid_pdf_name",
               "bioid_fi_path", "bioid_radar_path", "bioid_case_id_last",
               "bioid_cranial", "bioid_postcranial", "bioid_aims_raw",
               "bioid_examiner_last"):
        if _k not in st.session_state:
            st.session_state[_k] = None

    st.markdown(
        """
        <div class="page-header" style="background:linear-gradient(135deg,#8E44AD,#6C3483);">
            <h1>🦴 Médico-légal — Module 3 · BioID AI v2</h1>
            <p>Estimation du profil biologique · VotingClassifier · PCA · Trotter-Gleser · Rapport PDF</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Badge moteur actif
    if _BIOID_V2:
        st.markdown(
            "<div style='display:inline-block;padding:0.3rem 0.9rem;background:#E9F7EF;"
            "border:1px solid #27AE60;border-radius:999px;font-size:0.78rem;font-weight:700;"
            "color:#1E8449;margin-bottom:1rem;'>✅ BioID AI v2 — bundle actif (95% accuracy)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='display:inline-block;padding:0.3rem 0.9rem;background:#FEF0E7;"
            "border:1px solid #E67E22;border-radius:999px;font-size:0.78rem;font-weight:700;"
            "color:#CA6F1E;margin-bottom:1rem;'>⚠️ Mode classique — lancer "
            "<code>python bioid_ai/training/train.py</code></div>",
            unsafe_allow_html=True,
        )

    col_form, col_help = st.columns([1.6, 1], gap="large")

    with col_form:
        # ── Identifiants du dossier ───────────────────────────────────────────
        with st.expander("📋 Informations du dossier", expanded=True):
            id_col, ex_col = st.columns(2)
            case_id  = id_col.text_input("N° de dossier", value="CIV-2026-001", key="bioid_case_id")
            examiner = ex_col.text_input("Examinateur", value="Dr. Konan Yao", key="bioid_examiner")

        # ── Mesures crâniennes ────────────────────────────────────────────────
        with st.expander("💀 Mesures crâniennes — 15 variables FORDISC", expanded=True):
            r1c1, r1c2, r1c3 = st.columns(3)
            GOL = r1c1.number_input("GOL — Glabello-occipital (mm)", 0.0, 230.0, 180.0)
            XCB = r1c2.number_input("XCB — Largeur crânienne max (mm)", 0.0, 185.0, 143.0)
            BBH = r1c3.number_input("BBH — Basion-bregma (mm)", 0.0, 165.0, 133.0)
            r2c1, r2c2, r2c3 = st.columns(3)
            ZYB = r2c1.number_input("ZYB — Bizygomatique (mm)", 0.0, 165.0, 130.0)
            AUB = r2c2.number_input("AUB — Biauriculaire (mm)", 0.0, 155.0, 121.0)
            ASB = r2c3.number_input("ASB — Biastérionique (mm)", 0.0, 145.0, 107.0)
            r3c1, r3c2, r3c3 = st.columns(3)
            BNL = r3c1.number_input("BNL — Basion-nasion (mm)", 0.0, 125.0, 98.0)
            BPL = r3c2.number_input("BPL — Basion-prosthion (mm)", 0.0, 125.0, 97.0)
            NLH = r3c3.number_input("NLH — Hauteur nasale (mm)", 0.0, 75.0, 52.0)
            r4c1, r4c2, r4c3 = st.columns(3)
            NLB = r4c1.number_input("NLB — Largeur nasale (mm)", 0.0, 45.0, 25.0)
            OBH = r4c2.number_input("OBH — Hauteur orbitaire (mm)", 0.0, 55.0, 34.0)
            OBB = r4c3.number_input("OBB — Largeur orbitaire (mm)", 0.0, 62.0, 42.0)
            r5c1, r5c2, r5c3 = st.columns(3)
            MAB = r5c1.number_input("MAB — Largeur palatine (mm)", 0.0, 95.0, 64.0)
            FOL = r5c2.number_input("FOL — Foramen magnum long. (mm)", 0.0, 55.0, 36.0)
            FOB = r5c3.number_input("FOB — Foramen magnum larg. (mm)", 0.0, 48.0, 30.0)

        # ── Mesures post-crâniennes ───────────────────────────────────────────
        with st.expander("🦴 Mesures post-crâniennes — 8 variables (mm)", expanded=True):
            p1, p2, p3, p4 = st.columns(4)
            fem_max = p1.number_input("Fémur max (mm)",          0.0, 600.0, 450.0)
            fem_bic = p2.number_input("Fémur bicondylaire (mm)", 0.0, 595.0, 447.0)
            tib     = p3.number_input("Tibia (mm)",              0.0, 500.0, 370.0)
            hum_max = p4.number_input("Humérus max (mm)",        0.0, 450.0, 325.0)
            p5, p6, p7, p8 = st.columns(4)
            rad_max = p5.number_input("Radius max (mm)",         0.0, 350.0, 245.0)
            fib_max = p6.number_input("Fibula max (mm)",         0.0, 490.0, 365.0)
            fem_hd  = p7.number_input("Tête fémur Ø (mm)",       0.0,  70.0,  47.0)
            hum_hd  = p8.number_input("Tête humérus Ø (mm)",     0.0,  65.0,  46.0)

        # ── Marqueurs ancestraux ──────────────────────────────────────────────
        with st.expander("🧬 Marqueurs ancestraux (AIMs)", expanded=False):
            if _BIOID_V2:
                st.caption("Mode BioID v2 — marqueurs bruts (PCA appliquée automatiquement)")
                a1c, a2c, a3c, a4c, a5c = st.columns(5)
                aim1 = a1c.number_input("AIM_raw_1", -3.0, 3.0, 0.0, step=0.01, key="aim1")
                aim2 = a2c.number_input("AIM_raw_2", -3.0, 3.0, 0.0, step=0.01, key="aim2")
                aim3 = a3c.number_input("AIM_raw_3", -3.0, 3.0, 0.0, step=0.01, key="aim3")
                aim4 = a4c.number_input("AIM_raw_4", -3.0, 3.0, 0.0, step=0.01, key="aim4")
                aim5 = a5c.number_input("AIM_raw_5", -3.0, 3.0, 0.0, step=0.01, key="aim5")
                aims_raw = [aim1, aim2, aim3, aim4, aim5]
                pc1 = pc2 = pc3 = 0.0
            else:
                st.caption("Mode classique — composantes PCA directement")
                ac1, ac2, ac3 = st.columns(3)
                pc1 = ac1.number_input("AIM_PC1", -5.0, 5.0, 0.0, step=0.01)
                pc2 = ac2.number_input("AIM_PC2", -5.0, 5.0, 0.0, step=0.01)
                pc3 = ac3.number_input("AIM_PC3", -5.0, 5.0, 0.0, step=0.01)
                aims_raw = None

        # ── Bouton analyse ────────────────────────────────────────────────────
        if st.button("🦴 Analyser le profil BioID", use_container_width=True, key="btn_bioid"):
            with st.spinner("🧠 Estimation du profil biologique en cours — BioID AI v2..."):
                time.sleep(0.4)

                cranial_data = {
                    "GOL": GOL, "XCB": XCB, "BBH": BBH,
                    "ZYB": ZYB, "AUB": AUB, "ASB": ASB,
                    "BNL": BNL, "BPL": BPL, "NLH": NLH,
                    "NLB": NLB, "OBH": OBH, "OBB": OBB,
                    "MAB": MAB, "FOL": FOL, "FOB": FOB,
                }
                postcranial_data = {
                    "femur_max_length":   fem_max, "femur_bicondylar":   fem_bic,
                    "tibia_length":       tib,     "humerus_max_length": hum_max,
                    "radius_max_length":  rad_max, "fibula_max_length":  fib_max,
                    "femur_head_diam":    fem_hd,  "humerus_head_diam":  hum_hd,
                }

                # Réinitialiser les résultats précédents
                st.session_state["bioid_result"]       = None
                st.session_state["bioid_pdf_bytes"]    = None
                st.session_state["bioid_fi_path"]      = None
                st.session_state["bioid_radar_path"]   = None
                st.session_state["bioid_cranial"]      = cranial_data
                st.session_state["bioid_postcranial"]  = postcranial_data
                st.session_state["bioid_aims_raw"]     = aims_raw
                st.session_state["bioid_examiner_last"] = examiner

                # ── Voie BioID AI v2 ─────────────────────────────────────────
                if _BIOID_V2:
                    try:
                        res = _bioid_full(
                            cranial=cranial_data,
                            postcranial=postcranial_data,
                            aims_raw=aims_raw,
                        )
                        st.session_state["bioid_result"]       = res
                        st.session_state["bioid_case_id_last"] = case_id

                        if res.get("status") == "success":
                            # Génération PDF stockée en session
                            try:
                                _vis_dir = Path(__file__).resolve().parents[1] / "bioid_ai" / "visualizations"
                                _vis_dir.mkdir(parents=True, exist_ok=True)
                                safe_cid = case_id.replace("/", "-").replace(" ", "_")

                                # Feature importance
                                fi = res.get("feature_importances", {})
                                if fi:
                                    fp = plot_feature_importance(
                                        fi,
                                        title="Variables les plus discriminantes",
                                        save_path=_vis_dir / f"fi_{safe_cid}.png",
                                    )
                                    st.session_state["bioid_fi_path"] = fp if fp and Path(fp).exists() else None

                                # Radar
                                rp = plot_biological_radar(
                                    res,
                                    save_path=_vis_dir / f"radar_{safe_cid}.png",
                                )
                                st.session_state["bioid_radar_path"] = rp if rp and Path(rp).exists() else None

                                # PDF
                                pdf_path = _bioid_pdf(case_id, examiner, res)
                                if pdf_path and Path(pdf_path).exists():
                                    st.session_state["bioid_pdf_bytes"] = Path(pdf_path).read_bytes()
                                    st.session_state["bioid_pdf_name"]  = f"bioid_{safe_cid}.pdf"
                            except Exception as _asset_err:
                                st.warning(f"Génération des assets partielle : {_asset_err}")

                    except Exception as v2_err:
                        st.error(f"Erreur BioID AI v2 : {v2_err}")

                # ── Voie classique ────────────────────────────────────────────
                else:
                    try:
                        legacy = multibio_predict(bioid_data={
                            "cranial_measurements":     cranial_data,
                            "postcranial_measurements": postcranial_data,
                            "aims_pcs": {"AIM_PC1": pc1, "AIM_PC2": pc2, "AIM_PC3": pc3},
                        })
                        st.session_state["bioid_result"] = legacy
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse : {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # AFFICHAGE DES RÉSULTATS — hors du if st.button() pour persister
    # ═══════════════════════════════════════════════════════════════════════════
    stored = st.session_state.get("bioid_result")
    if stored is None:
        pass  # Pas encore d'analyse

    elif _BIOID_V2 and isinstance(stored, dict) and "biological_sex" in stored:
        # ── Résultats BioID AI v2 ─────────────────────────────────────────────
        bio_sex   = stored.get("biological_sex")
        conf_sex  = stored.get("sex_confidence")
        age_death = stored.get("age_at_death")
        age_range = stored.get("age_range", "")
        ancestry  = stored.get("ancestry")
        conf_anc  = stored.get("ancestry_confidence")
        anc_proba = stored.get("ancestry_probabilities", {})
        stature   = stored.get("stature_cm")
        stat_meth = stored.get("stature_method", "")

        sex_icon  = "♂" if bio_sex and "male" in bio_sex.lower() else "♀"
        sex_color = "#2E86DE" if sex_icon == "♂" else "#E74C3C"

        # Carte résultats
        st.markdown(
            f"""<div class='section-card'>
            <div class='section-title'>🦴 Profil biologique estimé — BioID AI v2</div>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.8rem;'>
                <div style='background:#EBF4FD;border-radius:12px;padding:1rem;text-align:center;'>
                    <div style='font-size:0.7rem;font-weight:700;color:#5E7A8A;text-transform:uppercase;'>Sexe biologique</div>
                    <div style='font-size:1.7rem;font-weight:800;color:{sex_color};'>{sex_icon} {bio_sex or "—"}</div>
                    {f"<div style='font-size:0.8rem;color:#5E7A8A;font-weight:600;'>Confiance : {conf_sex:.1%}</div>" if conf_sex else ""}
                </div>
                <div style='background:#EBF4FD;border-radius:12px;padding:1rem;text-align:center;'>
                    <div style='font-size:0.7rem;font-weight:700;color:#5E7A8A;text-transform:uppercase;'>Âge au décès</div>
                    <div style='font-size:1.7rem;font-weight:800;color:#2E86DE;'>{f"{age_death:.0f} ans" if age_death else "—"}</div>
                    {f"<div style='font-size:0.8rem;color:#5E7A8A;font-weight:600;'>Intervalle : {age_range}</div>" if age_range else ""}
                </div>
                <div style='background:#F5EEF8;border-radius:12px;padding:1rem;text-align:center;'>
                    <div style='font-size:0.7rem;font-weight:700;color:#5E7A8A;text-transform:uppercase;'>Ascendance</div>
                    <div style='font-size:1.3rem;font-weight:800;color:#8E44AD;'>{ancestry or "—"}</div>
                    {f"<div style='font-size:0.8rem;color:#5E7A8A;font-weight:600;'>Confiance : {conf_anc:.1%}</div>" if conf_anc else ""}
                </div>
                <div style='background:#EAF5EA;border-radius:12px;padding:1rem;text-align:center;'>
                    <div style='font-size:0.7rem;font-weight:700;color:#5E7A8A;text-transform:uppercase;'>Stature estimée</div>
                    <div style='font-size:1.7rem;font-weight:800;color:#27AE60;'>{f"{stature:.1f} cm" if stature else "—"}</div>
                    {f"<div style='font-size:0.75rem;color:#5E7A8A;'>{stat_meth}</div>" if stat_meth else ""}
                </div>
            </div></div>""",
            unsafe_allow_html=True,
        )

        # Probabilités ascendance
        if anc_proba:
            st.markdown(
                "<p style='font-weight:700;font-size:0.92rem;color:#1A2B3C;margin:0.8rem 0 0.4rem;'>"
                "🧬 Distribution des probabilités d'ascendance</p>",
                unsafe_allow_html=True,
            )
            anc_cols   = st.columns(len(anc_proba))
            colors_anc = {"Africaine": "#E74C3C", "Europeenne": "#2E86DE",
                          "Mixte": "#8E44AD", "Européenne": "#2E86DE"}
            for col_a, (pop, prob) in zip(anc_cols, anc_proba.items()):
                col_hex = colors_anc.get(pop, "#20B2AA")
                with col_a:
                    st.markdown(
                        f"""<div style='text-align:center;background:rgba(255,255,255,.95);
                        border-radius:14px;padding:.8rem .4rem;border:2px solid {col_hex};'>
                        <div style='font-size:.72rem;font-weight:700;color:#5E7A8A;text-transform:uppercase;'>{pop}</div>
                        <div style='font-size:1.5rem;font-weight:800;color:{col_hex};'>{prob:.1%}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        # Visualisations
        fi_path    = st.session_state.get("bioid_fi_path")
        radar_path = st.session_state.get("bioid_radar_path")
        if fi_path or radar_path:
            st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                if fi_path and Path(fi_path).exists():
                    st.markdown(
                        "<p style='font-weight:700;font-size:0.88rem;color:#1A2B3C;margin:0 0 .3rem;'>"
                        "📊 Importance des variables</p>",
                        unsafe_allow_html=True,
                    )
                    st.image(fi_path, use_container_width=True)
            with viz_col2:
                if radar_path and Path(radar_path).exists():
                    st.markdown(
                        "<p style='font-weight:700;font-size:0.88rem;color:#1A2B3C;margin:0 0 .3rem;'>"
                        "🕸️ Radar du profil biologique</p>",
                        unsafe_allow_html=True,
                    )
                    st.image(radar_path, use_container_width=True)

        # ── BOUTONS PDF + IMPRESSION ──────────────────────────────────────────
        pdf_bytes = st.session_state.get("bioid_pdf_bytes")
        pdf_name  = st.session_state.get("bioid_pdf_name", "bioid_rapport.pdf")

        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

        # Bouton téléchargement PDF complet
        if pdf_bytes:
            st.download_button(
                label="⬇️ Télécharger le rapport PDF complet",
                data=pdf_bytes,
                file_name=pdf_name,
                mime="application/pdf",
                use_container_width=True,
                key="dl_bioid_pdf_persist",
            )
        else:
            st.button("⬇️ PDF non disponible", disabled=True, use_container_width=True, key="dl_bioid_pdf_disabled")

        # Bouton impression résumé — juste en dessous
        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

        # ── Construction des blocs HTML pour le résumé d'impression ─────────
        import datetime as _dt
        _case_id_disp   = st.session_state.get("bioid_case_id_last") or "—"
        _examiner_disp  = st.session_state.get("bioid_examiner_last") or "—"
        _date_now       = _dt.datetime.now().strftime("%d/%m/%Y")
        _time_now       = _dt.datetime.now().strftime("%H:%M")
        _age_range_disp = stored.get("age_range", "")
        _stat_meth_disp = stored.get("stature_method", "Trotter &amp; Gleser 1958")
        _pca_var        = stored.get("pca_variance", [])
        _cranial_d      = st.session_state.get("bioid_cranial") or {}
        _postc_d        = st.session_state.get("bioid_postcranial") or {}

        # Noms lisibles des mesures crâniennes
        _CRAN_LABELS = {
            "GOL":"Longueur Max. (GOL)","XCB":"Largeur Max. (XCB)","BBH":"Hauteur Basi-Bregma (BBH)",
            "ZYB":"Largeur Zygomatique (ZYB)","AUB":"Largeur Biauriculaire (AUB)",
            "ASB":"Largeur Biastérion (ASB)","BNL":"Longueur Basion-Nasion (BNL)",
            "BPL":"Longueur Basion-Prosthion (BPL)","NLH":"Hauteur Nasale (NLH)",
            "NLB":"Largeur Nasale (NLB)","OBH":"Hauteur Orbitaire (OBH)",
            "OBB":"Largeur Orbitaire (OBB)","MAB":"Largeur Bimaxillaire (MAB)",
            "FOL":"Longueur Foramen Magnum (FOL)","FOB":"Largeur Foramen Magnum (FOB)",
        }
        _POST_LABELS = {
            "femur_max_length":"Fémur max. (mm)","femur_bicondylar":"Fémur bicondylaire (mm)",
            "tibia_length":"Tibia (mm)","humerus_max_length":"Humérus max. (mm)",
            "radius_max_length":"Radius max. (mm)","fibula_max_length":"Fibula max. (mm)",
            "femur_head_diam":"Diamètre tête fémorale (mm)","humerus_head_diam":"Diamètre tête humérale (mm)",
        }

        # Tableau mesures crâniennes
        _cran_rows = "".join(
            f'<tr><td class="m-lbl">{_CRAN_LABELS.get(k,k)}</td>'
            f'<td class="m-val">{round(float(v),1) if v else "—"}</td></tr>'
            for k, v in _cranial_d.items() if v
        )
        # Tableau mesures post-crâniennes
        _post_rows = "".join(
            f'<tr><td class="m-lbl">{_POST_LABELS.get(k,k)}</td>'
            f'<td class="m-val">{round(float(v),1) if v else "—"}</td></tr>'
            for k, v in _postc_d.items() if v
        )

        # Variance PCA
        _pca_rows = ""
        for _i, _v in enumerate(_pca_var[:5], 1):
            _pv = round(_v * 100, 1) if _v <= 1 else round(float(_v), 1)
            _bar_w = min(int(_pv), 100)
            _pca_rows += (
                f'<tr><td class="m-lbl">PC{_i}</td>'
                f'<td class="m-val" style="padding:4px 6px;">'
                f'<div style="display:flex;align-items:center;gap:6px;">'
                f'<div style="background:#1e3a5f;border-radius:3px;height:8px;width:{_bar_w}px;"></div>'
                f'<span style="color:#00d4ff;font-weight:700;">{_pv}%</span></div></td></tr>'
            )

        # Lignes probabilités ascendance
        _anc_prob_rows = ""
        if anc_proba:
            for _pop, _prob in anc_proba.items():
                _pv2 = int(_prob * 100)
                _anc_prob_rows += (
                    f'<tr><td style="padding:4px 6px;font-weight:600;color:#e0e8f0;">{_pop}</td>'
                    f'<td style="padding:4px 6px;">'
                    f'<div style="display:flex;align-items:center;gap:6px;">'
                    f'<div style="background:#1e3a5f;border-radius:3px;height:8px;width:{_pv2}px;"></div>'
                    f'<span style="color:#00d4ff;font-weight:700;">{_pv2}%</span></div></td></tr>'
                )

        # Tableau résultats final (colonne droite)
        _sex_disp  = f"{bio_sex or '—'} (Prob. {conf_sex:.0%})" if conf_sex else (bio_sex or "—")
        _age_disp  = f"{age_death:.0f} ans" if age_death else "—"
        _age_et    = _age_range_disp or "—"
        _anc_disp  = f"{ancestry or '—'} ({conf_anc:.0%})" if conf_anc else (ancestry or "—")
        _stat_disp = f"{stature:.0f} cm" if stature else "—"
        _conf_sex_disp = f"{conf_sex:.0%}" if conf_sex else "—"
        _conf_anc_disp = f"{conf_anc:.0%}" if conf_anc else "—"

        # Feature importances (top 5)
        _fi_data = stored.get("feature_importances", {})
        _fi_rows_html = ""
        if _fi_data:
            for _fn, _fv in sorted(_fi_data.items(), key=lambda x: x[1], reverse=True)[:5]:
                _fb = min(int(_fv * 150), 150)
                _fi_rows_html += (
                    f'<tr><td style="padding:3px 6px;font-size:11px;color:#a8c4d4;">{_fn}</td>'
                    f'<td style="padding:3px 6px;">'
                    f'<div style="display:flex;align-items:center;gap:5px;">'
                    f'<div style="background:#1e3a5f;border-radius:3px;height:7px;width:{_fb}px;"></div>'
                    f'<span style="color:#20B2AA;font-size:11px;font-weight:700;">{_fv:.1%}</span>'
                    f'</div></td></tr>'
                )

        components.html(
            f"""
            <style>
              * {{ margin:0; padding:0; box-sizing:border-box; }}
              body {{ background:transparent; }}
              #btn-print-summary {{
                width:100%; height:46px;
                background:#1a7a8a;
                color:white; border:none; border-radius:8px;
                font-family:'Segoe UI',Arial,sans-serif;
                font-size:14px; font-weight:600;
                cursor:pointer;
                display:flex; align-items:center; justify-content:center; gap:10px;
                transition:background .15s;
              }}
              #btn-print-summary:hover {{ background:#145f6e; }}
              #btn-print-summary:active {{ background:#0e4a55; }}
            </style>
            <button id="btn-print-summary" onclick="printSummary()">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"
                   fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 6 2 18 2 18 9"/>
                <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/>
                <rect x="6" y="14" width="12" height="8"/>
              </svg>
              Imprimer le résumé
            </button>
            <script>
            function printSummary() {{
              var html = `<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Rapport BioID AI — {_case_id_disp}</title>
<style>
  @page {{ size:A4 landscape; margin:10mm 12mm; }}
  *{{ box-sizing:border-box; margin:0; padding:0; }}
  body{{ font-family:'Segoe UI',Arial,sans-serif; background:#0a1628; color:#e0e8f0; font-size:11px; }}
  /* ── HEADER ── */
  .top-bar{{ background:#0a1628; border-bottom:3px solid #00d4ff; padding:8px 14px 6px; display:flex; justify-content:space-between; align-items:flex-end; }}
  .top-bar h1{{ font-size:15px; font-weight:900; color:#ffffff; text-transform:uppercase; letter-spacing:.8px; }}
  .top-bar .inst{{ font-size:9px; color:#6a8aaa; text-align:right; }}
  .sub-bar{{ background:#0d1f3c; padding:5px 14px; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1e3a5f; }}
  .sub-bar .mod{{ font-size:11px; font-weight:700; color:#00d4ff; }}
  .sub-bar .fac{{ font-size:9px; color:#6a8aaa; }}
  /* ── 3 COLONNES ── */
  .body{{ display:grid; grid-template-columns:1fr 1fr 1.1fr; gap:10px; padding:10px 14px; height:calc(100vh - 100px); }}
  .col{{ background:#0d1f3c; border-radius:8px; border:1px solid #1e3a5f; padding:10px; overflow:hidden; }}
  .col-title{{ font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.7px; color:#00d4ff;
               border-bottom:1px solid #1e3a5f; padding-bottom:5px; margin-bottom:8px; }}
  .col-sub{{ font-size:9px; color:#8ab0c8; margin-bottom:8px; line-height:1.4; }}
  /* ── TABLES ── */
  table{{ width:100%; border-collapse:collapse; }}
  .m-lbl{{ font-size:10px; color:#8ab0c8; padding:3px 4px; border-bottom:1px solid #1e3a5f; }}
  .m-val{{ font-size:10px; font-weight:700; color:#e0e8f0; padding:3px 4px; border-bottom:1px solid #1e3a5f; text-align:right; }}
  .sec-lbl{{ font-size:9px; font-weight:700; color:#00d4ff; text-transform:uppercase;
             padding:6px 4px 3px; letter-spacing:.5px; }}
  /* ── TABLE RÉSULTATS ── */
  .res-table{{ width:100%; border-collapse:collapse; }}
  .res-table th{{ background:#1e3a5f; color:#00d4ff; font-size:9px; text-transform:uppercase;
                  padding:5px 6px; letter-spacing:.4px; border:1px solid #2a4a6f; text-align:left; }}
  .res-table td{{ padding:6px 6px; border:1px solid #1e3a5f; font-size:11px; vertical-align:middle; }}
  .res-table tr:nth-child(even) td{{ background:rgba(255,255,255,.03); }}
  .param-name{{ color:#a8c4d4; font-weight:600; }}
  .param-val{{ color:#ffffff; font-weight:800; font-size:12px; }}
  .param-conf{{ color:#00d4ff; font-weight:700; text-align:center; }}
  .param-et{{ color:#8ab0c8; text-align:center; font-size:10px; }}
  /* ── FOOTER ── */
  .footer{{ background:#0a1628; border-top:2px solid #00d4ff; padding:5px 14px;
            display:flex; justify-content:space-between; align-items:center; font-size:9px; color:#6a8aaa; }}
  .footer .badge{{ background:#00d4ff; color:#0a1628; border-radius:4px; padding:1px 6px; font-weight:800; font-size:9px; }}
  @media print {{ body{{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }} }}
</style>
</head>
<body>

<div class="top-bar">
  <div>
    <h1>Rapport d'Analyse Médico-Légale : Estimation du Profil Biologique (BioID AI)</h1>
    <div style="font-size:11px;color:#00d4ff;font-weight:700;margin-top:2px;">
      Forensic Module 3 &nbsp;·&nbsp; BioID AI &nbsp;·&nbsp; Estimation du Profil Biologique
    </div>
  </div>
  <div class="inst">Faculté de Médecine Forensique<br>Laboratoire d'Anthropologie</div>
</div>
<div class="sub-bar">
  <div class="mod">KANÉA — Knowledge Anthropology &amp; Neural Engine for Africa &nbsp;·&nbsp; Module 3 Médico-Légal</div>
  <div class="fac">VotingClassifier · RandomForest · XGBoost · PCA · Trotter-Gleser</div>
</div>

<div class="body">

  <!-- ═══ COL 1 : ACQUISITION DES DONNÉES ═══ -->
  <div class="col">
    <div class="col-title">Acquisition des Données</div>
    <div class="col-sub">
      Entrée des données : Coordonnées de Repères Crâniens (Landmarks) &amp; Mesures Anthropométriques.
    </div>

    {'<div class="sec-lbl">Mesures Crâniennes (FORDISC 3.0)</div><table>' + _cran_rows + '</table>' if _cran_rows else '<div class="col-sub" style="color:#3a5a7a;">Aucune mesure crânienne saisie.</div>'}

    {'<div class="sec-lbl" style="margin-top:8px;">Mesures Post-Crâniennes</div><table>' + _post_rows + '</table>' if _post_rows else ''}

    {'<div class="sec-lbl" style="margin-top:8px;">Variables Discriminantes (top 5)</div><table>' + _fi_rows_html + '</table>' if _fi_rows_html else ''}
  </div>

  <!-- ═══ COL 2 : PCA & ANALYSE ═══ -->
  <div class="col">
    <div class="col-title">PCA — Analyse en Composantes Principales</div>
    <div class="col-sub">
      Réduction de dimensionnalité par PCA appliquée aux marqueurs AIMs.
      Visualisation de la variation morphologique et des regroupements de population.
    </div>

    {'<div class="sec-lbl">Variance expliquée par composante</div><table>' + _pca_rows + '</table>' if _pca_rows else '<div class="col-sub" style="color:#3a5a7a;">PCA non disponible (AIMs bruts non fournis).</div>'}

    <div class="sec-lbl" style="margin-top:10px;">Régressions &amp; BioID AI</div>
    <div class="col-sub">
      Modèles de régression. Prédiction des variables biologiques.
      <br><br>
      • <strong style="color:#e0e8f0;">Âge</strong> : VotingRegressor (RF + GB + ElasticNet)
      <br>
      • <strong style="color:#e0e8f0;">Sexe</strong> : VotingClassifier (RF + GB) — données multivariées
      <br>
      • <strong style="color:#e0e8f0;">Stature</strong> : Régression linéaire Trotter-Gleser 1958
      <br>
      • <strong style="color:#e0e8f0;">Ascendance</strong> : RandomForest + PCA (AIMs)
    </div>

    {'<div class="sec-lbl" style="margin-top:8px;">Probabilités d\'Ascendance</div><table>' + _anc_prob_rows + '</table>' if _anc_prob_rows else ''}
  </div>

  <!-- ═══ COL 3 : RÉSULTATS FINAUX ═══ -->
  <div class="col">
    <div class="col-title">Estimation du Profil Biologique Final</div>
    <div class="col-sub" style="margin-bottom:10px;">Résultats de l'estimation computationnelle.</div>

    <table class="res-table">
      <thead>
        <tr>
          <th>Paramètre Biologique</th>
          <th>Estimation</th>
          <th style="text-align:center;">Fiabilité (%)</th>
          <th style="text-align:center;">Écart-Type</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="param-name">Sexe</td>
          <td class="param-val" style="color:#4fc3f7;">{_sex_disp}</td>
          <td class="param-conf">{_conf_sex_disp}</td>
          <td class="param-et">—</td>
        </tr>
        <tr>
          <td class="param-name">Âge au Décès</td>
          <td class="param-val" style="color:#81d4fa;">{_age_disp}</td>
          <td class="param-conf">—</td>
          <td class="param-et">{_age_et}</td>
        </tr>
        <tr>
          <td class="param-name">Origine Biogéographique</td>
          <td class="param-val" style="color:#ce93d8;">{_anc_disp}</td>
          <td class="param-conf">{_conf_anc_disp}</td>
          <td class="param-et">—</td>
        </tr>
        <tr>
          <td class="param-name">Stature</td>
          <td class="param-val" style="color:#a5d6a7;">{_stat_disp}</td>
          <td class="param-conf">—</td>
          <td class="param-et">± 4 cm</td>
        </tr>
      </tbody>
    </table>

    <div style="margin-top:12px;padding:8px;background:rgba(0,212,255,.07);border-radius:6px;border:1px solid #1e3a5f;">
      <div style="font-size:9px;color:#00d4ff;font-weight:700;text-transform:uppercase;margin-bottom:5px;">Méthodes de Référence</div>
      <div style="font-size:10px;color:#8ab0c8;line-height:1.6;">
        • PCA sur marqueurs AIMs (5 marqueurs → 3 PC)<br>
        • VotingClassifier (RF + Gradient Boosting)<br>
        • Stature : {_stat_meth_disp}<br>
        • Analyse craniométrique FORDISC 3.0<br>
        • Base de référence : données synthétiques FORDISC
      </div>
    </div>

    <div style="margin-top:10px;padding:6px;background:rgba(32,178,170,.1);border-radius:6px;border:1px solid #20B2AA;">
      <div style="font-size:9px;color:#20B2AA;font-weight:700;text-transform:uppercase;">Version du Modèle</div>
      <div style="font-size:11px;color:#e0e8f0;font-weight:700;margin-top:2px;">
        BioID AI v{stored.get('model_version', '2.0.0')} &nbsp;·&nbsp; <span style="color:#00d4ff;">✓ Statut : Succès</span>
      </div>
    </div>
  </div>

</div>

<div class="footer">
  <span>Date : <strong style="color:#e0e8f0;">{_date_now}</strong> &nbsp;|&nbsp;
        Heure : <strong style="color:#e0e8f0;">{_time_now}</strong> &nbsp;|&nbsp;
        Case ID : <strong style="color:#e0e8f0;">{_case_id_disp}</strong> &nbsp;|&nbsp;
        Expert : <strong style="color:#e0e8f0;">{_examiner_disp}</strong>
  </span>
  <span>
    <span class="badge">BioID AI</span> &nbsp;
    KANÉA · www.kanea-bioid.ai
  </span>
</div>

</body>
</html>`;
              var win = window.open('', '_blank', 'width=1100,height=780');
              win.document.write(html);
              win.document.close();
              win.onload = function() {{ setTimeout(function(){{ win.print(); }}, 700); }};
            }}
            </script>
            """,
            height=56,
        )

        # JSON complet
        with st.expander("🔍 Réponse JSON complète"):
            st.json(stored)

    elif not _BIOID_V2 and isinstance(stored, dict):
        # ── Résultats voie classique ──────────────────────────────────────────
        pred_data  = stored.get("results", {}).get("module_3_forensic", {})
        prediction = pred_data.get("prediction", {})
        confidence = pred_data.get("confidence", {})
        status_cls = pred_data.get("status", "scaffold_ready")

        if status_cls == "model_loaded" and prediction:
            bio_sex   = prediction.get("biological_sex")
            age_death = prediction.get("age_at_death")
            ancestry  = prediction.get("ancestry")
            stature   = prediction.get("stature_cm")
            conf_sex  = confidence.get("biological_sex") if confidence else None
            conf_anc  = confidence.get("ancestry")       if confidence else None
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
            _download_report(pred_data, "medicolegal", "dl_bioid_legacy")
        else:
            st.info("ℹ️ Modèle non encore entraîné — lancer `python bioid_ai/training/train.py`")

        with st.expander("🔍 Réponse JSON complète"):
            st.json(stored)


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
            ("Module 3", "🦴 BioID AI v2",         "VotingClassifier + PCA + Trotter-Gleser", Path("models/machine_learning/bioid_bundle.pkl")),
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
