"""
KANÉA — Page Analyse Patient Multi-Images
==========================================
Le médecin entre le nom du patient, uploade N photos (1–50),
l'app détecte automatiquement le nombre, analyse tout et donne
un diagnostic consolidé plus fiable.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# Modules concernés par l'analyse multi-images
MULTI_IMAGE_MODULES = {
    "malaria":       {"name": "MalariaScan AI",    "icon": "🦟", "color": "#E74C3C",
                      "types": ["PNG","JPG","JPEG"], "desc": "Frottis sanguins — détection paludisme"},
    "breast_cancer": {"name": "Breast Cancer AI",  "icon": "🩺", "color": "#D91E7A",
                      "types": ["PNG","JPG","JPEG","DCM"], "desc": "Mammographies — Normal / Bénin / Malin"},
    "derm":          {"name": "DermAI",             "icon": "🔬", "color": "#8E44AD",
                      "types": ["PNG","JPG","JPEG"], "desc": "Lésions cutanées — Mélanome / Carcinome"},
    "retina":        {"name": "RetinaVision AI",   "icon": "👁️", "color": "#2980B9",
                      "types": ["PNG","JPG","JPEG"], "desc": "Fond d'œil — Rétinopathie / Glaucome"},
    "pulmoscan":     {"name": "PulmoScan AI",      "icon": "🫁", "color": "#1ABC9C",
                      "types": ["PNG","JPG","JPEG","DCM"], "desc": "Radiographies thoraciques"},
    "gastro":        {"name": "GastroAI",           "icon": "🩻", "color": "#E67E22",
                      "types": ["PNG","JPG","JPEG"], "desc": "Endoscopie — Polypes / Ulcères"},
    "histopath":     {"name": "HistoPath AI",       "icon": "🧬", "color": "#16A085",
                      "types": ["PNG","JPG","TIFF"], "desc": "Lames histologiques — Grading tumoral"},
    "osteo":         {"name": "OsteoDetect AI",     "icon": "🦴", "color": "#7F8C8D",
                      "types": ["PNG","JPG","DCM"],  "desc": "Radiographies osseuses — Fractures"},
    "hemato":        {"name": "HematoVision AI",    "icon": "🩸", "color": "#C0392B",
                      "types": ["PNG","JPG","JPEG"], "desc": "Frottis sanguins — Leucémies / Anémies"},
    "gyno":          {"name": "GynoCare AI",        "icon": "🎗️", "color": "#E91E8C",
                      "types": ["PNG","JPG","JPEG"], "desc": "Cytologie cervicale — CIN / Cancer"},
    "neuro":         {"name": "NeuroVision AI",     "icon": "🧠", "color": "#8E44AD",
                      "types": ["PNG","JPG","DCM","NII"], "desc": "IRM cérébrale — Tumeurs / AVC"},
}

_RELIABILITY_ICONS = {
    "Très élevée": "🟢",
    "Élevée":      "🟡",
    "Modérée":     "🟠",
    "Faible":      "🔴",
}


def render_patient_analysis() -> None:
    st.markdown(
        """
        <div class="page-header" style="background:linear-gradient(135deg,#1A3A5C,#2471A3);">
            <h1>👤 Analyse Patient — Multi-Images</h1>
            <p>Uploadez 1 à 50 photos · L'IA analyse automatiquement et donne un diagnostic consolidé</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Session state ─────────────────────────────────────────────────────────
    for key in ("pa_result", "pa_running"):
        if key not in st.session_state:
            st.session_state[key] = None
    if "pa_running" not in st.session_state:
        st.session_state["pa_running"] = False

    # ── Formulaire ────────────────────────────────────────────────────────────
    col_form, col_result = st.columns([1, 1.4], gap="large")

    with col_form:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📋 Informations patient</div>", unsafe_allow_html=True)

        patient_name = st.text_input(
            "Nom complet du patient *",
            placeholder="ex : Kouassi Jean-Baptiste",
            key="pa_patient_name",
        )

        dob = st.text_input(
            "Date de naissance (optionnel)",
            placeholder="ex : 15/03/1985",
            key="pa_dob",
        )

        st.markdown("---")
        st.markdown("<div class='section-title'>🔬 Module d'analyse</div>", unsafe_allow_html=True)

        module_options = {
            f"{cfg['icon']} {cfg['name']}": key
            for key, cfg in MULTI_IMAGE_MODULES.items()
        }
        selected_label = st.selectbox(
            "Choisir le module IA",
            options=list(module_options.keys()),
            key="pa_module",
        )
        module_key  = module_options[selected_label]
        module_cfg  = MULTI_IMAGE_MODULES[module_key]

        st.markdown(
            f"""<div style="background:#F0F4F8;border-left:3px solid {module_cfg['color']};
            padding:8px 12px;border-radius:0 8px 8px 0;font-size:0.82rem;color:#5E7A8A;margin:6px 0;">
            {module_cfg['desc']}
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("<div class='section-title'>📸 Images médicales</div>", unsafe_allow_html=True)

        accepted_types = [t.lower() for t in module_cfg["types"]]
        uploaded_files = st.file_uploader(
            f"Importer les images ({', '.join(module_cfg['types'])})",
            type=accepted_types,
            accept_multiple_files=True,
            key="pa_files",
            help="Recommandé : 30 à 50 images pour un diagnostic fiable. Minimum : 1 image.",
        )

        n_files = len(uploaded_files) if uploaded_files else 0

        if n_files > 0:
            # Barre de fiabilité estimée
            if n_files >= 30:
                fiab_color = "#27AE60"; fiab_txt = "Excellente fiabilité attendue"
            elif n_files >= 15:
                fiab_color = "#F39C12"; fiab_txt = "Bonne fiabilité"
            elif n_files >= 5:
                fiab_color = "#E67E22"; fiab_txt = "Fiabilité modérée"
            else:
                fiab_color = "#E74C3C"; fiab_txt = "Fiabilité faible — ajoutez plus de photos"

            st.markdown(
                f"""<div style="background:#F7F9FC;border-radius:10px;padding:10px 14px;
                border-left:4px solid {fiab_color};">
                <div style="font-size:0.82rem;font-weight:700;color:{fiab_color};">
                📸 {n_files} photo{'s' if n_files > 1 else ''} reçue{'s' if n_files > 1 else ''}
                </div>
                <div style="font-size:0.76rem;color:#5E7A8A;margin-top:2px;">{fiab_txt}</div>
                <div style="background:#E8F5E9;border-radius:4px;height:6px;margin-top:6px;">
                <div style="background:{fiab_color};width:{min(n_files/50*100,100):.0f}%;
                height:6px;border-radius:4px;"></div></div>
                <div style="font-size:0.70rem;color:#AAB7C4;margin-top:3px;">
                {n_files}/50 images</div></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div style="border:2px dashed #DDE8EE;border-radius:10px;padding:16px;
                text-align:center;color:#8AABB8;font-size:0.82rem;">
                Aucune image — importez entre 1 et 50 photos</div>""",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Bouton analyse
        btn_disabled = not patient_name or n_files == 0
        if st.button(
            f"🔬 Analyser {n_files} image{'s' if n_files != 1 else ''} — {patient_name or '...'}",
            use_container_width=True,
            disabled=btn_disabled,
            key="pa_btn_analyze",
        ):
            _run_analysis(
                uploaded_files=uploaded_files,
                module_key=module_key,
                module_name=module_cfg["name"],
                patient_name=patient_name.strip(),
            )

        if btn_disabled and not patient_name:
            st.caption("⚠️ Entrez le nom du patient pour continuer")

    # ── Résultats ─────────────────────────────────────────────────────────────
    with col_result:
        report = st.session_state.get("pa_result")

        if report:
            _render_report(report, module_cfg)
        else:
            st.markdown(
                """<div style="border:2px dashed #DDE8EE;border-radius:14px;padding:3rem;
                text-align:center;color:#8AABB8;">
                <div style="font-size:3rem;margin-bottom:.5rem;">👤</div>
                <div style="font-size:.92rem;font-weight:600;">Résultats du patient</div>
                <div style="font-size:.78rem;margin-top:.4rem;opacity:.7;">
                Entrez le nom du patient, sélectionnez le module,<br>
                importez les photos et lancez l'analyse</div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='disclaimer' style='margin-top:1rem;'>⚠️ Outil d'aide à la décision médicale. "
        "Le diagnostic final doit être validé par un médecin spécialiste.</div>",
        unsafe_allow_html=True,
    )


def _run_analysis(
    uploaded_files: list,
    module_key: str,
    module_name: str,
    patient_name: str,
) -> None:
    from src.multi_image_analyzer import analyze_patient_images

    progress_bar = st.progress(0.0)
    status_text  = st.empty()

    def on_progress(pct: float, msg: str) -> None:
        progress_bar.progress(pct)
        status_text.markdown(
            f"<div style='font-size:0.80rem;color:#5E7A8A;'>⚙️ {msg}</div>",
            unsafe_allow_html=True,
        )

    with st.spinner(f"Analyse de {len(uploaded_files)} images en cours..."):
        try:
            report = analyze_patient_images(
                uploaded_files=uploaded_files,
                module_key=module_key,
                module_name=module_name,
                patient_name=patient_name,
                progress_callback=on_progress,
            )
            st.session_state["pa_result"] = report
        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")
            return
        finally:
            progress_bar.empty()
            status_text.empty()


def _render_report(report: Any, module_cfg: dict) -> None:
    from src.multi_image_analyzer import MultiImageReport
    import plotly.graph_objects as go

    icon_rel = _RELIABILITY_ICONS.get(report.reliability_level, "⚪")
    color    = report.reliability_color

    # ── En-tête patient ───────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="background:linear-gradient(135deg,#1A3A5C,#2471A3);
        border-radius:14px;padding:1rem 1.4rem;color:#fff;margin-bottom:0.8rem;">
        <div style="font-size:0.72rem;opacity:.7;text-transform:uppercase;letter-spacing:1px;">
        Rapport d'analyse — {report.analyzed_at[:10]}</div>
        <div style="font-size:1.4rem;font-weight:800;margin:4px 0;">
        👤 {report.patient_name}</div>
        <div style="font-size:0.82rem;opacity:.8;">
        {module_cfg['icon']} {report.module_name} · {report.n_images} image{'s' if report.n_images > 1 else ''} analysée{'s' if report.n_images > 1 else ''}
        · ID {report.report_id}</div></div>""",
        unsafe_allow_html=True,
    )

    # ── Diagnostic final ──────────────────────────────────────────────────────
    _BG_MAP = {
        "#27AE60": "#EAFAF1", "#2ECC71": "#EAFAF1",
        "#F39C12": "#FEF9E7", "#E74C3C": "#FDECEA",
    }
    bg = _BG_MAP.get(color, "#F7F9FC")

    st.markdown(
        f"""<div style="background:{bg};border-left:6px solid {color};
        border-radius:0 14px 14px 0;padding:1.2rem 1.4rem;margin-bottom:0.8rem;">
        <div style="font-size:0.72rem;color:#5E7A8A;font-weight:600;text-transform:uppercase;">
        Diagnostic consolidé</div>
        <div style="font-size:1.3rem;font-weight:800;color:{color};margin:4px 0;">
        {module_cfg['icon']} {report.final_diagnosis}</div>
        <div style="font-size:0.84rem;color:#5E7A8A;">
        {icon_rel} Fiabilité <b style="color:{color};">{report.reliability_level}</b>
        &nbsp;·&nbsp; Consensus <b>{report.consensus_score:.0%}</b>
        &nbsp;·&nbsp; Confiance moyenne <b>{report.confidence_mean:.1%}</b>
        </div></div>""",
        unsafe_allow_html=True,
    )

    # ── Métriques ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📸 Images", report.n_images)
    c2.metric("🎯 Consensus", f"{report.consensus_score:.0%}")
    c3.metric("📊 Confiance", f"{report.confidence_mean:.1%}")
    c4.metric("🔒 Fiabilité", report.reliability_level)

    # ── Graphique distribution ────────────────────────────────────────────────
    if report.class_distribution:
        labels = list(report.class_distribution.keys())
        values = list(report.class_distribution.values())
        colors_pie = [
            module_cfg["color"] if lbl == report.final_diagnosis else "#BDC3C7"
            for lbl in labels
        ]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker_colors=colors_pie,
            textinfo="label+percent",
            textfont_size=11,
        )])
        fig.update_layout(
            title=dict(
                text=f"Distribution sur {report.n_images} images",
                font_size=12, x=0.5,
            ),
            showlegend=True,
            height=260,
            margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Recommandation ────────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="background:#EBF5FB;border-left:4px solid #2471A3;
        border-radius:0 10px 10px 0;padding:10px 14px;margin:6px 0;">
        <div style="font-size:0.72rem;color:#5E7A8A;font-weight:600;text-transform:uppercase;
        margin-bottom:4px;">Recommandation clinique</div>
        <div style="font-size:0.85rem;color:#1A2B3C;">{report.recommendation}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Alertes ───────────────────────────────────────────────────────────────
    if report.alerts:
        with st.expander(f"⚠️ {len(report.alerts)} alerte(s) détectée(s)"):
            for alert in report.alerts:
                lvl  = alert.get("level", "INFO")
                _bg  = {"CRITICAL": "#FDECEA", "HIGH": "#FEF0E7"}.get(lvl, "#EBF4FD")
                _brd = {"CRITICAL": "#C0392B", "HIGH": "#E74C3C"}.get(lvl, "#2E86DE")
                st.markdown(
                    f"""<div style="background:{_bg};border-left:3px solid {_brd};
                    padding:6px 10px;border-radius:0 6px 6px 0;margin-bottom:4px;">
                    <b style="font-size:0.80rem;">{alert.get('icon','')} {alert.get('title','')}</b><br>
                    <span style="font-size:0.74rem;color:#5E7A8A;">{alert.get('message','')}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ── Détail image par image ────────────────────────────────────────────────
    with st.expander(f"🔍 Détail image par image ({report.n_images} résultats)"):
        rows = []
        for i, pred in enumerate(report.individual_predictions, 1):
            rows.append({
                "#": i,
                "Fichier": pred.image_name[:35],
                "Diagnostic": pred.prediction,
                "Confiance": f"{pred.confidence:.1%}",
                "Accord": "✓" if pred.prediction == report.final_diagnosis else "✗",
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Export PDF ────────────────────────────────────────────────────────────
    pdf_bytes = _build_patient_pdf(report, module_cfg)
    if pdf_bytes:
        safe_name = report.patient_name.replace(" ", "_")
        st.download_button(
            f"📥 Télécharger rapport PDF — {report.patient_name}",
            data=pdf_bytes,
            file_name=f"KANEA_{report.module_key}_{safe_name}_{report.report_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_patient_pdf",
        )


def _build_patient_pdf(report: Any, module_cfg: dict) -> bytes | None:
    """Génère le rapport PDF patient multi-images."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        )
        import io

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                topMargin=14*mm, bottomMargin=14*mm,
                                leftMargin=17*mm, rightMargin=17*mm)
        W = A4[0] - 34*mm

        _DARK  = colors.HexColor("#1C2833")
        _WHITE = colors.white
        _BLUE  = colors.HexColor("#1A3A5C")
        _MUTED = colors.HexColor("#566573")
        _LIGHT = colors.HexColor("#EBF5FB")

        try:
            rel_color = colors.HexColor(report.reliability_color.lstrip("#"))
        except Exception:
            rel_color = colors.HexColor("#566573")

        s_h1 = ParagraphStyle("H1", fontSize=13, fontName="Helvetica-Bold",
                               textColor=_WHITE, leading=16)
        s_h2 = ParagraphStyle("H2", fontSize=10, fontName="Helvetica-Bold",
                               textColor=_DARK, leading=13, spaceAfter=3)
        s_b  = ParagraphStyle("B", fontSize=8.5, fontName="Helvetica",
                               textColor=_DARK, leading=11)
        s_m  = ParagraphStyle("M", fontSize=7.5, fontName="Helvetica",
                               textColor=_MUTED, leading=10)

        story = []

        # Header
        hdr = Table([[Paragraph(
            f'<font size="14"><b>KANÉA — Rapport Patient Multi-Images</b></font><br/>'
            f'<font size="8.5">{module_cfg["icon"]} {report.module_name} · '
            f'{report.n_images} images analysées · {report.analyzed_at[:10]}</font>',
            s_h1,
        )]], colWidths=[W])
        hdr.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,-1), _BLUE),
            ("TOPPADDING", (0,0),(-1,-1), 10),
            ("BOTTOMPADDING", (0,0),(-1,-1), 10),
            ("LEFTPADDING", (0,0),(-1,-1), 12),
        ]))
        story.extend([hdr, Spacer(1, 4*mm)])

        # Patient info
        story.append(Paragraph("1. Informations Patient", s_h2))
        info = Table([
            ["Patient",   report.patient_name, "ID Rapport", report.report_id],
            ["Module IA", report.module_name,  "Date",       report.analyzed_at[:16]],
            ["Images",    str(report.n_images), "Fiabilité", report.reliability_level],
        ], colWidths=[W*0.15, W*0.35, W*0.15, W*0.35])
        info.setStyle(TableStyle([
            ("FONTNAME",   (0,0),(-1,-1), "Helvetica"),
            ("FONTSIZE",   (0,0),(-1,-1), 8),
            ("FONTNAME",   (0,0),(0,-1),  "Helvetica-Bold"),
            ("FONTNAME",   (2,0),(2,-1),  "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,0),(-1,-1), [_LIGHT, _WHITE]),
            ("GRID",       (0,0),(-1,-1), 0.3, colors.HexColor("#D5E8F0")),
            ("TOPPADDING", (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING", (0,0),(-1,-1), 6),
        ]))
        story.extend([info, Spacer(1, 4*mm)])

        # Diagnostic final
        story.append(Paragraph("2. Diagnostic Consolidé", s_h2))
        diag = Table([[
            Paragraph(f"<b>Diagnostic :</b> {report.final_diagnosis}", s_b),
            Paragraph(f"<b>Consensus :</b> {report.consensus_score:.0%}", s_b),
            Paragraph(f"<b>Confiance :</b> {report.confidence_mean:.1%}", s_b),
        ]], colWidths=[W*0.40, W*0.30, W*0.30])
        diag.setStyle(TableStyle([
            ("FONTSIZE",    (0,0),(-1,-1), 9),
            ("BACKGROUND",  (0,0),(-1,-1), _LIGHT),
            ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#D5E8F0")),
            ("LINEBEFORE",  (0,0),(0,-1),  5, rel_color),
            ("TOPPADDING",  (0,0),(-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LEFTPADDING", (0,0),(-1,-1), 8),
        ]))
        story.extend([diag, Spacer(1, 4*mm)])

        # Distribution
        story.append(Paragraph("3. Distribution des résultats", s_h2))
        dist_rows = [["Classe", "Nb images", "% images", "Confiance moy."]]
        for cls, cnt in sorted(report.class_distribution.items(), key=lambda x: -x[1]):
            dist_rows.append([
                cls,
                str(cnt),
                f"{cnt/report.n_images:.0%}",
                f"{report.class_confidence.get(cls, 0):.1%}",
            ])
        dist_t = Table(dist_rows, colWidths=[W*0.40, W*0.18, W*0.18, W*0.24])
        dist_t.setStyle(TableStyle([
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 8),
            ("BACKGROUND",    (0,0),(-1,0), _BLUE),
            ("TEXTCOLOR",     (0,0),(-1,0), _WHITE),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [_LIGHT, _WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#D5E8F0")),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ]))
        story.extend([dist_t, Spacer(1, 4*mm)])

        # Recommandation
        story.append(Paragraph("4. Recommandation clinique", s_h2))
        story.append(Paragraph(report.recommendation, s_b))
        story.extend([Spacer(1, 4*mm)])

        # Footer
        story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#D5E8F0")))
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(
            "⚠ Outil d'aide à la décision — validation obligatoire par un médecin spécialiste. "
            "KANÉA 2026 — Knowledge Anthropology & Neural Engine for Africa.",
            s_m,
        ))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return None
    except Exception:
        return None
