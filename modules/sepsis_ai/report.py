"""
SepsisPredict AI — Rapport médical PDF A4 multi-pages
======================================================
Contient : données patient, scores cliniques, risques multi-horizons,
biomarqueurs critiques, alertes, recommandations SSC, explainability SHAP.
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime
from typing import Any

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    _RL_OK = True
except ImportError:
    _RL_OK = False

try:
    import qrcode
    _QR_OK = True
except ImportError:
    _QR_OK = False

# Palette
_TEAL    = colors.HexColor("#20B2AA") if _RL_OK else None
_RED     = colors.HexColor("#C0392B") if _RL_OK else None
_ORANGE  = colors.HexColor("#E67E22") if _RL_OK else None
_GREEN   = colors.HexColor("#27AE60") if _RL_OK else None
_BLUE    = colors.HexColor("#2E86DE") if _RL_OK else None
_DARK    = colors.HexColor("#1A2B3C") if _RL_OK else None
_MUTED   = colors.HexColor("#5E7A8A") if _RL_OK else None
_LIGHT   = colors.HexColor("#EEF2F6") if _RL_OK else None
_WHITE   = colors.white               if _RL_OK else None
_CRIT    = colors.HexColor("#922B21") if _RL_OK else None

_URGENCY_HEX = {
    "EXTRÊME": "922B21", "Critique": "C0392B",
    "Élevée": "E74C3C", "Modérée": "E67E22",
    "Faible": "27AE60",
}
_URGENCY_RL = {
    "EXTRÊME": colors.HexColor("#922B21") if _RL_OK else None,
    "Critique": colors.HexColor("#C0392B") if _RL_OK else None,
    "Élevée":  colors.HexColor("#E74C3C") if _RL_OK else None,
    "Modérée": colors.HexColor("#E67E22") if _RL_OK else None,
    "Faible":  colors.HexColor("#27AE60") if _RL_OK else None,
}


def _style(name: str, **kw) -> Any:
    return ParagraphStyle(name, **kw)


def _qr_img(text: str, size: int = 60) -> Any | None:
    if not _QR_OK or not _RL_OK:
        return None
    try:
        from reportlab.platypus import Image as RLImage
        qr = qrcode.QRCode(box_size=2, border=2)
        qr.add_data(text)
        qr.make(fit=True)
        img = qr.make_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return RLImage(buf, width=size, height=size)
    except Exception:
        return None


def build_sepsis_pdf_report(
    result: dict[str, Any],
    alerts: list[dict[str, Any]] | None = None,
    patient_id: str = "KANEA-AUTO",
    examiner: str = "KANÉA System",
    institution: str = "Réanimation — KANÉA",
) -> bytes | None:
    """Génère le rapport PDF SepsisPredict AI."""
    if not _RL_OK:
        return None

    alerts = alerts or []
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=14*mm, bottomMargin=14*mm,
                            leftMargin=17*mm, rightMargin=17*mm)
    W = A4[0] - 34*mm

    s_h1  = _style("H1", fontSize=14, fontName="Helvetica-Bold", textColor=_WHITE, leading=17)
    s_h2  = _style("H2", fontSize=10.5, fontName="Helvetica-Bold", textColor=_DARK, leading=13, spaceAfter=3)
    s_h3  = _style("H3", fontSize=9, fontName="Helvetica-Bold", textColor=_TEAL, leading=11)
    s_b   = _style("B", fontSize=8.5, fontName="Helvetica", textColor=_DARK, leading=11)
    s_m   = _style("M", fontSize=7.5, fontName="Helvetica", textColor=_MUTED, leading=10)
    s_r   = _style("R", fontSize=7.5, fontName="Helvetica", textColor=_MUTED, alignment=TA_RIGHT, leading=10)

    story: list[Any] = []

    # ── Données résultat ─────────────────────────────────────────────────────
    pred       = result.get("prediction", "—")
    conf       = result.get("confidence", 0)
    urgency    = result.get("clinical_profile", {}).get("urgency", "—")
    color_rl   = _URGENCY_RL.get(urgency, _MUTED)
    urg_hex    = _URGENCY_HEX.get(urgency, "5E7A8A")
    action     = result.get("recommended_action", "—")
    risk_score = result.get("risk_score", 0)
    request_id = result.get("request_id", str(uuid.uuid4()))[:16]
    date_str   = datetime.now().strftime("%d/%m/%Y à %H:%M")
    model_ver  = result.get("model_version", "v2.0")

    # ── HEADER ───────────────────────────────────────────────────────────────
    qr = _qr_img(f"KANEA-SEPSIS-{request_id}", 55)
    header_left = Paragraph(
        f'<font size="15"><b>KANÉA — SepsisPredict AI {model_ver}</b></font><br/>'
        f'<font size="8.5">Rapport de surveillance sepsis · Soins intensifs · Réanimation</font>',
        s_h1,
    )
    header_right_content = [
        Paragraph(f'<font size="7.5">N° {request_id}<br/>{date_str}<br/>Dossier : {patient_id}</font>',
                  _style("HR", fontSize=7.5, fontName="Helvetica", textColor=_WHITE, alignment=TA_RIGHT, leading=10)),
    ]
    if qr:
        header_right_content.append(qr)

    header_data = [[header_left, header_right_content[-1] if not qr else qr]]
    header_t = Table(header_data, colWidths=[W * 0.72, W * 0.28])
    header_t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), _DARK),
        ("TOPPADDING",   (0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(header_t)
    story.append(Spacer(1, 4*mm))

    # ── 1. RENSEIGNEMENTS ────────────────────────────────────────────────────
    story.append(Paragraph("1. Renseignements du dossier", s_h2))
    info_data = [
        ["N° dossier", patient_id, "Date", date_str],
        ["Examinateur", examiner, "Institution", institution],
        ["Modèle IA", "XGBoost + LightGBM + LSTM", "Version", model_ver],
        ["Temps analyse", f"{result.get('processing_ms',0)} ms", "ID requête", request_id],
    ]
    info_t = Table(info_data, colWidths=[W*0.16, W*0.34, W*0.14, W*0.36])
    info_t.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0),(-1,-1), 8),
        ("FONTNAME",    (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTNAME",    (2,0),(2,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,0),(0,-1),  _MUTED),
        ("TEXTCOLOR",   (2,0),(2,-1),  _MUTED),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[_LIGHT, _WHITE]),
        ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#DDE8EE")),
        ("TOPPADDING",  (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0),(-1,-1), 6),
    ]))
    story.append(info_t)
    story.append(Spacer(1, 4*mm))

    # ── 2. PRÉDICTION PRINCIPALE ─────────────────────────────────────────────
    story.append(Paragraph("2. Résultat de l'analyse IA", s_h2))
    pred_data = [
        [Paragraph("<b>Diagnostic IA</b>", s_m),
         Paragraph(f"<b><font size='12' color='#{urg_hex}'>{pred}</font></b>", s_b),
         Paragraph("<b>Confiance</b>", s_m),
         Paragraph(f"<b><font size='11' color='#{urg_hex}'>{conf:.1%}</font></b>", s_b)],
        [Paragraph("<b>Urgence</b>", s_m),
         Paragraph(f"<b><font color='#{urg_hex}'>{urgency}</font></b>", s_b),
         Paragraph("<b>Score de risque</b>", s_m),
         Paragraph(f"<b>{risk_score:.1%}</b>", s_b)],
        [Paragraph("<b>Action</b>", s_m),
         Paragraph(f"<font size='8'>{action[:80]}</font>", s_b),
         Paragraph("<b>Référence</b>", s_m),
         Paragraph("<font size='7.5'>SSC 2021 · Sepsis-3</font>", s_m)],
    ]
    pred_t = Table(pred_data, colWidths=[W*0.14, W*0.36, W*0.14, W*0.36])
    pred_t.setStyle(TableStyle([
        ("FONTSIZE",    (0,0),(-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#F7F9FC"), _WHITE]),
        ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#DDE8EE")),
        ("TOPPADDING",  (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0),(-1,-1), 7),
        ("LINEBEFORE",  (0,0),(0,-1), 4, color_rl),
    ]))
    story.append(pred_t)
    story.append(Spacer(1, 4*mm))

    # ── 3. SCORES CLINIQUES ──────────────────────────────────────────────────
    story.append(Paragraph("3. Scores cliniques de réanimation", s_h2))
    scores  = result.get("clinical_scores", {})
    sc_rows = [["Score", "Résultat", "Sévérité / Interprétation", "Action"]]

    sofa   = scores.get("sofa", {})
    qsofa  = scores.get("qsofa", {})
    news2  = scores.get("news2", {})
    apache = scores.get("apache_ii", {})
    mews   = scores.get("mews", {})

    if sofa and "total" in sofa:
        sc_rows.append(["SOFA", f"{sofa['total']}/24",
                         f"{sofa.get('severity','—')} — Mortalité {sofa.get('mortality','—')}",
                         f"Organes: {', '.join(sofa.get('organ_failures',['—']) or ['Aucun'])}"])
    if qsofa and "score" in qsofa:
        sc_rows.append(["qSOFA", f"{qsofa['score']}/3",
                         "SEPSIS PROBABLE" if qsofa.get("high_risk") else "Faible risque",
                         qsofa.get("recommendation","—")[:60]])
    if news2 and "total" in news2:
        sc_rows.append(["NEWS2", f"{news2['total']}/20",
                         news2.get("clinical_risk","—"),
                         news2.get("escalation","—")[:60]])
    if apache and "score" in apache:
        sc_rows.append(["APACHE II", f"{apache['score']}/71",
                         f"{apache.get('severity','—')} — Mortalité {apache.get('mortality','—')}",
                         "Transfert USI si > 25"])
    if mews and "score" in mews:
        sc_rows.append(["MEWS", f"{mews['score']}/14",
                         mews.get("risk_level","—"),
                         mews.get("action","—")[:60]])

    sc_t = Table(sc_rows, colWidths=[W*0.12, W*0.12, W*0.40, W*0.36])
    sc_t.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 7.5),
        ("BACKGROUND",  (0,0),(-1,0), _BLUE),
        ("TEXTCOLOR",   (0,0),(-1,0), _WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LIGHT,_WHITE]),
        ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#DDE8EE")),
        ("TOPPADDING",  (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0),(-1,-1), 5),
        ("WORDWRAP",    (0,0),(-1,-1), True),
    ]))
    story.append(sc_t)
    story.append(Spacer(1, 4*mm))

    # ── 4. RISQUES MULTI-HORIZONS ────────────────────────────────────────────
    story.append(Paragraph("4. Risques prédits", s_h2))
    mort_risk = result.get("mortality_risk", {})
    shock_risk = result.get("septic_shock_risk", "—")
    organ_risk = result.get("organ_failure_risk", {})

    risk_rows = [["Horizon", "Mortalité estimée", "Choc septique", "DMV"]]
    for horizon, mort in mort_risk.items():
        risk_rows.append([
            horizon,
            f"{mort:.1%}",
            shock_risk,
            "Oui" if len([v for v in organ_risk.values() if v == "Élevé"]) >= 2 else "Non",
        ])
    risk_t = Table(risk_rows, colWidths=[W*0.12, W*0.22, W*0.22, W*0.44])
    risk_t.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 8),
        ("BACKGROUND",  (0,0),(-1,0), _RED),
        ("TEXTCOLOR",   (0,0),(-1,0), _WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LIGHT,_WHITE]),
        ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#DDE8EE")),
        ("TOPPADDING",  (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0),(-1,-1), 6),
    ]))
    story.append(risk_t)
    story.append(Spacer(1, 4*mm))

    # ── 5. BIOMARQUEURS CRITIQUES ────────────────────────────────────────────
    critical_bio = result.get("critical_biomarkers", [])
    if critical_bio:
        story.append(Paragraph("5. Biomarqueurs hors normes critiques", s_h2))
        bio_rows = [["Biomarqueur", "Valeur", "Statut"]]
        for b in critical_bio:
            bio_rows.append([
                b.get("biomarker","—"),
                f"{b.get('value','—')} {b.get('unit','')}",
                b.get("status","—"),
            ])
        bio_t = Table(bio_rows, colWidths=[W*0.30, W*0.25, W*0.45])
        bio_t.setStyle(TableStyle([
            ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0),(-1,-1), 8),
            ("BACKGROUND",  (0,0),(-1,0), colors.HexColor("#8E44AD")),
            ("TEXTCOLOR",   (0,0),(-1,0), _WHITE),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LIGHT,_WHITE]),
            ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#DDE8EE")),
            ("TOPPADDING",  (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ("LEFTPADDING", (0,0),(-1,-1), 6),
        ]))
        story.append(bio_t)
        story.append(Spacer(1, 4*mm))

    # ── 6. ALERTES ───────────────────────────────────────────────────────────
    if alerts:
        story.append(Paragraph("6. Alertes médicales", s_h2))
        alert_level_colors = {
            "CRITICAL": colors.HexColor("#FDECEA"),
            "HIGH":     colors.HexColor("#FEF0E7"),
            "WARNING":  colors.HexColor("#FEF9E7"),
            "INFO":     colors.HexColor("#EBF4FD"),
        }
        alert_border = {
            "CRITICAL": colors.HexColor("#922B21"),
            "HIGH":     colors.HexColor("#E74C3C"),
            "WARNING":  colors.HexColor("#E67E22"),
            "INFO":     colors.HexColor("#2E86DE"),
        }
        for a in alerts[:8]:  # max 8 alertes dans le PDF
            lvl = a.get("level", "INFO")
            alert_data = [[
                Paragraph(
                    f"<b>{a.get('icon','')} [{lvl}] {a.get('title','—')}</b><br/>"
                    f"<font size='7.5'>{a.get('message','—')}</font><br/>"
                    f"<font size='7'><i>Action : {a.get('clinical_action','—')[:80]}</i></font>",
                    _style(f"AL_{lvl}", fontSize=8, fontName="Helvetica", textColor=_DARK, leading=11),
                )
            ]]
            alert_t = Table(alert_data, colWidths=[W])
            alert_t.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), alert_level_colors.get(lvl, _LIGHT)),
                ("LINEBEFORE",   (0,0),(-1,-1), 4, alert_border.get(lvl, _MUTED)),
                ("TOPPADDING",   (0,0),(-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                ("LEFTPADDING",  (0,0),(-1,-1), 8),
            ]))
            story.append(alert_t)
            story.append(Spacer(1, 2*mm))
        story.append(Spacer(1, 2*mm))

    # ── 7. EXPLAINABILITY ───────────────────────────────────────────────────
    expl = result.get("explainability", {})
    top3 = expl.get("top_3_drivers", [])
    fi   = expl.get("feature_importance", {})
    if top3 or fi:
        story.append(Paragraph("7. Explainability IA — Variables déterminantes", s_h2))
        if top3:
            top3_str = " · ".join([f"<b>{k}</b> ({v:.0f}%)" for k, v in top3])
            story.append(Paragraph(f"Top 3 facteurs déclencheurs : {top3_str}", s_b))
            story.append(Spacer(1, 2*mm))
        if fi:
            fi_rows = [["Variable", "Importance relative", "Contribution"]]
            for feat, imp in sorted(fi.items(), key=lambda x: -x[1])[:8]:
                bar_w = int(imp / 100 * 50)
                fi_rows.append([feat, f"{imp:.1f}%", "█" * bar_w + "░" * (50 - bar_w)])
            fi_t = Table(fi_rows, colWidths=[W*0.30, W*0.15, W*0.55])
            fi_t.setStyle(TableStyle([
                ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
                ("FONTSIZE",    (0,0),(-1,-1), 7.5),
                ("FONTNAME",    (0,1),(2,-1),  "Courier"),
                ("BACKGROUND",  (0,0),(-1,0), _TEAL),
                ("TEXTCOLOR",   (0,0),(-1,0), _WHITE),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LIGHT,_WHITE]),
                ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#DDE8EE")),
                ("TOPPADDING",  (0,0),(-1,-1), 3),
                ("BOTTOMPADDING",(0,0),(-1,-1), 3),
                ("LEFTPADDING", (0,0),(-1,-1), 5),
            ]))
            story.append(fi_t)
        story.append(Spacer(1, 4*mm))

    # ── 8. SSC BUNDLE ────────────────────────────────────────────────────────
    ssc = scores.get("ssc_bundle", {})
    if ssc and "bundle_items" in ssc:
        story.append(Paragraph("8. Surviving Sepsis Campaign — Bundle 1h", s_h2))
        bundle_rows = [["Item SSC Bundle", "Statut"]]
        for item, done in ssc["bundle_items"].items():
            bundle_rows.append([item, "✔ Réalisé" if done else "✘ Non réalisé"])
        bundle_t = Table(bundle_rows, colWidths=[W*0.70, W*0.30])
        bundle_t.setStyle(TableStyle([
            ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0),(-1,-1), 8),
            ("BACKGROUND",  (0,0),(-1,0), colors.HexColor("#C0392B")),
            ("TEXTCOLOR",   (0,0),(-1,0), _WHITE),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LIGHT,_WHITE]),
            ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#DDE8EE")),
            ("TOPPADDING",  (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ("LEFTPADDING", (0,0),(-1,-1), 6),
        ]))
        story.append(bundle_t)
        comp_pct = ssc.get("compliance_pct", 0)
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(
            f"Compliance Bundle 1h : <b>{comp_pct:.0f}%</b> ({ssc.get('completed',0)}/{ssc.get('total',5)} items réalisés)",
            s_b,
        ))
        story.append(Spacer(1, 3*mm))

    # ── FOOTER ───────────────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#DDE8EE")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "⚠ Outil d'aide à la décision médicale — ne remplace pas le jugement clinique. "
        "Toute décision thérapeutique doit être validée par un médecin réanimateur. "
        "SepsisPredict AI v2.0 — KANÉA 2026 — XGBoost + LightGBM + LSTM — MIMIC-IV + PhysioNet Sepsis 2019.",
        s_m,
    ))

    doc.build(story)
    return buf.getvalue()


def build_sepsis_html_report(
    result: dict[str, Any],
    alerts: list[dict[str, Any]] | None = None,
    patient_id: str = "KANEA",
) -> str:
    """Rapport HTML léger (fallback)."""
    alerts = alerts or []
    pred     = result.get("prediction", "—")
    conf     = result.get("confidence", 0)
    urgency  = result.get("clinical_profile", {}).get("urgency", "—")
    action   = result.get("recommended_action", "—")
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    urg_hex  = "#" + _URGENCY_HEX.get(urgency, "5E7A8A")

    mort = result.get("mortality_risk", {})
    mort_rows = "".join(f"<tr><td>{h}</td><td>{v:.1%}</td></tr>" for h, v in mort.items())
    alert_rows = "".join(
        f'<div style="border-left:4px solid {a.get("color","#999")};padding:5px 10px;'
        f'margin:4px 0;background:#f9f9f9;font-size:10px;">'
        f'<b>{a.get("icon","")} [{a.get("level","")}] {a.get("title","")}</b><br/>'
        f'{a.get("message","")}</div>'
        for a in alerts[:6]
    )

    return f"""<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8">
<title>SepsisPredict AI — {patient_id}</title>
<style>body{{font-family:Arial,sans-serif;font-size:11px;color:#1A2B3C;margin:0;padding:12px}}
h1{{background:#1A2B3C;color:#fff;padding:8px 12px;font-size:13px}}
h2{{font-size:10px;color:#20B2AA;text-transform:uppercase;margin:10px 0 4px}}
table{{width:100%;border-collapse:collapse;margin-bottom:6px}}
th{{background:#2E86DE;color:#fff;padding:3px 6px;font-size:9px;text-align:left}}
td{{padding:3px 6px;border:1px solid #DDE8EE;font-size:9px}}
tr:nth-child(even){{background:#F7F9FC}}
.disc{{background:#FFF8E1;border-left:4px solid #F39C12;padding:6px 10px;font-size:8px}}</style>
</head><body>
<h1>KANÉA — SepsisPredict AI v2.0 · Rapport sepsis</h1>
<p style="background:#F0F4F8;padding:5px 10px;margin:0;font-size:9px">Dossier : {patient_id} · {date_str}</p>
<h2>Résultat IA</h2>
<p><b style="color:{urg_hex};font-size:13px">{pred}</b>
&nbsp;&nbsp; Confiance : <b>{conf:.1%}</b> &nbsp;&nbsp; Urgence : <b style="color:{urg_hex}">{urgency}</b></p>
<p><b>Action :</b> {action}</p>
<h2>Risque de mortalité</h2>
<table><tr><th>Horizon</th><th>Probabilité</th></tr>{mort_rows}</table>
<h2>Alertes</h2>{alert_rows}
<div class="disc">⚠ Outil d'aide à la décision médicale — validation médicale obligatoire. KANÉA 2026.</div>
</body></html>"""
