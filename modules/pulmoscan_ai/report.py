"""
PulmoScan AI — Rapport médical PDF A4
======================================
Génère un rapport clinique complet : diagnostic IA, scores, heatmaps,
recommandations, références OMS/guidelines internationales.
"""
from __future__ import annotations

import base64
import io
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# ReportLab
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    _REPORTLAB_OK = True
except ImportError:
    _REPORTLAB_OK = False

# QR Code
try:
    import qrcode
    _QRCODE_OK = True
except ImportError:
    _QRCODE_OK = False


# ── Couleurs KANEA / PulmoScan ────────────────────────────────────────────────
_TEAL    = colors.HexColor("#20B2AA")
_BLUE    = colors.HexColor("#2980B9")
_RED     = colors.HexColor("#C0392B")
_ORANGE  = colors.HexColor("#E67E22")
_GREEN   = colors.HexColor("#27AE60")
_PURPLE  = colors.HexColor("#8E44AD")
_DARK    = colors.HexColor("#1A2B3C")
_MUTED   = colors.HexColor("#5E7A8A")
_LIGHT   = colors.HexColor("#EEF2F6")
_WHITE   = colors.white
_BLACK   = colors.black

_URGENCY_COLORS = {
    "Critique":       colors.HexColor("#C0392B"),
    "Urgence vitale": colors.HexColor("#922B21"),
    "Élevée":         colors.HexColor("#E74C3C"),
    "Modérée":        colors.HexColor("#E67E22"),
    "Faible":         colors.HexColor("#27AE60"),
    "Normal":         colors.HexColor("#27AE60"),
}


def _urgency_color(urgency: str) -> Any:
    return _URGENCY_COLORS.get(urgency, _MUTED)


def _qr_image(text: str, size: int = 80) -> Any | None:
    if not _QRCODE_OK:
        return None
    try:
        qr = qrcode.QRCode(box_size=2, border=2)
        qr.add_data(text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return RLImage(buf, width=size, height=size)
    except Exception:
        return None


def _b64_to_rl_image(b64: str, width: float = 200, height: float = 200) -> Any | None:
    try:
        data = base64.b64decode(b64)
        buf = io.BytesIO(data)
        return RLImage(buf, width=width, height=height)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Générateur principal
# ═══════════════════════════════════════════════════════════════════════════════

def build_pulmoscan_pdf_report(
    result: dict[str, Any],
    patient_id: str = "KANEA-AUTO",
    examiner: str = "KANÉA System",
    institution: str = "Institut de Pneumologie — KANÉA",
) -> bytes | None:
    """
    Génère le rapport PDF PulmoScan AI complet.

    Args:
        result: dict retourné par predict_pulmoscan()
        patient_id: identifiant du patient / dossier
        examiner: nom du médecin examinateur
        institution: nom de l'institution

    Returns:
        bytes du PDF ou None si ReportLab non disponible
    """
    if not _REPORTLAB_OK:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    W = A4[0] - 36 * mm  # largeur utile

    # ── Styles personnalisés ─────────────────────────────────────────────────
    def _style(name, **kw) -> ParagraphStyle:
        return ParagraphStyle(name, **kw)

    s_h1 = _style("H1", fontSize=15, fontName="Helvetica-Bold", textColor=_WHITE,
                  alignment=TA_LEFT, leading=18)
    s_h2 = _style("H2", fontSize=11, fontName="Helvetica-Bold", textColor=_DARK,
                  alignment=TA_LEFT, leading=14, spaceAfter=3)
    s_h3 = _style("H3", fontSize=9.5, fontName="Helvetica-Bold", textColor=_TEAL,
                  alignment=TA_LEFT, leading=12)
    s_body = _style("Body", fontSize=9, fontName="Helvetica", textColor=_DARK,
                    alignment=TA_LEFT, leading=12)
    s_muted = _style("Muted", fontSize=8, fontName="Helvetica", textColor=_MUTED,
                     alignment=TA_LEFT, leading=10)
    s_center = _style("Center", fontSize=9, fontName="Helvetica", textColor=_DARK,
                      alignment=TA_CENTER, leading=12)
    s_right = _style("Right", fontSize=8, fontName="Helvetica", textColor=_MUTED,
                     alignment=TA_RIGHT, leading=10)

    story: list[Any] = []

    # ── 1. HEADER ────────────────────────────────────────────────────────────
    prediction   = result.get("prediction", "—")
    confidence   = result.get("confidence", 0)
    urgency      = result.get("clinical_profile", {}).get("urgency", "—")
    icd10        = result.get("clinical_profile", {}).get("icd10", "—")
    severity     = result.get("severity", "—")
    action       = result.get("recommended_action", "—")
    who_ref      = result.get("reference_guidelines", "—")
    pattern      = result.get("clinical_profile", {}).get("pattern", "—")
    request_id   = result.get("request_id", str(uuid.uuid4()))[:16]
    model_ver    = result.get("model_version", "v2.0")
    backend      = result.get("inference_backend", "—")
    proc_ms      = result.get("processing_ms", 0)
    date_str     = datetime.now().strftime("%d/%m/%Y à %H:%M")
    urg_color = _urgency_color(urgency)
    # Hex string 6 chars pour inline HTML (ReportLab hexval() renvoie '0xRRGGBB')
    _URGENCY_HEX = {
        "Critique": "C0392B", "Urgence vitale": "922B21",
        "Élevée": "E74C3C", "Modérée": "E67E22",
        "Faible": "27AE60", "Normal": "27AE60",
    }
    urg_hex = _URGENCY_HEX.get(urgency, "5E7A8A")

    # Bandeau header
    header_data = [[
        Paragraph(
            f'<font size="16"><b>KANÉA — PulmoScan AI {model_ver}</b></font><br/>'
            f'<font size="9">Rapport d\'analyse pneumologique — Radiographie / TDM thoracique</font>',
            s_h1,
        ),
        Paragraph(
            f'<font size="8">N° {request_id}<br/>'
            f'{date_str}<br/>'
            f'Backend : {backend}</font>',
            _style("HR", fontSize=8, fontName="Helvetica", textColor=_WHITE,
                   alignment=TA_RIGHT, leading=11),
        ),
    ]]
    header_table = Table(header_data, colWidths=[W * 0.68, W * 0.32])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), _DARK),
        ("TOPPADDING",  (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",(0, 0), (-1, -1), 10),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 5 * mm))

    # ── 2. INFORMATIONS PATIENT & DOSSIER ────────────────────────────────────
    story.append(Paragraph("1. Renseignements du dossier", s_h2))
    info_data = [
        ["N° de dossier", patient_id, "Date d'analyse", date_str],
        ["Examinateur", examiner, "Institution", institution],
        ["Modèle IA", f"DenseNet121 — NIH ChestXray14 + CheXpert", "Version", model_ver],
        ["Temps d'analyse", f"{proc_ms} ms", "Identifiant requête", request_id],
    ]
    info_table = Table(info_data, colWidths=[W * 0.18, W * 0.32, W * 0.18, W * 0.32])
    info_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0, 0), (0, -1), _MUTED),
        ("TEXTCOLOR",   (2, 0), (2, -1), _MUTED),
        ("BACKGROUND",  (0, 0), (-1, -1), _LIGHT),
        ("BACKGROUND",  (0, 1), (-1, 1), _WHITE),
        ("BACKGROUND",  (0, 3), (-1, 3), _WHITE),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#DDE8EE")),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 5 * mm))

    # ── 3. RÉSULTAT PRINCIPAL ────────────────────────────────────────────────
    story.append(Paragraph("2. Résultat de l'analyse IA", s_h2))

    result_data = [
        [
            Paragraph(f"<b>Diagnostic IA</b>", s_muted),
            Paragraph(f"<b><font size='13' color='#{urg_hex}'>{prediction}</font></b>", s_body),
            Paragraph("<b>Confiance</b>", s_muted),
            Paragraph(f"<b><font size='12' color='#{urg_hex}'>{confidence:.1%}</font></b>", s_body),
        ],
        [
            Paragraph("<b>Sévérité</b>", s_muted),
            Paragraph(f"<b>{severity}</b>", s_body),
            Paragraph("<b>Urgence</b>", s_muted),
            Paragraph(f"<b><font color='#{urg_hex}'>{urgency}</font></b>", s_body),
        ],
        [
            Paragraph("<b>CIM-10</b>", s_muted),
            Paragraph(icd10, s_body),
            Paragraph("<b>Pattern radiologique</b>", s_muted),
            Paragraph(pattern[:60], s_body),
        ],
    ]
    result_table = Table(result_data, colWidths=[W * 0.15, W * 0.35, W * 0.15, W * 0.35])
    result_table.setStyle(TableStyle([
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor("#F7F9FC")),
        ("BACKGROUND",  (0, 1), (-1, 1), _WHITE),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#DDE8EE")),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBEFORE",  (0, 0), (0, -1), 3, urg_color),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 4 * mm))

    # Diagnostic différentiel
    diff = result.get("differential_diagnosis", [])
    if diff:
        story.append(Paragraph("Diagnostic différentiel (Top 3)", s_h3))
        diff_data = [["Classe", "Probabilité", "CIM-10"]] + [
            [d["class"], f"{d['probability']:.1%}", d.get("icd10", "—")]
            for d in diff
        ]
        diff_table = Table(diff_data, colWidths=[W * 0.55, W * 0.22, W * 0.23])
        diff_table.setStyle(TableStyle([
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
            ("BACKGROUND",  (0, 0), (-1, 0), _TEAL),
            ("TEXTCOLOR",   (0, 0), (-1, 0), _WHITE),
            ("BACKGROUND",  (0, 1), (-1, 1), colors.HexColor("#EAF7F5")),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#DDE8EE")),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(diff_table)
        story.append(Spacer(1, 4 * mm))

    # ── 4. SCORES CLINIQUES ──────────────────────────────────────────────────
    scores = result.get("clinical_scores", {})
    if len(scores) > 2:
        story.append(Paragraph("3. Scores cliniques", s_h2))
        score_rows = [["Score / Système", "Résultat", "Sévérité / Classe", "Recommandation"]]

        if "curb65" in scores:
            c = scores["curb65"]
            score_rows.append(["CURB-65", str(c.get("score", "—")),
                                c.get("severity", "—"), c.get("recommendation", "—")[:50]])
        if "psi" in scores:
            p = scores["psi"]
            score_rows.append(["PSI (Fine Score)", f"Classe {p.get('class', '—')}",
                                f"Mortalité {p.get('mortality', '—')}", p.get("recommendation", "—")[:50]])
        if "covid_ct_severity" in scores:
            cv = scores["covid_ct_severity"]
            score_rows.append(["COVID CT Severity",
                                f"{cv.get('total_score', '—')}/25 ({cv.get('involvement', '—')})",
                                cv.get("severity", "—"), cv.get("recommendation", "—")[:50]])
        if "lung_rads" in scores:
            lr = scores["lung_rads"]
            score_rows.append(["Lung-RADS", f"Cat. {lr.get('category', '—')}",
                                lr.get("probability", "—"), lr.get("management", "—")[:50]])
        if "tnm" in scores:
            tnm = scores["tnm"]
            score_rows.append(["TNM (IASLC 9e éd.)",
                                f"T{tnm.get('T','?')} N{tnm.get('N','?')} M{tnm.get('M','?')}",
                                f"Stade {tnm.get('stage', '—')} — {tnm.get('survival', '—')}",
                                tnm.get("treatment", "—")[:50]])
        if "gold" in scores:
            g = scores["gold"]
            score_rows.append(["GOLD (BPCO)", f"Groupe {g.get('group', '—')}",
                                g.get("grade", "—"), g.get("treatment", "—")[:50]])
        if "tuberculosis" in scores:
            tb = scores["tuberculosis"]
            score_rows.append(["TB OMS 2022", tb.get("form", "—"),
                                tb.get("severity", "—"), tb.get("treatment", "—")[:50]])

        if len(score_rows) > 1:
            sc_table = Table(score_rows, colWidths=[W * 0.20, W * 0.20, W * 0.25, W * 0.35])
            sc_table.setStyle(TableStyle([
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1), 8),
                ("BACKGROUND",  (0, 0), (-1, 0), _BLUE),
                ("TEXTCOLOR",   (0, 0), (-1, 0), _WHITE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT, _WHITE]),
                ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#DDE8EE")),
                ("TOPPADDING",  (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("WORDWRAP",    (0, 0), (-1, -1), True),
            ]))
            story.append(sc_table)
            story.append(Spacer(1, 4 * mm))

    # ── 5. EXPLAINABILITY / HEATMAP ─────────────────────────────────────────
    expl = result.get("explainability", {})
    heatmap_b64 = expl.get("heatmap_b64") or expl.get("grad_cam")
    if heatmap_b64:
        story.append(Paragraph("4. Explainability — Grad-CAM", s_h2))
        hm_img = _b64_to_rl_image(heatmap_b64, width=180, height=180)
        if hm_img:
            hm_table = Table(
                [[hm_img,
                  Paragraph(
                      f"<b>Méthode :</b> Gradient-weighted Class Activation Mapping<br/>"
                      f"<b>Prédiction :</b> {prediction}<br/>"
                      f"<b>Lobes analysés :</b> {', '.join(expl.get('lobes_activated', ['—']))}<br/><br/>"
                      f"<i>La carte thermique rouge indique les régions pulmonaires "
                      f"ayant le plus contribué à la classification.<br/>"
                      f"Les zones jaune-rouge correspondent aux anomalies détectées.</i>",
                      s_body,
                  )]],
                colWidths=[185, W - 185],
            )
            hm_table.setStyle(TableStyle([
                ("VALIGN",  (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (1, 0), (1, 0), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(hm_table)
            story.append(Spacer(1, 4 * mm))

    # ── 6. QUANTIFICATION ────────────────────────────────────────────────────
    quant = result.get("quantification", {})
    if quant:
        story.append(Paragraph("5. Quantification des lésions", s_h2))
        q_data = [
            ["Paramètre", "Valeur", "Interprétation"],
            ["Surface atteinte (estimation)", f"{quant.get('lesion_coverage_pct', 0):.1f}%",
             "< 25% léger · 25–50% modéré · > 50% sévère"],
            ["Intensité moyenne image", f"{quant.get('image_mean_intensity', 0):.3f}",
             "0=noir, 1=blanc — évalue la densité radiologique"],
            ["Contraste image", f"{quant.get('image_contrast', 0):.3f}",
             "Hétérogénéité des densités (élevé = lésions multiples)"],
        ]
        q_table = Table(q_data, colWidths=[W * 0.30, W * 0.20, W * 0.50])
        q_table.setStyle(TableStyle([
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
            ("BACKGROUND",  (0, 0), (-1, 0), _PURPLE),
            ("TEXTCOLOR",   (0, 0), (-1, 0), _WHITE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT, _WHITE]),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#DDE8EE")),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(q_table)
        story.append(Spacer(1, 4 * mm))

    # ── 7. RECOMMANDATIONS ────────────────────────────────────────────────────
    story.append(Paragraph("6. Recommandations cliniques", s_h2))
    rec_data = [
        [Paragraph(f"<b><font color='#{urg_hex}'>Action recommandée :</font></b> {action}", s_body)],
        [Paragraph(f"<b>Référence :</b> {who_ref}", s_muted)],
    ]
    rec_table = Table(rec_data, colWidths=[W])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor("#F7F9FC")),
        ("LINEBEFORE",  (0, 0), (-1, -1), 4, urg_color),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#DDE8EE")),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 3 * mm))

    # Alerte clinique
    safety = result.get("clinical_safety", {})
    if safety.get("level") in ("warning", "critical"):
        alert_color = _RED if safety["level"] == "critical" else _ORANGE
        alert_table = Table(
            [[Paragraph(f"⚠ {safety.get('message', '')}", s_body)]],
            colWidths=[W],
        )
        alert_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor("#FDECEA")),
            ("LINEBEFORE",  (0, 0), (-1, -1), 5, alert_color),
            ("TOPPADDING",  (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(alert_table)
        story.append(Spacer(1, 3 * mm))

    # ── 8. DISCLAIMER + FOOTER ────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#DDE8EE")))
    story.append(Spacer(1, 2 * mm))

    # QR Code
    qr = _qr_image(f"KANEA-PULMOSCAN-{request_id}", size=55)
    disclaimer_text = (
        "<b>⚠ Outil d'aide à la décision médicale — ne remplace pas le diagnostic clinique.</b> "
        "Ce rapport est généré automatiquement par PulmoScan AI v2.0 et doit être interprété "
        "sous supervision d'un médecin ou radiologue qualifié. Toute décision thérapeutique "
        "doit être validée par un professionnel de santé conformément aux protocoles en vigueur. "
        "DenseNet121 — NIH ChestXray14 + CheXpert. MIT Licence — KANÉA 2026."
    )
    if qr:
        footer_data = [[
            Paragraph(disclaimer_text, s_muted),
            qr,
        ]]
        footer_table = Table(footer_data, colWidths=[W - 65, 65])
    else:
        footer_data = [[Paragraph(disclaimer_text, s_muted)]]
        footer_table = Table(footer_data, colWidths=[W])

    footer_table.setStyle(TableStyle([
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
    ]))
    story.append(footer_table)

    doc.build(story)
    return buf.getvalue()


def build_pulmoscan_html_report(result: dict[str, Any], patient_id: str = "KANEA") -> str:
    """Rapport HTML léger (fallback si ReportLab indisponible)."""
    prediction  = result.get("prediction", "—")
    confidence  = result.get("confidence", 0)
    urgency     = result.get("clinical_profile", {}).get("urgency", "—")
    action      = result.get("recommended_action", "—")
    severity    = result.get("severity", "—")
    date_str    = datetime.now().strftime("%d/%m/%Y %H:%M")
    request_id  = result.get("request_id", "—")[:16]

    color_map = {"Critique": "#C0392B", "Élevée": "#E74C3C",
                 "Modérée": "#E67E22", "Faible": "#27AE60", "Normal": "#27AE60"}
    urg_color = color_map.get(urgency, "#5E7A8A")

    diff = result.get("differential_diagnosis", [])
    diff_rows = "".join(
        f"<tr><td>{d['class']}</td><td>{d['probability']:.1%}</td><td>{d.get('icd10','—')}</td></tr>"
        for d in diff
    )

    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>PulmoScan AI — {patient_id}</title>
<style>
body{{font-family:Arial,sans-serif;font-size:11px;color:#1A2B3C;margin:0;padding:16px;background:#fff}}
h1{{background:#1A2B3C;color:#fff;padding:10px 14px;margin:0;font-size:14px}}
h2{{font-size:11px;color:#20B2AA;margin:12px 0 4px;text-transform:uppercase;letter-spacing:.5px}}
.badge{{display:inline-block;padding:3px 10px;border-radius:4px;font-weight:700;font-size:12px}}
table{{width:100%;border-collapse:collapse;margin-bottom:8px}}
th{{background:#20B2AA;color:#fff;padding:4px 7px;font-size:9px;text-align:left}}
td{{padding:4px 7px;border:1px solid #DDE8EE;font-size:9px}}
tr:nth-child(even){{background:#F7F9FC}}
.disclaimer{{background:#FFF8E1;border-left:4px solid #F39C12;padding:8px 10px;font-size:8px;color:#7D6608}}
.action{{background:#F7F9FC;border-left:4px solid {urg_color};padding:8px 10px;margin:8px 0}}
</style></head>
<body>
<h1>KANÉA — PulmoScan AI v2.0 · Rapport d'analyse pneumologique</h1>
<p style="background:#F0F4F8;padding:6px 10px;margin:0;font-size:9px">
N° {request_id} &nbsp;·&nbsp; {date_str} &nbsp;·&nbsp; Dossier : {patient_id}
</p>
<h2>Résultat IA</h2>
<p><span class="badge" style="background:{urg_color};color:#fff">{prediction}</span>
&nbsp;&nbsp; Confiance : <b>{confidence:.1%}</b> &nbsp;&nbsp; Sévérité : <b>{severity}</b>
&nbsp;&nbsp; Urgence : <b style="color:{urg_color}">{urgency}</b></p>
<h2>Diagnostic différentiel</h2>
<table><tr><th>Classe</th><th>Probabilité</th><th>CIM-10</th></tr>{diff_rows}</table>
<h2>Action recommandée</h2>
<div class="action"><b style="color:{urg_color}">💡 {action}</b></div>
<h2>Références</h2>
<p style="font-size:9px;color:#5E7A8A">{result.get('reference_guidelines','—')}</p>
<div class="disclaimer">⚠ Outil d'aide à la décision médicale — ne remplace pas le diagnostic clinique.
Interprétation sous supervision médicale obligatoire. KANÉA 2026.</div>
</body></html>"""
