"""
RetinaVision AI — Rapport PDF médical ophtalmologique A4
=========================================================
Contient : diagnostic IA, ETDRS, AREDS2, CDR, CRT, VFI,
Grad-CAM anatomique, stade, recommandations, QR Code.

Format : A4 multi-pages · imprimable hôpital · signature numérique.
"""
from __future__ import annotations

import base64
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
        HRFlowable, Image as RLImage, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    _RL_OK = True
except ImportError:
    _RL_OK = False

try:
    import qrcode as _qrc
    _QR_OK = True
except ImportError:
    _QR_OK = False

# ─── Palette de couleurs ophtalmologique ─────────────────────────────────────
_CYAN   = colors.HexColor("#0D9ED9") if _RL_OK else None
_NAVY   = colors.HexColor("#1A2B4A") if _RL_OK else None
_TEAL   = colors.HexColor("#0E7490") if _RL_OK else None
_GREEN  = colors.HexColor("#10B981") if _RL_OK else None
_RED    = colors.HexColor("#DC2626") if _RL_OK else None
_ORANGE = colors.HexColor("#EA580C") if _RL_OK else None
_PURPLE = colors.HexColor("#7C3AED") if _RL_OK else None
_LIGHT  = colors.HexColor("#EFF6FF") if _RL_OK else None
_MUTED  = colors.HexColor("#64748B") if _RL_OK else None
_WHITE  = colors.white               if _RL_OK else None
_DARK   = colors.HexColor("#0F172A") if _RL_OK else None

_URG_COLOR = {
    "Urgence absolue": colors.HexColor("#7B241C") if _RL_OK else None,
    "Urgente":         colors.HexColor("#C0392B") if _RL_OK else None,
    "Élevée":          colors.HexColor("#E74C3C") if _RL_OK else None,
    "Modérée":         colors.HexColor("#E67E22") if _RL_OK else None,
    "Faible":          colors.HexColor("#27AE60") if _RL_OK else None,
}

_URG_HEX = {
    "Urgence absolue": "7B241C",
    "Urgente":         "C0392B",
    "Élevée":          "E74C3C",
    "Modérée":         "E67E22",
    "Faible":          "27AE60",
}


def _s(name, **kw):
    return ParagraphStyle(name, **kw)


def _qr(text: str, size: int = 60) -> Any | None:
    if not _QR_OK or not _RL_OK:
        return None
    try:
        qr = _qrc.QRCode(box_size=2, border=2)
        qr.add_data(text)
        qr.make(fit=True)
        img = qr.make_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return RLImage(buf, width=size, height=size)
    except Exception:
        return None


def _b64_to_rl_image(b64str: str, width: float = 120, height: float = 120) -> Any | None:
    if not _RL_OK or not b64str:
        return None
    try:
        buf = io.BytesIO(base64.b64decode(b64str))
        return RLImage(buf, width=width, height=height)
    except Exception:
        return None


def generate_retina_report(result: dict[str, Any], patient_info: dict | None = None) -> bytes | None:
    """
    Génère le rapport PDF médical complet RetinaVision AI.
    Retourne les bytes du PDF ou None si ReportLab non disponible.
    """
    if not _RL_OK:
        return None

    patient_info = patient_info or {}
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm,
    )
    W = A4[0] - 30 * mm

    now       = datetime.now()
    report_id = str(uuid.uuid4())[:12].upper()
    prediction  = result.get("prediction", "—")
    confidence  = result.get("confidence", 0)
    severity    = result.get("severity", "—")
    urgency     = result.get("overall_urgency", result.get("clinical_profile", {}).get("urgency", "Faible"))
    profile     = result.get("clinical_profile", {})
    scores      = result.get("clinical_scores", {})
    quant       = result.get("quantification", {})
    diff        = result.get("differential_diagnosis", [])
    safety      = result.get("clinical_safety", {})
    xai         = result.get("explainability", {})
    heatmap_b64 = xai.get("heatmap_b64", "")
    urg_color   = _URG_COLOR.get(urgency, _ORANGE)
    urg_hex     = _URG_HEX.get(urgency, "E67E22")

    # ── Styles ────────────────────────────────────────────────────────────────
    TITLE = _s("Title", fontSize=18, textColor=_WHITE, fontName="Helvetica-Bold",
               alignment=TA_CENTER, spaceAfter=2)
    SUB   = _s("Sub",   fontSize=10, textColor=_LIGHT, fontName="Helvetica",
               alignment=TA_CENTER, spaceAfter=2)
    H2    = _s("H2",    fontSize=12, textColor=_NAVY, fontName="Helvetica-Bold",
               spaceBefore=8, spaceAfter=4)
    H3    = _s("H3",    fontSize=10, textColor=_TEAL, fontName="Helvetica-Bold",
               spaceBefore=5, spaceAfter=3)
    BODY  = _s("Body",  fontSize=8.5, textColor=_DARK, fontName="Helvetica",
               spaceBefore=2, spaceAfter=2, leading=12)
    BOLD  = _s("Bold",  fontSize=8.5, textColor=_DARK, fontName="Helvetica-Bold",
               spaceBefore=2, spaceAfter=2)
    SMALL = _s("Small", fontSize=7.5, textColor=_MUTED, fontName="Helvetica",
               spaceAfter=1)
    WARN  = _s("Warn",  fontSize=8.5, textColor=_WHITE, fontName="Helvetica-Bold",
               alignment=TA_CENTER, spaceAfter=2)
    DIAG  = _s("Diag",  fontSize=14, textColor=_WHITE, fontName="Helvetica-Bold",
               alignment=TA_CENTER)

    story = []

    # ════════════════════════════════════════════════════════════════════════════
    # PAGE 1 — EN-TÊTE + DIAGNOSTIC PRINCIPAL
    # ════════════════════════════════════════════════════════════════════════════

    # Header principal
    header_tbl = Table([[
        Paragraph("<b>KANEA</b>", _s("H", fontSize=22, textColor=_CYAN,
                                     fontName="Helvetica-Bold", alignment=TA_LEFT)),
        Paragraph("RetinaVision AI<br/><font size='9' color='#94A3B8'>Rapport ophtalmologique · Rétinologie IA</font>",
                  _s("C", fontSize=14, textColor=_WHITE, fontName="Helvetica-Bold",
                     alignment=TA_CENTER)),
        _qr(f"KANEA-RETINA-{report_id}", size=50) or Paragraph("", BODY),
    ]], colWidths=[W * 0.22, W * 0.56, W * 0.22])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), _NAVY),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",        (0, 0), (0, 0),   "LEFT"),
        ("ALIGN",        (1, 0), (1, 0),   "CENTER"),
        ("ALIGN",        (2, 0), (2, 0),   "RIGHT"),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [4, 4, 4, 4]),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 5))

    # Bande méta-données
    meta = Table([[
        Paragraph(f"<b>ID Rapport :</b> RETINA-{report_id}", SMALL),
        Paragraph(f"<b>Date :</b> {now.strftime('%d/%m/%Y %H:%M')}", SMALL),
        Paragraph(f"<b>Module :</b> RetinaVision AI v1.0", SMALL),
        Paragraph(f"<b>Modèle :</b> EfficientNet-B3 ONNX", SMALL),
    ]], colWidths=[W / 4] * 4)
    meta.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _LIGHT),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
        ("TEXTCOLOR",     (0, 0), (-1, -1), _MUTED),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("BOX",           (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
    ]))
    story.append(meta)
    story.append(Spacer(1, 6))

    # Informations patient
    pat_name = patient_info.get("name", "Patient Anonyme")
    pat_age  = patient_info.get("age", "—")
    pat_sex  = patient_info.get("sex", "—")
    pat_id   = patient_info.get("id", f"PAT-{report_id[:6]}")
    pat_diab = patient_info.get("diabete", "Non précisé")
    pat_hta  = patient_info.get("hypertension", "Non précisé")

    pat_tbl = Table([
        [Paragraph(f"<b>Patient :</b> {pat_name}", BOLD),
         Paragraph(f"<b>Âge :</b> {pat_age} ans", BODY),
         Paragraph(f"<b>Sexe :</b> {pat_sex}", BODY),
         Paragraph(f"<b>ID :</b> {pat_id}", BODY)],
        [Paragraph(f"<b>Diabète :</b> {pat_diab}", BODY),
         Paragraph(f"<b>HTA :</b> {pat_hta}", BODY),
         Paragraph(f"<b>Œil :</b> {patient_info.get('eye', 'Droit (OD)')}", BODY),
         Paragraph(f"<b>Opérateur :</b> {patient_info.get('operator', 'KANEA IA')}", BODY)],
    ], colWidths=[W * 0.30, W * 0.23, W * 0.23, W * 0.24])
    pat_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("BOX",           (0, 0), (-1, -1), 0.5, _CYAN),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
    ]))
    story.append(pat_tbl)
    story.append(Spacer(1, 8))

    # ── Bloc diagnostic principal ─────────────────────────────────────────────
    diag_color = colors.HexColor(f"#{urg_hex}")
    diag_tbl = Table([[
        Table([[
            [Paragraph("DIAGNOSTIC IA", _s("DH", fontSize=8, textColor=colors.HexColor("#94A3B8"),
                                           fontName="Helvetica-Bold", alignment=TA_CENTER))],
            [Paragraph(prediction, _s("DP", fontSize=13, textColor=_WHITE,
                                      fontName="Helvetica-Bold", alignment=TA_CENTER,
                                      leading=16))],
            [Paragraph(f"<b>Confiance IA : {confidence:.1%}</b>", _s("DC", fontSize=10,
                                                                       textColor=_CYAN,
                                                                       fontName="Helvetica-Bold",
                                                                       alignment=TA_CENTER))],
        ]], colWidths=[W * 0.52]),
        Table([[
            [Paragraph("SÉVÉRITÉ", _s("SH", fontSize=8, textColor=colors.HexColor("#94A3B8"),
                                      fontName="Helvetica-Bold", alignment=TA_CENTER))],
            [Paragraph(f"<b>{severity}</b>",
                       _s("SV", fontSize=16, textColor=urg_color,
                          fontName="Helvetica-Bold", alignment=TA_CENTER))],
            [Paragraph(f"Urgence : {urgency}", _s("SU", fontSize=9,
                                                   textColor=_MUTED, fontName="Helvetica",
                                                   alignment=TA_CENTER))],
        ]], colWidths=[W * 0.38]),
    ]], colWidths=[W * 0.60, W * 0.40])
    diag_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, 0), _NAVY),
        ("BACKGROUND",    (1, 0), (1, 0), colors.HexColor("#F8FAFC")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("BOX",           (0, 0), (-1, -1), 1.0, _CYAN),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROUNDEDCORNERS",(0, 0), (-1, -1), [4, 4, 4, 4]),
    ]))
    story.append(diag_tbl)
    story.append(Spacer(1, 6))

    # Alerte clinique
    if safety.get("level") in ("warning", "critical", "emergency"):
        alert_bg = {"warning": "#92400E", "critical": "#991B1B", "emergency": "#7B241C"}
        alert_bg_c = colors.HexColor(alert_bg.get(safety["level"], "#92400E"))
        alert_tbl = Table([[Paragraph(f"⚠ {safety['message']}", WARN)]],
                          colWidths=[W])
        alert_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), alert_bg_c),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ]))
        story.append(alert_tbl)
        story.append(Spacer(1, 4))

    # ── Profil clinique ────────────────────────────────────────────────────────
    story.append(Paragraph("PROFIL CLINIQUE & EXAMENS", H2))
    story.append(HRFlowable(width=W, thickness=1.5, color=_CYAN, spaceAfter=4))

    findings_text = " · ".join(profile.get("key_findings", [])) or "—"
    clin_data = [
        ["CIM-10", profile.get("icd10", "—"), "Catégorie", profile.get("category", "—")],
        ["Urgence", profile.get("urgency", "—"), "Exam. recommandé", profile.get("exam_type", "—")],
        ["Pattern clinique", profile.get("pattern", "—"), "Anti-VEGF", "Oui" if profile.get("anti_vegf_indication") else "Non"],
        ["Signes-clés", findings_text, "Laser indiqué", "Oui" if profile.get("laser_indication") else "Non"],
    ]
    clin_tbl = Table(
        [[Paragraph(str(r), BOLD) if i % 2 == 0 else Paragraph(str(r), BODY) for i, r in enumerate(row)]
         for row in clin_data],
        colWidths=[W * 0.18, W * 0.32, W * 0.18, W * 0.32],
    )
    clin_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), colors.HexColor("#EFF6FF")),
        ("BACKGROUND",    (2, 0), (2, -1), colors.HexColor("#EFF6FF")),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
        ("BOX",           (0, 0), (-1, -1), 0.5, _CYAN),
    ]))
    story.append(clin_tbl)
    story.append(Spacer(1, 8))

    # ════════════════════════════════════════════════════════════════════════════
    # SCORES CLINIQUES — ETDRS, AREDS2, CDR, CRT, VFI
    # ════════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("SCORES CLINIQUES & CLASSIFICATIONS", H2))
    story.append(HRFlowable(width=W, thickness=1.5, color=_CYAN, spaceAfter=4))

    def _score_row(label: str, value: str, detail: str, color: Any = None):
        bg = color or colors.HexColor("#F8FAFC")
        return [
            Paragraph(f"<b>{label}</b>", BOLD),
            Paragraph(str(value), _s("SV2", fontSize=9, textColor=color or _DARK,
                                     fontName="Helvetica-Bold")),
            Paragraph(detail, BODY),
        ]

    # ETDRS
    etdrs = scores.get("etdrs", {})
    story.append(Paragraph("ETDRS — Rétinopathie Diabétique", H3))
    etdrs_tbl = Table([
        _score_row("Niveau ETDRS", str(etdrs.get("level", "—")),
                   f"Stade : {etdrs.get('stage', '—')}"),
        _score_row("Risque", etdrs.get("risk", "—"),
                   etdrs.get("description", "—")),
        _score_row("Urgence ETDRS", etdrs.get("urgency", "—"),
                   etdrs.get("recommendation", "—"),
                   _URG_COLOR.get(etdrs.get("urgency", ""), None)),
    ], colWidths=[W * 0.22, W * 0.20, W * 0.58])
    etdrs_tbl.setStyle(TableStyle([
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#F8FAFC"), _WHITE]),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#93C5FD")),
    ]))
    story.append(etdrs_tbl)
    story.append(Spacer(1, 5))

    # AREDS2
    areds = scores.get("areds2", {})
    story.append(Paragraph("AREDS2 — Dégénérescence Maculaire (DMLA)", H3))
    areds_tbl = Table([
        _score_row("Catégorie AREDS2", f"Cat. {areds.get('category', '—')}",
                   areds.get("stage", "—")),
        _score_row("Risque 5 ans", areds.get("five_year_risk", "—"),
                   areds.get("description", "—")),
        _score_row("Supplémentation", areds.get("supplement", "—"),
                   areds.get("recommendation", "—")),
    ], colWidths=[W * 0.22, W * 0.20, W * 0.58])
    areds_tbl.setStyle(TableStyle([
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#F5F3FF"), _WHITE]),
        ("BOX",           (0, 0), (-1, -1), 0.5, _PURPLE),
    ]))
    story.append(areds_tbl)
    story.append(Spacer(1, 5))

    # CDR + VFI côte à côte
    cdr = scores.get("cup_to_disc", {})
    vf  = scores.get("visual_field", {})

    glau_tbl = Table([[
        Table([
            [Paragraph("CDR — Glaucome (Cup-to-Disc)", H3)],
            [Paragraph(f"<b>CDR :</b> {cdr.get('cdr', '—')}", BOLD)],
            [Paragraph(f"Stade : {cdr.get('stage', '—')}", BODY)],
            [Paragraph(f"RNFL : {cdr.get('rnfl', '—')}", BODY)],
            [Paragraph(f"Perte anneau : {cdr.get('rim_loss', '—')}", SMALL)],
            [Paragraph(f"Urgence : {cdr.get('urgency', '—')}", BOLD)],
        ], colWidths=[W * 0.46]),
        Table([
            [Paragraph("VFI — Champ Visuel (Humphrey HFA)", H3)],
            [Paragraph(f"<b>VFI :</b> {vf.get('vfi_percent', '—')}%  |  MD : {vf.get('md_db', '—')} dB", BOLD)],
            [Paragraph(f"PSD : {vf.get('psd_db', '—')} dB", BODY)],
            [Paragraph(f"Stade : {vf.get('stage', '—')}", BODY)],
            [Paragraph(f"Progression : {vf.get('rate', '—')}", SMALL)],
            [Paragraph(f"Conduite : {vf.get('recommendation', '—')}", BOLD)],
        ], colWidths=[W * 0.46]),
    ]], colWidths=[W * 0.50, W * 0.50])
    glau_tbl.setStyle(TableStyle([
        ("VALIGN",  (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BOX",     (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
    ]))
    story.append(glau_tbl)
    story.append(Spacer(1, 5))

    # CRT Œdème maculaire
    crt = scores.get("crt", {})
    story.append(Paragraph("CRT — Épaisseur Maculaire Centrale (OCT)", H3))
    crt_tbl = Table([
        _score_row("CRT (µm)", f"{crt.get('crt_um', '—')} µm",
                   f"Normal : 250–280 µm | Seuil œdème : > 300 µm (DRCR.net)"),
        _score_row("Volume mac.", f"{crt.get('volume_mm3', '—')} mm³",
                   f"Normal : 8.0–9.5 mm³ | Stade : {crt.get('stage', '—')}"),
        _score_row("Type œdème", crt.get("edema_type", "—"),
                   f"Anti-VEGF : {'Oui' if crt.get('anti_vegf') else 'Non'} | {crt.get('recommendation', '—')}"),
    ], colWidths=[W * 0.22, W * 0.20, W * 0.58])
    crt_tbl.setStyle(TableStyle([
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#FFF7ED"), _WHITE]),
        ("BOX",           (0, 0), (-1, -1), 0.5, _ORANGE),
    ]))
    story.append(crt_tbl)
    story.append(Spacer(1, 8))

    # ════════════════════════════════════════════════════════════════════════════
    # GRAD-CAM + DIAGNOSTIC DIFFÉRENTIEL
    # ════════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("VISUALISATION IA & DIAGNOSTIC DIFFÉRENTIEL", H2))
    story.append(HRFlowable(width=W, thickness=1.5, color=_CYAN, spaceAfter=4))

    # Heatmap Grad-CAM
    rl_heatmap = _b64_to_rl_image(heatmap_b64, width=140, height=140)
    cam_zone   = xai.get("anatomic_zone", "—")
    cam_method = xai.get("method", "Grad-CAM anatomique")

    if rl_heatmap:
        cam_info = Table([[
            rl_heatmap,
            Table([
                [Paragraph("<b>Grad-CAM — Carte d'activation rétinienne</b>", H3)],
                [Paragraph(f"Zone anatomique : {cam_zone}", BODY)],
                [Paragraph(f"Méthode : {cam_method}", BODY)],
                [Paragraph("La carte thermique met en évidence les régions rétiniennes "
                           "ayant influencé la décision du modèle IA. Bleu = faible activation, "
                           "Rouge = activation maximale.", SMALL)],
            ], colWidths=[W - 155]),
        ]], colWidths=[150, W - 150])
        cam_info.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("BOX",          (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ]))
        story.append(cam_info)
        story.append(Spacer(1, 6))

    # Diagnostic différentiel
    diff_header = [Paragraph(h, _s("DH2", fontSize=8.5, textColor=_WHITE,
                                    fontName="Helvetica-Bold", alignment=TA_CENTER))
                   for h in ["Diagnostic", "Probabilité", "CIM-10", "Catégorie", "Urgence"]]
    diff_rows = [diff_header]
    for i, d in enumerate(diff):
        bg = _LIGHT if i % 2 == 0 else _WHITE
        prob_pct = f"{d.get('probability', 0):.1%}"
        diff_rows.append([
            Paragraph(d.get("class", "—"), BODY),
            Paragraph(f"<b>{prob_pct}</b>", _s("P", fontSize=9, fontName="Helvetica-Bold",
                                                 textColor=_TEAL if i == 0 else _DARK,
                                                 alignment=TA_CENTER)),
            Paragraph(d.get("icd10", "—"), SMALL),
            Paragraph(d.get("category", "—"), SMALL),
            Paragraph(d.get("urgency", "—"),
                      _s("U", fontSize=8, fontName="Helvetica",
                         textColor=_URG_COLOR.get(d.get("urgency", ""), _MUTED) or _MUTED,
                         alignment=TA_CENTER)),
        ])

    diff_tbl = Table(diff_rows,
                     colWidths=[W * 0.35, W * 0.13, W * 0.12, W * 0.22, W * 0.18])
    diff_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _NAVY),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EFF6FF"), _WHITE]),
        ("BOX",           (0, 0), (-1, -1), 0.5, _CYAN),
    ]))
    story.append(diff_tbl)
    story.append(Spacer(1, 8))

    # ════════════════════════════════════════════════════════════════════════════
    # QUANTIFICATION LÉSIONNELLE
    # ════════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("QUANTIFICATION LÉSIONNELLE", H2))
    story.append(HRFlowable(width=W, thickness=1.5, color=_CYAN, spaceAfter=4))

    quant_data = [
        ["Paramètre", "Valeur", "Seuil / Référence", "Interprétation"],
        ["Surface lésionnelle (estimation)", f"{quant.get('lesion_coverage_pct', 0):.1f}%",
         "< 5% = minimal", "Proportion rétine atteinte"],
        ["Ratio hémorragies", f"{quant.get('hemorrhage_ratio', 0):.4f}",
         "Seuil NPDR : > 0.02", "Hémorragies intra-rétiniennes estimées"],
        ["Ratio exsudats durs", f"{quant.get('exudate_ratio', 0):.4f}",
         "Seuil CSME : > 0.05", "Exsudats lipidiques périfovéaux"],
        ["Ratio drusen", f"{quant.get('drusen_ratio', 0):.4f}",
         "AREDS2 : > 0.03 = moyen", "Dépôts drusen maculaires"],
        ["Ratio pigmentation", f"{quant.get('pigment_ratio', 0):.4f}",
         "RP : > 0.06", "Pigmentation pathologique périphérique"],
        ["Microanévrismes (est.)", f"{quant.get('microaneurysm_count_est', 0):.1f}",
         "ETDRS : > 5 = légère", "Comptage microanévrismes estimé"],
        ["Brightness fovéale", f"{quant.get('foveal_brightness', 0):.3f}",
         "Normale : 0.35–0.55", "Luminosité région fovéale (OCT)"],
        ["Indice pâleur disque", f"{quant.get('disc_pallor_index', 0):.3f}",
         "Normal : < 0.55", "Pâleur papille optique (glaucome)"],
    ]
    quant_tbl = Table(
        [[Paragraph(str(c), _s("QH", fontSize=8, textColor=_WHITE, fontName="Helvetica-Bold",
                               alignment=TA_CENTER) if r == 0 else BODY, )
          for c in row] for r, row in enumerate(quant_data)],
        colWidths=[W * 0.30, W * 0.15, W * 0.25, W * 0.30],
    )
    # Rebuild header as Paragraph list
    q_rows = []
    for r, row in enumerate(quant_data):
        if r == 0:
            q_rows.append([Paragraph(c, _s("QH2", fontSize=8, textColor=_WHITE,
                                           fontName="Helvetica-Bold", alignment=TA_CENTER))
                           for c in row])
        else:
            q_rows.append([Paragraph(str(c), BODY) for c in row])
    quant_tbl2 = Table(q_rows, colWidths=[W * 0.30, W * 0.15, W * 0.25, W * 0.30])
    quant_tbl2.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _TEAL),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ECFDF5"), _WHITE]),
        ("BOX",           (0, 0), (-1, -1), 0.5, _TEAL),
    ]))
    story.append(quant_tbl2)
    story.append(Spacer(1, 8))

    # ════════════════════════════════════════════════════════════════════════════
    # RISQUE VASCULAIRE
    # ════════════════════════════════════════════════════════════════════════════
    vasc = scores.get("vascular_risk", {})
    story.append(Paragraph("RISQUE VASCULAIRE RÉTINIEN", H2))
    story.append(HRFlowable(width=W, thickness=1.5, color=_CYAN, spaceAfter=4))

    vasc_findings = " · ".join(vasc.get("findings", [])) or "Aucune anomalie vasculaire détectée"
    vasc_tbl = Table([
        [Paragraph("<b>Score vasculaire</b>", BOLD),
         Paragraph(f"{vasc.get('score', 0)}/16", _s("VS", fontSize=14, textColor=_NAVY,
                                                      fontName="Helvetica-Bold", alignment=TA_CENTER)),
         Paragraph(f"<b>Niveau :</b> {vasc.get('level', '—')}", BOLD)],
        [Paragraph("<b>Signes rétiniens</b>", BOLD),
         Paragraph(vasc_findings, BODY), Paragraph("", BODY)],
        [Paragraph("<b>Conduite à tenir</b>", BOLD),
         Paragraph(vasc.get("recommendation", "—"), BODY), Paragraph("", BODY)],
    ], colWidths=[W * 0.22, W * 0.38, W * 0.40])
    vasc_tbl.setStyle(TableStyle([
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("SPAN",          (1, 1), (2, 1)),
        ("SPAN",          (1, 2), (2, 2)),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#FFF1F2"), _WHITE, colors.HexColor("#FFF1F2")]),
        ("BOX",           (0, 0), (-1, -1), 0.5, _RED),
    ]))
    story.append(vasc_tbl)
    story.append(Spacer(1, 8))

    # ════════════════════════════════════════════════════════════════════════════
    # RECOMMANDATIONS CLINIQUES FINALES
    # ════════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("RECOMMANDATIONS CLINIQUES & RÉFÉRENCES", H2))
    story.append(HRFlowable(width=W, thickness=1.5, color=_CYAN, spaceAfter=4))

    action_text = profile.get("action", result.get("recommended_action", "—"))
    guide_text  = profile.get("guidelines", result.get("reference_guidelines", "—"))
    proc_ms     = result.get("processing_ms", "—")

    rec_tbl = Table([
        [Paragraph("<b>Conduite à tenir</b>", BOLD),
         Paragraph(action_text, BODY)],
        [Paragraph("<b>Références scientifiques</b>", BOLD),
         Paragraph(guide_text, SMALL)],
        [Paragraph("<b>Avertissement IA</b>", BOLD),
         Paragraph("Ce rapport est généré par une IA à visée d'aide au diagnostic. "
                   "Il ne remplace pas l'avis d'un ophtalmologue ou rétinologue qualifié. "
                   "Toute décision thérapeutique doit être validée par un médecin.", SMALL)],
        [Paragraph("<b>Temps traitement</b>", BOLD),
         Paragraph(f"{proc_ms} ms · Modèle EfficientNet-B3 ONNX CPU", SMALL)],
    ], colWidths=[W * 0.25, W * 0.75])
    rec_tbl.setStyle(TableStyle([
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("BACKGROUND",    (0, 0), (0, -1), colors.HexColor("#EFF6FF")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#BFDBFE")),
        ("BOX",           (0, 0), (-1, -1), 0.8, _CYAN),
    ]))
    story.append(rec_tbl)
    story.append(Spacer(1, 8))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=1, color=_MUTED, spaceAfter=3))
    footer_tbl = Table([[
        Paragraph(f"KANEA — RetinaVision AI v1.0 · Knowledge Anthropology & Neural Engine for Africa",
                  _s("F", fontSize=7, textColor=_MUTED, fontName="Helvetica")),
        Paragraph(f"Rapport {report_id} · {now.strftime('%d/%m/%Y %H:%M')} · Confiance IA : {confidence:.1%}",
                  _s("F2", fontSize=7, textColor=_MUTED, fontName="Helvetica", alignment=TA_RIGHT)),
    ]], colWidths=[W * 0.60, W * 0.40])
    footer_tbl.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
    ]))
    story.append(footer_tbl)

    doc.build(story)
    return buf.getvalue()
