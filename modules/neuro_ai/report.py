"""
NeuroVision AI — Rapport PDF médical neurologique A4
=====================================================
Contient : diagnostic IA, NIHSS, mRS, ASPECTS, GCS, Marshall,
MMSE, MoCA, CDR, Hoehn-Yahr, EDSS, ICH Score,
Grad-CAM cérébral, stade, recommandations, fenêtre thérapeutique, QR Code.
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
        HRFlowable, Image as RLImage,
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

_NAVY   = colors.HexColor("#0F1B2D") if _RL_OK else None
_BLUE   = colors.HexColor("#1D4ED8") if _RL_OK else None
_CYAN   = colors.HexColor("#0891B2") if _RL_OK else None
_GREEN  = colors.HexColor("#059669") if _RL_OK else None
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


def _s(name, **kw):
    return ParagraphStyle(name, **kw)


def _qr(text: str, size: int = 55) -> Any | None:
    if not _QR_OK or not _RL_OK:
        return None
    try:
        qr = _qrc.QRCode(box_size=2, border=2)
        qr.add_data(text); qr.make(fit=True)
        img = qr.make_image(); buf = io.BytesIO()
        img.save(buf, format="PNG"); buf.seek(0)
        return RLImage(buf, width=size, height=size)
    except Exception:
        return None


def _b64_img(b64: str, w: float = 130, h: float = 130) -> Any | None:
    if not _RL_OK or not b64:
        return None
    try:
        return RLImage(io.BytesIO(base64.b64decode(b64)), width=w, height=h)
    except Exception:
        return None


def generate_neuro_report(result: dict[str, Any], patient_info: dict | None = None) -> bytes | None:
    if not _RL_OK:
        return None

    patient_info = patient_info or {}
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                              leftMargin=14*mm, rightMargin=14*mm,
                              topMargin=11*mm, bottomMargin=11*mm)
    W = A4[0] - 28*mm
    now = datetime.now()
    rid = str(uuid.uuid4())[:12].upper()

    pred     = result.get("prediction", "—")
    conf     = result.get("confidence", 0)
    severity = result.get("severity", "—")
    urgency  = result.get("overall_urgency", result.get("clinical_profile", {}).get("urgency", "Faible"))
    profile  = result.get("clinical_profile", {})
    scores   = result.get("clinical_scores", {})
    quant    = result.get("quantification", {})
    diff     = result.get("differential_diagnosis", [])
    safety   = result.get("clinical_safety", {})
    xai      = result.get("explainability", {})
    hm_b64   = xai.get("heatmap_b64", "")
    urg_c    = _URG_COLOR.get(urgency, _ORANGE)

    TITLE = _s("T", fontSize=16, textColor=_WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
    H2    = _s("H2", fontSize=11, textColor=_NAVY, fontName="Helvetica-Bold", spaceBefore=7, spaceAfter=3)
    H3    = _s("H3", fontSize=9.5, textColor=_BLUE, fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=2)
    BODY  = _s("B", fontSize=8.5, textColor=_DARK, fontName="Helvetica", spaceBefore=1, spaceAfter=1, leading=12)
    BOLD  = _s("BD", fontSize=8.5, textColor=_DARK, fontName="Helvetica-Bold", spaceBefore=1, spaceAfter=1)
    SMALL = _s("S", fontSize=7.5, textColor=_MUTED, fontName="Helvetica", spaceAfter=1)
    WARN  = _s("W", fontSize=8.5, textColor=_WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = Table([[
        Paragraph("<b>KANEA</b>", _s("KH", fontSize=20, textColor=colors.HexColor("#60A5FA"),
                                     fontName="Helvetica-Bold", alignment=TA_LEFT)),
        Paragraph("NeuroVision AI<br/><font size='9' color='#94A3B8'>Rapport neurologique · Neuroradiologie IA</font>",
                  _s("NH", fontSize=13, textColor=_WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)),
        _qr(f"KANEA-NEURO-{rid}", 48) or Paragraph("", BODY),
    ]], colWidths=[W*0.22, W*0.56, W*0.22])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), _NAVY),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 8), ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING",   (0,0),(-1,-1), 10), ("RIGHTPADDING", (0,0),(-1,-1), 10),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 4))

    meta = Table([[
        Paragraph(f"<b>ID :</b> NEURO-{rid}", SMALL),
        Paragraph(f"<b>Date :</b> {now.strftime('%d/%m/%Y %H:%M')}", SMALL),
        Paragraph("<b>Module :</b> NeuroVision AI v1.0", SMALL),
        Paragraph("<b>Modèle :</b> EfficientNet-B4 ONNX", SMALL),
    ]], colWidths=[W/4]*4)
    meta.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#F1F5F9")),
        ("FONTSIZE",(0,0),(-1,-1),7.5), ("TEXTCOLOR",(0,0),(-1,-1),_MUTED),
        ("TOPPADDING",(0,0),(-1,-1),3), ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",(0,0),(-1,-1),8),
        ("BOX",(0,0),(-1,-1),0.3,colors.HexColor("#CBD5E1")),
    ]))
    story.append(meta)
    story.append(Spacer(1, 5))

    # Infos patient
    pat_tbl = Table([
        [Paragraph(f"<b>Patient :</b> {patient_info.get('name','Anonyme')}", BOLD),
         Paragraph(f"<b>Âge :</b> {patient_info.get('age','—')} ans", BODY),
         Paragraph(f"<b>Sexe :</b> {patient_info.get('sex','—')}", BODY),
         Paragraph(f"<b>ID :</b> PAT-{rid[:6]}", BODY)],
        [Paragraph(f"<b>Modalité :</b> {patient_info.get('modality', profile.get('modality','—'))}", BODY),
         Paragraph(f"<b>Côté :</b> {patient_info.get('side','Bilatéral')}", BODY),
         Paragraph(f"<b>Opérateur :</b> {patient_info.get('operator','KANEA IA')}", BODY),
         Paragraph(f"<b>Service :</b> {patient_info.get('service','Neurologie/Neuroradiologie')}", BODY)],
    ], colWidths=[W*0.30, W*0.23, W*0.23, W*0.24])
    pat_tbl.setStyle(TableStyle([
        ("FONTSIZE",(0,0),(-1,-1),8.5), ("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4), ("LEFTPADDING",(0,0),(-1,-1),7),
        ("BOX",(0,0),(-1,-1),0.5,_BLUE), ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#E2E8F0")),
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#F8FAFC")),
    ]))
    story.append(pat_tbl)
    story.append(Spacer(1, 7))

    # ── Diagnostic principal ───────────────────────────────────────────────────
    diag_tbl = Table([[
        Table([
            [Paragraph("DIAGNOSTIC NeuroVision AI", _s("DL", fontSize=7.5, textColor=colors.HexColor("#94A3B8"),
                                                        fontName="Helvetica-Bold", alignment=TA_CENTER))],
            [Paragraph(pred, _s("DP", fontSize=12, textColor=_WHITE, fontName="Helvetica-Bold",
                                alignment=TA_CENTER, leading=15))],
            [Paragraph(f"<b>Confiance IA : {conf:.1%}</b>",
                       _s("DC", fontSize=10, textColor=colors.HexColor("#60A5FA"),
                          fontName="Helvetica-Bold", alignment=TA_CENTER))],
        ], colWidths=[W*0.54]),
        Table([
            [Paragraph("SÉVÉRITÉ / URGENCE", _s("SL", fontSize=7.5, textColor=colors.HexColor("#94A3B8"),
                                                fontName="Helvetica-Bold", alignment=TA_CENTER))],
            [Paragraph(f"<b>{severity}</b>", _s("SV", fontSize=15, textColor=urg_c,
                                                fontName="Helvetica-Bold", alignment=TA_CENTER))],
            [Paragraph(f"{urgency}", _s("SU", fontSize=9, textColor=_MUTED,
                                        fontName="Helvetica", alignment=TA_CENTER))],
        ], colWidths=[W*0.38]),
    ]], colWidths=[W*0.60, W*0.40])
    diag_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,0),_NAVY), ("BACKGROUND",(1,0),(1,0),colors.HexColor("#F8FAFC")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),9), ("BOTTOMPADDING",(0,0),(-1,-1),9),
        ("LEFTPADDING",(0,0),(-1,-1),10),
        ("BOX",(0,0),(-1,-1),1.0,_BLUE),
        ("INNERGRID",(0,0),(-1,-1),0.5,colors.HexColor("#CBD5E1")),
    ]))
    story.append(diag_tbl)
    story.append(Spacer(1, 5))

    # Alerte
    if safety.get("level") in ("warning","critical","emergency"):
        abg = {"warning":"#92400E","critical":"#991B1B","emergency":"#7B241C"}.get(safety["level"],"#92400E")
        alert_t = Table([[Paragraph(f"⚠ {safety['message']}", WARN)]], colWidths=[W])
        alert_t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),colors.HexColor(abg)),
            ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LEFTPADDING",(0,0),(-1,-1),10),
        ]))
        story.append(alert_t)
        story.append(Spacer(1, 4))

    # Fenêtre thérapeutique
    window = profile.get("therapeutic_window", result.get("therapeutic_window","—"))
    if window and window != "—":
        win_t = Table([[
            Paragraph("<b>⏱ FENÊTRE THÉRAPEUTIQUE</b>", _s("WH", fontSize=8.5, textColor=_WHITE,
                                                            fontName="Helvetica-Bold")),
            Paragraph(window, _s("WB", fontSize=8.5, textColor=_WHITE, fontName="Helvetica")),
        ]], colWidths=[W*0.32, W*0.68])
        win_t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#B91C1C")),
            ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),10), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(win_t)
        story.append(Spacer(1, 5))

    # ── Profil clinique ────────────────────────────────────────────────────────
    story.append(Paragraph("PROFIL CLINIQUE & IMAGERIE", H2))
    story.append(HRFlowable(width=W, thickness=1.5, color=_BLUE, spaceAfter=3))

    findings_txt = " · ".join(profile.get("key_findings",[]))
    clin_tbl = Table([
        [Paragraph("<b>CIM-10</b>",BOLD), Paragraph(profile.get("icd10","—"),BODY),
         Paragraph("<b>Catégorie</b>",BOLD), Paragraph(profile.get("category","—"),BODY)],
        [Paragraph("<b>Urgence</b>",BOLD), Paragraph(profile.get("urgency","—"),BODY),
         Paragraph("<b>Modalité d'imagerie</b>",BOLD), Paragraph(profile.get("modality","—"),BODY)],
        [Paragraph("<b>Pattern clinique</b>",BOLD), Paragraph(profile.get("pattern","—"),BODY),
         Paragraph("<b>Conduite</b>",BOLD), Paragraph(profile.get("action","—"),BODY)],
        [Paragraph("<b>Signes-clés</b>",BOLD), Paragraph(findings_txt,BODY),
         Paragraph("<b>Références</b>",BOLD), Paragraph(profile.get("guidelines","—"),SMALL)],
    ], colWidths=[W*0.18, W*0.32, W*0.18, W*0.32])
    clin_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#EFF6FF")),
        ("BACKGROUND",(2,0),(2,-1),colors.HexColor("#EFF6FF")),
        ("FONTSIZE",(0,0),(-1,-1),8.5), ("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3), ("LEFTPADDING",(0,0),(-1,-1),6),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#CBD5E1")),
        ("BOX",(0,0),(-1,-1),0.5,_BLUE),
    ]))
    story.append(clin_tbl)
    story.append(Spacer(1, 7))

    # ── Scores AVC ───────────────────────────────────────────────────────────
    story.append(Paragraph("SCORES AVC — NIHSS · mRS · ASPECTS · ICH Score", H2))
    story.append(HRFlowable(width=W, thickness=1.5, color=_BLUE, spaceAfter=3))

    nihss = scores.get("nihss",{}); mrs = scores.get("mrs",{}); asp = scores.get("aspects",{})
    ich   = scores.get("ich_score",{})

    def _sr(label, val, detail, color=None):
        return [Paragraph(f"<b>{label}</b>",BOLD),
                Paragraph(str(val), _s("V",fontSize=9,textColor=color or _DARK,fontName="Helvetica-Bold")),
                Paragraph(detail,BODY)]

    avc_tbl = Table([
        _sr("NIHSS Total", nihss.get("total","—"), nihss.get("stage","—")),
        _sr("Thrombolyse IV", "Éligible" if nihss.get("thrombolysis") else "Non indiquée",
            nihss.get("recommendation","—")),
        _sr("mRS Score", f"{mrs.get('score','—')}/6", mrs.get("level","—")),
        _sr("Autonomie", mrs.get("autonomy","—"), mrs.get("clinical_use","—")),
        _sr("ASPECTS", f"{asp.get('score','—')}/10", asp.get("outcome","—")),
        _sr("Thrombectomie", asp.get("thrombectomy_benefit","—"), asp.get("recommendation","—")),
        _sr("ICH Score", f"{ich.get('total','—')}/6",
            f"Mortalité J30 : {ich.get('mortality_30d','—')} — Bon outcome : {ich.get('good_outcome','—')}"),
    ], colWidths=[W*0.22, W*0.18, W*0.60])
    avc_tbl.setStyle(TableStyle([
        ("FONTSIZE",(0,0),(-1,-1),8.5), ("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3), ("LEFTPADDING",(0,0),(-1,-1),6),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#EFF6FF"),_WHITE]),
        ("BOX",(0,0),(-1,-1),0.5,_BLUE),
    ]))
    story.append(avc_tbl)
    story.append(Spacer(1, 6))

    # ── Scores Traumatisme — GCS + Marshall ──────────────────────────────────
    story.append(Paragraph("TRAUMATISME CRÂNIEN — GCS · Marshall", H2))
    story.append(HRFlowable(width=W, thickness=1.5, color=_BLUE, spaceAfter=3))

    gcs = scores.get("gcs",{}); marsh = scores.get("marshall",{})
    tbi_tbl = Table([
        _sr("GCS Total", f"{gcs.get('total','—')}/15",
            f"Yeux {gcs.get('eye','—')} · Verbal {gcs.get('verbal','—')} · Moteur {gcs.get('motor','—')}"),
        _sr("Sévérité TBI", gcs.get("severity","—"),
            f"Intubation : {'Oui' if gcs.get('intubation') else 'Non'} · Monitoring PIC : {'Oui' if gcs.get('icp_monitoring') else 'Non'}"),
        _sr("Conduite GCS", "", gcs.get("recommendation","—")),
        _sr("Marshall Grade", f"{marsh.get('grade','—')}/6", marsh.get("label","—")),
        _sr("Description", marsh.get("description","—"), f"Mortalité : {marsh.get('mortality','—')} · ICP : {marsh.get('icp_risk','—')}"),
        _sr("Conduite Marshall", "", marsh.get("recommendation","—")),
    ], colWidths=[W*0.22, W*0.18, W*0.60])
    tbi_tbl.setStyle(TableStyle([
        ("FONTSIZE",(0,0),(-1,-1),8.5), ("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3), ("LEFTPADDING",(0,0),(-1,-1),6),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#FFF7ED"),_WHITE]),
        ("BOX",(0,0),(-1,-1),0.5,_ORANGE),
    ]))
    story.append(tbi_tbl)
    story.append(Spacer(1, 6))

    # ── Neurodégénératives — MMSE · MoCA · CDR · Parkinson · EDSS ────────────
    story.append(Paragraph("NEURODÉGÉNÉRATIVES & SEP — MMSE · CDR · UPDRS · EDSS", H2))
    story.append(HRFlowable(width=W, thickness=1.5, color=_BLUE, spaceAfter=3))

    mmse = scores.get("mmse",{}); cdr = scores.get("cdr",{})
    pk   = scores.get("parkinson",{}); edss = scores.get("edss",{})

    ndeg_tbl = Table([
        _sr("MMSE", f"{mmse.get('score','—')}/30", mmse.get("severity","—")),
        _sr("Stade Alzheimer", mmse.get("stage","—"), mmse.get("recommendation","—")),
        _sr("CDR Score", str(cdr.get("cdr","—")), cdr.get("stage","—")),
        _sr("CDR SoB", f"{cdr.get('sb','—')}/18", f"Progression : {cdr.get('progression','—')}"),
        _sr("Hoehn & Yahr", str(pk.get("hy_stage","—")), pk.get("hy_desc","—")),
        _sr("UPDRS III", f"{pk.get('updrs_iii','—')}/132",
            f"Sévérité : {pk.get('severity','—')} — DBS candidat : {'Oui' if pk.get('dbs_candidate') else 'Non'}"),
        _sr("EDSS (SEP)", f"{edss.get('score','—')}/10", edss.get("stage","—")),
        _sr("Marche", edss.get("walking","—"), edss.get("treatment","—")),
    ], colWidths=[W*0.22, W*0.18, W*0.60])
    ndeg_tbl.setStyle(TableStyle([
        ("FONTSIZE",(0,0),(-1,-1),8.5), ("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3), ("LEFTPADDING",(0,0),(-1,-1),6),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#F5F3FF"),_WHITE]),
        ("BOX",(0,0),(-1,-1),0.5,_PURPLE),
    ]))
    story.append(ndeg_tbl)
    story.append(Spacer(1, 6))

    # ── Grad-CAM + Différentiel ───────────────────────────────────────────────
    story.append(Paragraph("VISUALISATION IA & DIAGNOSTIC DIFFÉRENTIEL", H2))
    story.append(HRFlowable(width=W, thickness=1.5, color=_BLUE, spaceAfter=3))

    rl_cam = _b64_img(hm_b64, 135, 135)
    cam_zone = xai.get("anatomic_zone","—")
    if rl_cam:
        cam_row = Table([[
            rl_cam,
            Table([
                [Paragraph("<b>Grad-CAM — Cartographie cérébrale IA</b>", H3)],
                [Paragraph(f"Zone anatomique : {cam_zone}", BODY)],
                [Paragraph("Méthode : Grad-CAM anatomique adaptatif (fovéa/hémisphère/substance blanche/hippocampe)", SMALL)],
                [Paragraph("La carte thermique met en évidence les régions cérébrales ayant "
                           "influencé la décision du modèle IA. Bleu = faible activation, "
                           "Rouge/Jaune = activation maximale.", SMALL)],
            ], colWidths=[W-148])
        ]], colWidths=[145, W-145])
        cam_row.setStyle(TableStyle([
            ("VALIGN",(0,0),(-1,-1),"TOP"), ("LEFTPADDING",(0,0),(-1,-1),6),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#E2E8F0")),
            ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#F8FAFC")),
            ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ]))
        story.append(cam_row)
        story.append(Spacer(1, 5))

    diff_hdr = [Paragraph(h, _s("DH", fontSize=8, textColor=_WHITE, fontName="Helvetica-Bold",
                                 alignment=TA_CENTER))
                for h in ["Diagnostic", "Probabilité", "CIM-10", "Catégorie", "Urgence"]]
    diff_rows = [diff_hdr]
    for i, d in enumerate(diff):
        diff_rows.append([
            Paragraph(d.get("class","—"), BODY),
            Paragraph(f"<b>{d.get('probability',0):.1%}</b>",
                      _s("P", fontSize=9, fontName="Helvetica-Bold",
                         textColor=_BLUE if i==0 else _DARK, alignment=TA_CENTER)),
            Paragraph(d.get("icd10","—"), SMALL),
            Paragraph(d.get("category","—"), SMALL),
            Paragraph(d.get("urgency","—"),
                      _s("U", fontSize=8, fontName="Helvetica",
                         textColor=_URG_COLOR.get(d.get("urgency",""),_MUTED) or _MUTED,
                         alignment=TA_CENTER)),
        ])
    diff_tbl = Table(diff_rows, colWidths=[W*0.36, W*0.12, W*0.12, W*0.22, W*0.18])
    diff_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),_NAVY),
        ("FONTSIZE",(0,0),(-1,-1),8.5), ("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3), ("LEFTPADDING",(0,0),(-1,-1),6),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#EFF6FF"),_WHITE]),
        ("BOX",(0,0),(-1,-1),0.5,_BLUE),
    ]))
    story.append(diff_tbl)
    story.append(Spacer(1, 6))

    # ── Quantification ─────────────────────────────────────────────────────────
    story.append(Paragraph("QUANTIFICATION LÉSIONNELLE", H2))
    story.append(HRFlowable(width=W, thickness=1.5, color=_BLUE, spaceAfter=3))

    q_rows = [[Paragraph(h, _s("QH", fontSize=8, textColor=_WHITE, fontName="Helvetica-Bold"))
               for h in ["Paramètre","Valeur","Référence","Interprétation"]]]
    q_data = [
        ("Volume lésionnel", f"{quant.get('lesion_volume_ml',0):.1f} mL", "Critique : > 30 mL", "Volume hématome/infarctus estimé"),
        ("Shift médian", f"{quant.get('midline_shift_mm',0):.1f} mm", "Chirurgie si > 5 mm", "Déviation ligne médiane"),
        ("Ratio hyperdense (CT)", f"{quant.get('hyperdense_ratio',0):.4f}", "Hémorragie : > 0.05", "Zones hyperdenses (sang/calcification)"),
        ("Ratio hypodense (CT)", f"{quant.get('hypodense_ratio',0):.4f}", "Ischémie : > 0.06", "Zones hypodenses (œdème/nécrose)"),
        ("Lésions SB (IRM)", f"{quant.get('wm_lesion_ratio',0):.4f}", "SEP : > 0.05", "Plaques substance blanche T2/FLAIR"),
        ("Atrophie cérébrale", f"{quant.get('atrophy_ratio',0):.4f}", "Sévère : > 0.15", "Élargissement sulci/espaces liquidiens"),
        ("Atrophie hippocampe", f"{quant.get('hippocampal_atrophy',0):.3f}", "Alzheimer : > 0.25", "Atrophie temporale médiane"),
        ("Asymétrie hémisphères", f"{quant.get('asymmetry_index',0):.3f}", "Pathologique : > 0.05", "Asymétrie HG/HD parenchyme"),
    ]
    for r in q_data:
        q_rows.append([Paragraph(str(c), BODY) for c in r])
    q_tbl = Table(q_rows, colWidths=[W*0.28, W*0.16, W*0.24, W*0.32])
    q_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0E7490")),
        ("FONTSIZE",(0,0),(-1,-1),8.5), ("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3), ("LEFTPADDING",(0,0),(-1,-1),6),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#ECFDF5"),_WHITE]),
        ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#0E7490")),
    ]))
    story.append(q_tbl)
    story.append(Spacer(1, 7))

    # ── Recommandations ────────────────────────────────────────────────────────
    story.append(Paragraph("RECOMMANDATIONS CLINIQUES & AVERTISSEMENT", H2))
    story.append(HRFlowable(width=W, thickness=1.5, color=_BLUE, spaceAfter=3))

    rec_tbl = Table([
        [Paragraph("<b>Conduite à tenir</b>",BOLD), Paragraph(profile.get("action",result.get("recommended_action","—")),BODY)],
        [Paragraph("<b>Fenêtre thérapeutique</b>",BOLD), Paragraph(window,BODY)],
        [Paragraph("<b>Références</b>",BOLD), Paragraph(profile.get("guidelines",result.get("reference_guidelines","—")),SMALL)],
        [Paragraph("<b>Avertissement IA</b>",BOLD),
         Paragraph("Ce rapport est généré par une IA d'aide au diagnostic. Il ne remplace pas "
                   "l'avis d'un neurologue ou neuroradiologue. Toute décision thérapeutique "
                   "doit être validée par un médecin qualifié.", SMALL)],
        [Paragraph("<b>Traitement</b>",BOLD), Paragraph(f"{result.get('processing_ms','—')} ms · EfficientNet-B4 ONNX CPU",SMALL)],
    ], colWidths=[W*0.25, W*0.75])
    rec_tbl.setStyle(TableStyle([
        ("FONTSIZE",(0,0),(-1,-1),8.5), ("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4), ("LEFTPADDING",(0,0),(-1,-1),8),
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#EFF6FF")),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#BFDBFE")),
        ("BOX",(0,0),(-1,-1),0.8,_BLUE),
    ]))
    story.append(rec_tbl)
    story.append(Spacer(1, 6))

    # Footer
    story.append(HRFlowable(width=W, thickness=0.8, color=_MUTED, spaceAfter=3))
    foot = Table([[
        Paragraph(f"KANEA — NeuroVision AI v1.0 · Knowledge Anthropology & Neural Engine for Africa",
                  _s("F",fontSize=7,textColor=_MUTED,fontName="Helvetica")),
        Paragraph(f"NEURO-{rid} · {now.strftime('%d/%m/%Y %H:%M')} · Confiance : {conf:.1%}",
                  _s("F2",fontSize=7,textColor=_MUTED,fontName="Helvetica",alignment=TA_RIGHT)),
    ]], colWidths=[W*0.60, W*0.40])
    foot.setStyle(TableStyle([
        ("TOPPADDING",(0,0),(-1,-1),2), ("BOTTOMPADDING",(0,0),(-1,-1),2),
        ("LEFTPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(foot)

    doc.build(story)
    return buf.getvalue()
