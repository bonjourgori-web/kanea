"""
HepatoScan AI — Rapport PDF hépatologique A4 multi-pages
=========================================================
Contient : informations patient · diagnostic IA · bilan biologique ·
scores hépatologiques (Child-Pugh, MELD, BCLC, LI-RADS, FIB-4, APRI, CAP,
METAVIR, ALBI, CLIF-C) · findings critiques · explainability SHAP ·
recommandations EASL/AASLD · QR code.
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
    import qrcode as _qrc
    _QR_OK = True
except ImportError:
    _QR_OK = False

# ── Palette HepatoScan (brun hépatique + orange) ──────────────────────────────
_BROWN  = colors.HexColor("#6D4C41") if _RL_OK else None
_ORANGE = colors.HexColor("#E67E22") if _RL_OK else None
_RED    = colors.HexColor("#C0392B") if _RL_OK else None
_CRIT   = colors.HexColor("#922B21") if _RL_OK else None
_GREEN  = colors.HexColor("#27AE60") if _RL_OK else None
_BLUE   = colors.HexColor("#2980B9") if _RL_OK else None
_PURPLE = colors.HexColor("#884EA0") if _RL_OK else None
_DARK   = colors.HexColor("#1A2B3C") if _RL_OK else None
_MUTED  = colors.HexColor("#5E7A8A") if _RL_OK else None
_LIGHT  = colors.HexColor("#EEF2F6") if _RL_OK else None
_WHITE  = colors.white               if _RL_OK else None

_URG_RL = {
    "Critique": colors.HexColor("#922B21") if _RL_OK else None,
    "Urgente":  colors.HexColor("#C0392B") if _RL_OK else None,
    "Élevée":   colors.HexColor("#E74C3C") if _RL_OK else None,
    "Modérée":  colors.HexColor("#E67E22") if _RL_OK else None,
    "Faible":   colors.HexColor("#27AE60") if _RL_OK else None,
}
_URG_HEX = {
    "Critique": "922B21", "Urgente": "C0392B",
    "Élevée": "E74C3C",   "Modérée": "E67E22",
    "Faible": "27AE60",
}
_SEV_HEX = {
    "CRITIQUE": "922B21", "ÉLEVÉ": "E74C3C",
    "MODÉRÉ": "E67E22",   "INFO": "2980B9",
}


def _s(name: str, **kw) -> "ParagraphStyle":
    return ParagraphStyle(name, **kw)


def _qr(text: str, size: int = 55):
    if not (_QR_OK and _RL_OK):
        return None
    try:
        from reportlab.platypus import Image as RLImage
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


def build_hepato_pdf_report(
    result: dict[str, Any],
    patient_id: str = "KANEA-HS",
    examiner: str = "KANÉA System",
    institution: str = "Hépatologie — KANÉA",
    patient_name: str = "",
    patient_age: str = "",
    patient_sex: str = "",
    clinical_context: str = "",
) -> bytes | None:
    """Génère le rapport PDF hépatologique A4 multi-pages."""
    if not _RL_OK:
        return None

    buf = io.BytesIO()
    W, H = A4
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm,  bottomMargin=16 * mm,
        title=f"HepatoScan AI — {patient_id}",
        author="KANÉA Medical AI Platform",
    )
    story = []
    pw = W - 36 * mm

    # ── Styles ────────────────────────────────────────────────────────────────
    S_title = _s("t",  fontSize=17, textColor=_BROWN, fontName="Helvetica-Bold",
                 alignment=TA_CENTER, spaceAfter=2)
    S_sub   = _s("su", fontSize=9,  textColor=_MUTED, fontName="Helvetica",
                 alignment=TA_CENTER, spaceAfter=6)
    S_h1    = _s("h1", fontSize=12, textColor=_DARK,  fontName="Helvetica-Bold",
                 spaceBefore=8, spaceAfter=4)
    S_h2    = _s("h2", fontSize=10, textColor=_BROWN, fontName="Helvetica-Bold",
                 spaceBefore=4, spaceAfter=3)
    S_body  = _s("bd", fontSize=8,  textColor=_DARK,  fontName="Helvetica",
                 spaceAfter=3, leading=12)
    S_small = _s("sm", fontSize=7,  textColor=_MUTED, fontName="Helvetica",
                 spaceAfter=2)
    S_warn  = _s("wn", fontSize=8,  textColor=_RED,   fontName="Helvetica-Bold",
                 spaceAfter=3)

    def _tbl(data, cws, extra=None):
        cmds = [
            ("FONTNAME",       (0,0),(-1,-1),"Helvetica"),
            ("FONTSIZE",       (0,0),(-1,-1),8),
            ("ROWBACKGROUNDS", (0,0),(-1,-1),[_LIGHT,_WHITE]),
            ("GRID",           (0,0),(-1,-1),0.3,colors.HexColor("#D5DCE8")),
            ("VALIGN",         (0,0),(-1,-1),"MIDDLE"),
            ("LEFTPADDING",    (0,0),(-1,-1),5),
            ("RIGHTPADDING",   (0,0),(-1,-1),5),
            ("TOPPADDING",     (0,0),(-1,-1),3),
            ("BOTTOMPADDING",  (0,0),(-1,-1),3),
        ]
        if extra:
            cmds.extend(extra)
        return Table(data, colWidths=cws, style=TableStyle(cmds))

    def _htbl(data, cws):
        cmds = [
            ("BACKGROUND",    (0,0),(-1,0),_BROWN),
            ("TEXTCOLOR",     (0,0),(-1,0),_WHITE),
            ("FONTNAME",      (0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1),8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LIGHT,_WHITE]),
            ("GRID",          (0,0),(-1,-1),0.3,colors.HexColor("#D5DCE8")),
            ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
            ("LEFTPADDING",   (0,0),(-1,-1),5),
            ("RIGHTPADDING",  (0,0),(-1,-1),5),
            ("TOPPADDING",    (0,0),(-1,-1),3),
            ("BOTTOMPADDING", (0,0),(-1,-1),3),
        ]
        return Table(data, colWidths=cws, style=TableStyle(cmds))

    now_str    = datetime.now().strftime("%d/%m/%Y à %H:%M")
    report_id  = str(uuid.uuid4())[:8].upper()
    prediction = result.get("prediction", "—")
    confidence = result.get("confidence", 0.0)
    profile    = result.get("clinical_profile", {})
    urgency    = profile.get("urgency", "—")
    action     = profile.get("action", "—")
    urg_color  = _URG_RL.get(urgency, _ORANGE)
    urg_hex    = _URG_HEX.get(urgency, "E67E22")
    scores     = result.get("clinical_scores", {})
    critical   = result.get("critical_findings", [])
    feat_imp   = result.get("explainability", {}).get("feature_importance", {})
    bio        = result.get("biological_summary", {})
    safety     = result.get("clinical_safety", {})
    surv_risk  = profile.get("survival_risk", {})

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — EN-TÊTE + DIAGNOSTIC + BIOLOGIE
    # ══════════════════════════════════════════════════════════════════════════

    story.append(Paragraph("🧬 HepatoScan AI — Rapport Hépatologique", S_title))
    story.append(Paragraph(
        f"KANÉA Medical AI Platform | {institution} | {now_str} | ID {report_id}", S_sub))
    story.append(HRFlowable(width=pw, thickness=2, color=_BROWN, spaceAfter=6))

    # Infos patient
    story.append(Paragraph("Informations Patient", S_h1))
    story.append(_tbl([
        ["ID Patient",   patient_id or "—",    "Examinateur",  examiner],
        ["Nom / Prénom", patient_name or "—",   "Âge",          patient_age or "—"],
        ["Sexe",         patient_sex or "—",    "Contexte",     clinical_context or "—"],
        ["Date rapport", now_str,                "Version IA",   result.get("model_version","v1.0")],
    ], [pw*0.20, pw*0.30, pw*0.20, pw*0.30], extra=[
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
        ("TEXTCOLOR",(0,0),(0,-1),_BROWN),("TEXTCOLOR",(2,0),(2,-1),_BROWN),
    ]))
    story.append(Spacer(1, 6))

    # Bandeau diagnostic
    diag_data = [[
        Paragraph(f'<font color="#{urg_hex}"><b>DIAGNOSTIC : {prediction}</b></font>',
                  _s("d1", fontSize=11, fontName="Helvetica-Bold", textColor=_DARK)),
        Paragraph(f'<b>Confiance : {confidence:.1%}</b><br/>'
                  f'Urgence : <font color="#{urg_hex}"><b>{urgency}</b></font>',
                  _s("d2", fontSize=9, fontName="Helvetica", textColor=_DARK, alignment=TA_RIGHT)),
    ]]
    story.append(Table(diag_data, colWidths=[pw*0.68, pw*0.32], style=TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor(f"#{urg_hex}20")),
        ("BOX",(0,0),(-1,-1),1.5,urg_color or _ORANGE),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ])))
    story.append(Spacer(1, 3))
    story.append(Paragraph(f"<b>Action recommandée :</b> {action}", S_body))

    if safety.get("level") in ("warning","critical"):
        hex_s = "C0392B" if safety["level"] == "critical" else "E67E22"
        story.append(Paragraph(
            f'<font color="#{hex_s}"><b>⚠ {safety.get("message","")}</b></font>', S_warn))

    story.append(HRFlowable(width=pw, thickness=0.5, color=_MUTED, spaceBefore=4, spaceAfter=4))

    # ── Bilan biologique ─────────────────────────────────────────────────────
    story.append(Paragraph("Bilan Biologique Hépatique", S_h2))
    bio_rows = [["Paramètre", "Valeur"]]
    for k, v in bio.items():
        bio_rows.append([k.replace("_"," ").title(), str(v)])
    story.append(_htbl(bio_rows, [pw*0.38, pw*0.62]))
    story.append(Spacer(1, 5))

    # ── Probabilités ─────────────────────────────────────────────────────────
    story.append(Paragraph("Distribution des Probabilités Diagnostiques", S_h2))
    probs = result.get("probabilities", {})
    if probs:
        prob_sorted = sorted(probs.items(), key=lambda x: -x[1])[:10]
        pd = [["Pathologie", "Probabilité", "Barre"]]
        for cls, p in prob_sorted:
            bar = "█"*int(p*28) + "░"*(28-int(p*28))
            pd.append([cls, f"{p:.1%}", bar[:22]])
        story.append(_htbl(pd, [pw*0.52, pw*0.14, pw*0.34]))

    # ── Survie estimée ────────────────────────────────────────────────────────
    if surv_risk:
        story.append(Spacer(1, 4))
        story.append(Paragraph("Risque de Mortalité Estimé", S_h2))
        sd = [["Horizon", "Risque estimé"]]
        labels = {"1an":"À 1 an","3ans":"À 3 ans","5ans":"À 5 ans"}
        for k, v in surv_risk.items():
            sd.append([labels.get(k,k), f"{float(v):.1%}"])
        story.append(_htbl(sd, [pw*0.40, pw*0.60]))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — SCORES HÉPATOLOGIQUES
    # ══════════════════════════════════════════════════════════════════════════

    story.append(Paragraph("📊 Scores Hépatologiques de Référence", S_h1))
    story.append(HRFlowable(width=pw, thickness=1, color=_BROWN, spaceAfter=5))

    def _render_score(title: str, sd: dict, color=None):
        if not sd:
            return
        c = color or _BROWN
        story.append(Paragraph(title, _s("sh", fontSize=9, fontName="Helvetica-Bold",
                                          textColor=c, spaceBefore=5, spaceAfter=2)))
        skip = {"interpretation", "recommendation", "organ_failures", "major_features"}
        rows = []
        for k, v in sd.items():
            if k in skip or v is None:
                continue
            label = k.replace("_"," ").title()
            if isinstance(v, bool): val = "Oui" if v else "Non"
            elif isinstance(v, list): val = ", ".join(str(x) for x in v) if v else "—"
            elif isinstance(v, float): val = f"{v:.3f}" if v != int(v) else str(int(v))
            else: val = str(v)
            rows.append([label, val])
        if rows:
            story.append(_tbl(rows, [pw*0.45, pw*0.55], extra=[
                ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
                ("TEXTCOLOR",(0,0),(0,-1),_DARK),
            ]))
        # organ_failures / major_features
        for list_key in ("organ_failures","major_features"):
            lst = sd.get(list_key,[])
            if lst:
                story.append(Paragraph(
                    f"<b>{list_key.replace('_',' ').title()} :</b> " + " · ".join(lst), S_small))
        interp = sd.get("interpretation","")
        reco   = sd.get("recommendation","")
        if interp: story.append(Paragraph(f"<i>{interp}</i>", S_small))
        if reco:   story.append(Paragraph(f"<b>Recommandation :</b> {reco}", S_small))
        story.append(Spacer(1, 4))

    _render_score("Child-Pugh Score — Cirrhose hépatique",           scores.get("child_pugh",{}),  _BROWN)
    _render_score("MELD Score / MELD-Na — Transplantation",          scores.get("meld",{}),        _RED)
    _render_score("FIB-4 Index — Fibrose non invasive",              scores.get("fib4",{}),         _ORANGE)
    _render_score("APRI Score — Fibrose VHC/VHB",                    scores.get("apri",{}),         _ORANGE)
    _render_score("CAP Score — Stéatose hépatique (FibroScan)",      scores.get("cap",{}),          _ORANGE)
    _render_score("METAVIR — Fibrose + Activité histologique",       scores.get("metavir",{}),      _BROWN)
    _render_score("ALBI Grade — Fonction hépatique (CHC)",           scores.get("albi",{}),         _PURPLE)
    _render_score("BCLC Staging — Carcinome hépatocellulaire",       scores.get("bclc",{}),         _RED)
    _render_score("LI-RADS — Caractérisation nodule hépatique IRM",  scores.get("lirads",{}),       _CRIT)
    _render_score("CLIF-C ACLF — Insuffisance hépatique aiguë-sur-chronique",
                  scores.get("clif_aclf",{}), _CRIT)

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — FINDINGS CRITIQUES + EXPLAINABILITY
    # ══════════════════════════════════════════════════════════════════════════

    story.append(Paragraph("⚠ Findings Critiques & Explainability", S_h1))
    story.append(HRFlowable(width=pw, thickness=1, color=_BROWN, spaceAfter=5))

    story.append(Paragraph("Findings Hépatiques Critiques", S_h2))
    if critical:
        for f in critical:
            sev   = f.get("severity","INFO")
            hx    = f.get("color","#E67E22").replace("#","")
            mkr   = f.get("marker","—")
            val   = f.get("value","")
            det   = f.get("detail","")
            vs    = f" = {val}" if val and val != "—" else ""
            story.append(Paragraph(
                f'<font color="#{hx}"><b>[{sev}]</b></font> <b>{mkr}{vs}</b> — {det}',
                S_body))
        story.append(Spacer(1, 5))
    else:
        story.append(Paragraph(
            '<font color="#27AE60"><b>Aucun finding critique hépatique détecté.</b></font>',
            S_body))
        story.append(Spacer(1, 4))

    story.append(HRFlowable(width=pw, thickness=0.5, color=_MUTED, spaceAfter=4))

    # Urgence globale
    ug = scores.get("overall_urgency","—")
    sev_val = profile.get("severity", 0)
    story.append(Paragraph("Indicateurs de Gravité Globale", S_h2))
    grav_data = [["Indicateur", "Valeur", "Interprétation"]]
    grav_data.append(["Urgence clinique globale", ug,
                       "Critique / Urgente / Élevée / Modérée / Faible"])
    grav_data.append(["Score de sévérité IA", f"{float(sev_val):.1%}",
                       "Élevé ≥ 0.5 · Modéré 0.25–0.5 · Faible < 0.25"])
    story.append(_htbl(grav_data, [pw*0.35, pw*0.20, pw*0.45]))
    story.append(Spacer(1, 6))

    # Feature Importance
    story.append(Paragraph("🤖 Explainability — Importance des Variables (SHAP-inspired)", S_h2))
    if feat_imp:
        fi_sorted = sorted(feat_imp.items(), key=lambda x: -x[1])[:14]
        fi_data   = [["Variable", "Importance (%)", "Contribution"]]
        for feat, sc in fi_sorted:
            bar = "█"*int(sc/100*22) + "░"*(22-int(sc/100*22))
            fi_data.append([feat, f"{sc:.1f}%", bar[:20]])
        story.append(_htbl(fi_data, [pw*0.48, pw*0.17, pw*0.35]))
        story.append(Spacer(1, 4))

    top5 = result.get("explainability",{}).get("top_5_drivers",[])
    if top5:
        story.append(Paragraph(
            "<b>Top 5 variables décisives :</b> " +
            " · ".join(f"<b>{k}</b> ({v:.0f}%)" for k,v in top5), S_body))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — RECOMMANDATIONS + GUIDELINES + QR CODE
    # ══════════════════════════════════════════════════════════════════════════

    story.append(Paragraph("📋 Recommandations Cliniques & Guidelines", S_h1))
    story.append(HRFlowable(width=pw, thickness=1, color=_BROWN, spaceAfter=5))

    story.append(Paragraph("Recommandations Issues des Scores Cliniques", S_h2))
    reco_rows = [["Score", "Recommandation"]]
    for sk, sv in scores.items():
        if isinstance(sv, dict) and sv.get("recommendation"):
            reco_rows.append([sk.replace("_"," ").upper(), sv["recommendation"]])
    if len(reco_rows) > 1:
        story.append(_htbl(reco_rows, [pw*0.20, pw*0.80]))
    story.append(Spacer(1, 6))

    # Action principale
    story.append(Paragraph("Action Principale Recommandée", S_h2))
    story.append(Table([[
        Paragraph(f'<font color="#{urg_hex}"><b>{urgency.upper()}</b></font> — {action}',
                  _s("act", fontSize=9, fontName="Helvetica", textColor=_DARK))
    ]], colWidths=[pw], style=TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor(f"#{urg_hex}18")),
        ("BOX",(0,0),(-1,-1),1,urg_color or _BROWN),
        ("LEFTPADDING",(0,0),(-1,-1),10),("TOPPADDING",(0,0),(-1,-1),6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
    ])))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Références & Guidelines", S_h2))
    story.append(Paragraph(result.get("guidelines_ref",""), S_small))
    story.append(Spacer(1, 5))

    story.append(Paragraph("Sources Bibliographiques", S_h2))
    for src in [
        "EASL Clinical Practice Guidelines on HCV, HBV, Cirrhosis, NAFLD (2023)",
        "AASLD Practice Guidance for HCV, NAFLD (2023)",
        "BCLC 2022 — Reig et al., Journal of Hepatology 2022",
        "LI-RADS v2018 — American College of Radiology",
        "METAVIR Cooperative Study Group, Hepatology 1994",
        "FIB-4 — Sterling et al., Hepatology 2006",
        "APRI — Wai et al., Hepatology 2003",
        "ALBI Score — Johnson et al., JCO 2015",
        "CLIF-C ACLF — Moreau et al., Hepatology 2013",
        "CAP Score — de Lédinghen et al., J Hepatol 2012",
        "WHO HCV Treatment Guidelines 2023 · WHO HBV 2022",
    ]:
        story.append(Paragraph(f"• {src}", S_small))
    story.append(Spacer(1, 8))

    story.append(HRFlowable(width=pw, thickness=0.5, color=_MUTED, spaceAfter=4))
    story.append(Paragraph(
        "<b>⚠ Disclaimer médical :</b> Ce rapport est généré par KANÉA HepatoScan AI v1.0 "
        "à titre d'aide au diagnostic hépatologique. Il ne remplace pas l'expertise d'un "
        "hépatologue ou gastro-entérologue. Toute décision thérapeutique doit être validée "
        "en réunion de concertation pluridisciplinaire (RCP). "
        "Conforme aux recommandations EASL/AASLD 2023.",
        _s("disc", fontSize=7, textColor=_MUTED, fontName="Helvetica", spaceAfter=6, leading=10),
    ))

    qr_text = (
        f"KANEA-Hepato|ID:{report_id}|Patient:{patient_id}|"
        f"Diagnostic:{prediction}|Confiance:{confidence:.1%}|Urgence:{urgency}"
    )
    qr_img = _qr(qr_text, size=60)
    story.append(Table([[
        Paragraph(
            f"<b>KANÉA HepatoScan AI</b><br/>"
            f"Rapport ID : {report_id}<br/>"
            f"Généré le : {now_str}<br/>"
            f"Patient : {patient_id}<br/>"
            f"Examinateur : {examiner}",
            _s("sig", fontSize=8, fontName="Helvetica", textColor=_DARK),
        ),
        qr_img or Paragraph("QR non disponible", S_small),
    ]], colWidths=[pw*0.75, pw*0.25], style=TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#D5DCE8")),
        ("LEFTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ])))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def generate_hepato_report(
    result: dict[str, Any],
    output_path: str | None = None,
    **kwargs,
) -> bytes | None:
    """Interface publique — génère et optionnellement sauvegarde le rapport PDF."""
    pdf_bytes = build_hepato_pdf_report(result, **kwargs)
    if pdf_bytes and output_path:
        with open(output_path, "wb") as fh:
            fh.write(pdf_bytes)
    return pdf_bytes
