"""
CardioSense AI — Rapport PDF cardiologique A4 multi-pages
==========================================================
Contient : informations patient · diagnostic IA · analyse ECG · biomarqueurs cardiaques ·
scores (SCORE2, Framingham, GRACE, TIMI, CHA₂DS₂-VASc, HAS-BLED, NYHA, QTc) ·
risque de mortalité · explainability SHAP · recommandations ESC/AHA · QR code.
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

# ── Palette CardioSense (rouge cardiaque) ─────────────────────────────────────
_RED    = colors.HexColor("#C0392B") if _RL_OK else None
_CRIT   = colors.HexColor("#922B21") if _RL_OK else None
_ORANGE = colors.HexColor("#E67E22") if _RL_OK else None
_GREEN  = colors.HexColor("#27AE60") if _RL_OK else None
_BLUE   = colors.HexColor("#2980B9") if _RL_OK else None
_PURPLE = colors.HexColor("#8E44AD") if _RL_OK else None
_DARK   = colors.HexColor("#1A2B3C") if _RL_OK else None
_MUTED  = colors.HexColor("#5E7A8A") if _RL_OK else None
_LIGHT  = colors.HexColor("#EEF2F6") if _RL_OK else None
_WHITE  = colors.white               if _RL_OK else None

_URG_RL = {
    "EXTRÊME": colors.HexColor("#922B21") if _RL_OK else None,
    "Critique": colors.HexColor("#C0392B") if _RL_OK else None,
    "Élevée":   colors.HexColor("#E74C3C") if _RL_OK else None,
    "Modérée":  colors.HexColor("#E67E22") if _RL_OK else None,
    "Faible":   colors.HexColor("#27AE60") if _RL_OK else None,
    "INFO":     colors.HexColor("#2980B9") if _RL_OK else None,
}
_URG_HEX = {
    "EXTRÊME": "922B21", "Critique": "C0392B",
    "Élevée": "E74C3C",  "Modérée": "E67E22",
    "Faible": "27AE60",  "INFO": "2980B9",
}
_SEV_HEX = {
    "EXTRÊME": "922B21", "CRITIQUE": "C0392B",
    "ÉLEVÉ": "E74C3C",   "MODÉRÉ": "E67E22",
    "INFO": "2980B9",    "Faible": "27AE60",
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


def build_cardio_pdf_report(
    result: dict[str, Any],
    patient_id: str = "KANEA-CS",
    examiner: str = "KANÉA System",
    institution: str = "Cardiologie — KANÉA",
    patient_name: str = "",
    patient_age: str = "",
    patient_sex: str = "",
    clinical_context: str = "",
) -> bytes | None:
    """
    Génère un rapport PDF cardiologique A4 multi-pages.
    """
    if not _RL_OK:
        return None

    buf = io.BytesIO()
    W, H = A4
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm,  bottomMargin=16 * mm,
        title=f"CardioSense AI — {patient_id}",
        author="KANÉA Medical AI Platform",
    )
    story = []
    pw = W - 36 * mm

    # ── Styles ────────────────────────────────────────────────────────────────
    S_title  = _s("title", fontSize=18, textColor=_RED,  fontName="Helvetica-Bold",
                  alignment=TA_CENTER, spaceAfter=2)
    S_sub    = _s("sub",   fontSize=9,  textColor=_MUTED, fontName="Helvetica",
                  alignment=TA_CENTER, spaceAfter=6)
    S_h1     = _s("h1",   fontSize=12, textColor=_DARK,  fontName="Helvetica-Bold",
                  spaceBefore=8, spaceAfter=4)
    S_h2     = _s("h2",   fontSize=10, textColor=_RED,   fontName="Helvetica-Bold",
                  spaceBefore=4, spaceAfter=3)
    S_body   = _s("body", fontSize=8,  textColor=_DARK,  fontName="Helvetica",
                  spaceAfter=3, leading=12)
    S_small  = _s("small",fontSize=7,  textColor=_MUTED, fontName="Helvetica",
                  spaceAfter=2)
    S_warn   = _s("warn", fontSize=8,  textColor=_RED,   fontName="Helvetica-Bold",
                  spaceAfter=3)

    def _tbl(data, col_widths, extra_cmds=None):
        cmds = [
            ("FONTNAME",       (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",       (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_LIGHT, _WHITE]),
            ("GRID",           (0, 0), (-1, -1), 0.3, colors.HexColor("#D5DCE8")),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",    (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 5),
            ("TOPPADDING",     (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
        ]
        if extra_cmds:
            cmds.extend(extra_cmds)
        return Table(data, colWidths=col_widths, style=TableStyle(cmds))

    def _htbl(data, col_widths):
        cmds = [
            ("BACKGROUND",    (0, 0), (-1, 0), _RED),
            ("TEXTCOLOR",     (0, 0), (-1, 0), _WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_LIGHT, _WHITE]),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#D5DCE8")),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        return Table(data, colWidths=col_widths, style=TableStyle(cmds))

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
    markers    = result.get("critical_markers", [])
    feat_imp   = result.get("explainability", {}).get("feature_importance", {})
    ecg_sum    = result.get("ecg_summary", {})
    safety     = result.get("clinical_safety", {})
    mort_risk  = profile.get("mortality_risk", {})

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — EN-TÊTE + DIAGNOSTIC + ECG
    # ══════════════════════════════════════════════════════════════════════════

    story.append(Paragraph("❤️ CardioSense AI — Rapport Cardiologique", S_title))
    story.append(Paragraph(
        f"KANÉA Medical AI Platform | {institution} | {now_str} | ID {report_id}",
        S_sub,
    ))
    story.append(HRFlowable(width=pw, thickness=2, color=_RED, spaceAfter=6))

    # Informations patient
    story.append(Paragraph("Informations Patient", S_h1))
    pat_data = [
        ["ID Patient", patient_id or "—", "Examinateur", examiner],
        ["Nom / Prénom", patient_name or "—", "Âge", patient_age or "—"],
        ["Sexe", patient_sex or "—", "Contexte clinique", clinical_context or "—"],
        ["Date rapport", now_str, "Version IA", result.get("model_version", "v2.0")],
    ]
    story.append(_tbl(
        pat_data, col_widths=[pw * 0.20, pw * 0.30, pw * 0.20, pw * 0.30],
        extra_cmds=[
            ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",   (2, 0), (2, -1), "Helvetica-Bold"),
            ("TEXTCOLOR",  (0, 0), (0, -1), _RED),
            ("TEXTCOLOR",  (2, 0), (2, -1), _RED),
        ],
    ))
    story.append(Spacer(1, 6))

    # Bandeau diagnostic
    diag_data = [[
        Paragraph(
            f'<font color="#{urg_hex}"><b>DIAGNOSTIC : {prediction}</b></font>',
            _s("d1", fontSize=12, fontName="Helvetica-Bold", textColor=_DARK),
        ),
        Paragraph(
            f'<b>Confiance : {confidence:.1%}</b><br/>'
            f'Urgence : <font color="#{urg_hex}"><b>{urgency}</b></font>',
            _s("d2", fontSize=9, fontName="Helvetica", textColor=_DARK, alignment=TA_RIGHT),
        ),
    ]]
    story.append(Table(
        diag_data, colWidths=[pw * 0.70, pw * 0.30],
        style=TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor(f"#{urg_hex}20")),
            ("BOX",          (0, 0), (-1, -1), 1.5, urg_color or _RED),
            ("LEFTPADDING",  (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING",   (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ]),
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Action recommandée :</b> {action}", S_body))

    if safety.get("level") in ("warning", "critical"):
        sev_hex = "C0392B" if safety["level"] == "critical" else "E67E22"
        story.append(Paragraph(
            f'<font color="#{sev_hex}"><b>⚠ {safety.get("message", "")}</b></font>',
            S_warn,
        ))

    story.append(HRFlowable(width=pw, thickness=0.5, color=_MUTED, spaceBefore=4, spaceAfter=4))

    # ── Résumé ECG ────────────────────────────────────────────────────────────
    story.append(Paragraph("Analyse ECG 12 Dérivations", S_h2))
    ecg_data = [["Paramètre ECG", "Valeur", "Statut"]]
    ecg_rows = [
        ("Rythme",          ecg_sum.get("rhythm", "—"),                          ""),
        ("Fréquence cardiaque", f"{ecg_sum.get('heart_rate', '—')} bpm",        ""),
        ("Intervalle PR",   ecg_sum.get("pr_interval", "—"),                     "Normal < 200 ms"),
        ("Durée QRS",       ecg_sum.get("qrs_duration", "—"),                    "Normal < 120 ms"),
        ("Intervalle QT",   ecg_sum.get("qt_interval", "—"),                     ""),
        ("QTc (Bazett)",    f"{ecg_sum.get('qtc', '—')} ms",                     "Normal < 440 ms (H) / 460 ms (F)"),
        ("Urgence ECG",     ecg_sum.get("ecg_urgency", "—"),                     ""),
    ]
    for label, val, note in ecg_rows:
        ecg_data.append([label, str(val), note])
    story.append(_htbl(ecg_data, col_widths=[pw * 0.35, pw * 0.30, pw * 0.35]))

    # Anomalies ECG
    ecg_findings = ecg_sum.get("ecg_findings", [])
    if ecg_findings:
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "<b>Anomalies ECG détectées :</b> " + " · ".join(ecg_findings[:6]),
            S_body,
        ))
    story.append(Spacer(1, 6))

    # Probabilités diagnostiques
    story.append(Paragraph("Distribution des Probabilités Diagnostiques", S_h2))
    probs = result.get("probabilities", {})
    if probs:
        prob_sorted = sorted(probs.items(), key=lambda x: -x[1])
        prob_data   = [["Diagnostic", "Probabilité", "Barre"]]
        for cls, p in prob_sorted:
            bar = "█" * int(p * 25) + "░" * (25 - int(p * 25))
            prob_data.append([cls, f"{p:.1%}", bar[:20]])
        story.append(_htbl(prob_data, col_widths=[pw * 0.50, pw * 0.15, pw * 0.35]))

    # Risque mortalité
    if mort_risk:
        story.append(Spacer(1, 4))
        story.append(Paragraph("Risque de Mortalité Estimé", S_h2))
        mort_data = [["Horizon", "Risque estimé"]]
        labels = {"30j": "À 30 jours", "1an": "À 1 an", "5ans": "À 5 ans"}
        for k, v in mort_risk.items():
            mort_data.append([labels.get(k, k), f"{float(v):.1%}"])
        story.append(_htbl(mort_data, col_widths=[pw * 0.40, pw * 0.60]))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — SCORES CLINIQUES CARDIOVASCULAIRES
    # ══════════════════════════════════════════════════════════════════════════

    story.append(Paragraph("📊 Scores Cliniques Cardiovasculaires", S_h1))
    story.append(HRFlowable(width=pw, thickness=1, color=_RED, spaceAfter=5))

    def _render_score(title: str, score_dict: dict, color=None):
        if not score_dict:
            return
        c = color or _RED
        story.append(Paragraph(title, _s("sh", fontSize=9, fontName="Helvetica-Bold",
                                          textColor=c, spaceBefore=5, spaceAfter=2)))
        skip = {"interpretation", "recommendation", "components", "modifiable_factors",
                "findings", "abnormalities"}
        rows = []
        for k, v in score_dict.items():
            if k in skip or v is None:
                continue
            label = k.replace("_", " ").title()
            if isinstance(v, bool):
                val = "Oui" if v else "Non"
            elif isinstance(v, list):
                val = ", ".join(str(x) for x in v) if v else "—"
            elif isinstance(v, float):
                val = f"{v:.2f}" if v != int(v) else str(int(v))
            else:
                val = str(v)
            rows.append([label, val])
        if rows:
            story.append(_tbl(
                rows, col_widths=[pw * 0.48, pw * 0.52],
                extra_cmds=[("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                             ("TEXTCOLOR", (0, 0), (0, -1), _DARK)],
            ))
        # Components dict (CHA2DS2-VASc, HAS-BLED)
        if "components" in score_dict and score_dict["components"]:
            comp_rows = [["Composant", "Points"]]
            for ck, cv in score_dict["components"].items():
                comp_rows.append([ck, str(cv)])
            story.append(_htbl(comp_rows, col_widths=[pw * 0.75, pw * 0.25]))
        interp = score_dict.get("interpretation", "")
        reco   = score_dict.get("recommendation", "")
        if interp:
            story.append(Paragraph(f"<i>{interp}</i>", S_small))
        if reco:
            story.append(Paragraph(f"<b>Recommandation :</b> {reco}", S_small))
        story.append(Spacer(1, 4))

    _render_score("SCORE2 — Risque CV à 10 ans (ESC 2021)", scores.get("score2", {}), _RED)
    _render_score("Framingham Risk Score — Risque coronarien à 10 ans", scores.get("framingham", {}), _ORANGE)
    _render_score("GRACE Score — Stratification SCA", scores.get("grace", {}), _CRIT)
    _render_score("TIMI NSTEMI/UA Score", scores.get("timi_nstemi", {}), _RED)
    _render_score("TIMI STEMI Score", scores.get("timi_stemi", {}), _CRIT)
    _render_score("CHA₂DS₂-VASc — Risque AVC en FA", scores.get("cha2ds2_vasc", {}), _PURPLE)
    _render_score("HAS-BLED — Risque hémorragique", scores.get("has_bled", {}), _ORANGE)
    _render_score("NYHA — Insuffisance cardiaque", scores.get("nyha", {}), _BLUE)
    _render_score("QTc — Correction intervalle QT", scores.get("qtc", {}), _RED)

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — BIOMARQUEURS + EXPLAINABILITY
    # ══════════════════════════════════════════════════════════════════════════

    story.append(Paragraph("🩸 Biomarqueurs Cardiaques & Explainability", S_h1))
    story.append(HRFlowable(width=pw, thickness=1, color=_RED, spaceAfter=5))

    # Biomarqueurs critiques
    story.append(Paragraph("Biomarqueurs Cardiaques Critiques", S_h2))
    if markers:
        for m in markers:
            sev     = m.get("severity", "INFO")
            hex_c   = m.get("color", "#E67E22").replace("#", "")
            marker  = m.get("marker", "—")
            value   = m.get("value", "")
            detail  = m.get("detail", "")
            val_str = f" = {value}" if value and value != "—" else ""
            story.append(Paragraph(
                f'<font color="#{hex_c}"><b>[{sev}]</b></font> '
                f'<b>{marker}{val_str}</b> — {detail}',
                S_body,
            ))
        story.append(Spacer(1, 6))
    else:
        story.append(Paragraph(
            '<font color="#27AE60"><b>Aucun biomarqueur cardiaque critique détecté.</b></font>',
            S_body,
        ))
        story.append(Spacer(1, 4))

    story.append(HRFlowable(width=pw, thickness=0.5, color=_MUTED, spaceAfter=4))

    # Scores risque rapide
    story.append(Paragraph("Indicateurs de Risque Rapide", S_h2))
    cv_risk   = profile.get("cv_risk_score", 0)
    urg_score = profile.get("urgency_score", 0)
    risk_data = [["Indicateur", "Valeur", "Interprétation"]]
    risk_data.append(["Score d'urgence immédiate", f"{float(urg_score):.1%}",
                       "Critique ≥ 0.5 · Élevé 0.3–0.5 · Modéré < 0.3"])
    risk_data.append(["Risque CV long terme", f"{float(cv_risk):.1%}",
                       "Élevé ≥ 0.4 · Modéré 0.2–0.4 · Faible < 0.2"])
    story.append(_htbl(risk_data, col_widths=[pw * 0.35, pw * 0.20, pw * 0.45]))
    story.append(Spacer(1, 6))

    # Feature Importance
    story.append(Paragraph("🤖 Explainability — Feature Importance (SHAP-inspired)", S_h2))
    if feat_imp:
        fi_sorted = sorted(feat_imp.items(), key=lambda x: -x[1])[:14]
        fi_data   = [["Variable", "Importance (%)", "Contribution"]]
        for feature, score in fi_sorted:
            bar = "█" * int(score / 100 * 20) + "░" * (20 - int(score / 100 * 20))
            fi_data.append([feature, f"{score:.1f}%", bar[:20]])
        story.append(_htbl(fi_data, col_widths=[pw * 0.48, pw * 0.17, pw * 0.35]))
        story.append(Spacer(1, 4))

    top5 = result.get("explainability", {}).get("top_5_drivers", [])
    if top5:
        story.append(Paragraph(
            "<b>Top 5 variables décisives :</b> " +
            " · ".join(f"<b>{k}</b> ({v:.0f}%)" for k, v in top5),
            S_body,
        ))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — RECOMMANDATIONS + GUIDELINES + QR CODE
    # ══════════════════════════════════════════════════════════════════════════

    story.append(Paragraph("📋 Recommandations Cliniques & Guidelines", S_h1))
    story.append(HRFlowable(width=pw, thickness=1, color=_RED, spaceAfter=5))

    # Recommandations par score
    story.append(Paragraph("Recommandations Issues des Scores", S_h2))
    reco_rows = [["Score", "Recommandation"]]
    for sk, sv in scores.items():
        if isinstance(sv, dict) and sv.get("recommendation"):
            reco_rows.append([sk.replace("_", " ").upper(), sv["recommendation"]])
    if len(reco_rows) > 1:
        story.append(_htbl(reco_rows, col_widths=[pw * 0.20, pw * 0.80]))
    story.append(Spacer(1, 6))

    # Action principale
    story.append(Paragraph("Action Principale Recommandée", S_h2))
    act_data = [[
        Paragraph(
            f'<font color="#{urg_hex}"><b>{urgency.upper()}</b></font> — {action}',
            _s("act", fontSize=9, fontName="Helvetica", textColor=_DARK),
        )
    ]]
    story.append(Table(
        act_data, colWidths=[pw],
        style=TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor(f"#{urg_hex}18")),
            ("BOX",         (0, 0), (-1, -1), 1, urg_color or _RED),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING",  (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ]),
    ))
    story.append(Spacer(1, 8))

    # Références guidelines
    story.append(Paragraph("Références & Guidelines", S_h2))
    story.append(Paragraph(result.get("guidelines_ref", ""), S_small))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Sources Bibliographiques", S_h2))
    sources = [
        "ESC Guidelines for the management of ACS without persistent ST-segment elevation (2020)",
        "ESC Guidelines on Atrial Fibrillation (2020) — Hindricks et al., Eur Heart J",
        "ESC Guidelines on Heart Failure (2021) — McDonagh et al., Eur Heart J",
        "GRACE 2.0 ACS Risk Calculator — Fox et al., BMJ 2006",
        "TIMI Risk Score for UA/NSTEMI — Antman et al., JAMA 2000",
        "TIMI Risk Score for STEMI — Morrow et al., Circulation 2000",
        "Framingham Risk Score — D'Agostino et al., Circulation 2008",
        "SCORE2 — Visseren et al., Eur Heart J 2021",
        "CHA₂DS₂-VASc — Lip et al., Chest 2010",
        "HAS-BLED — Pisters et al., Chest 2010",
        "NYHA Functional Classification (1994)",
        "Bazett HC (1920) · Fridericia LS (1920) · Hodges M (1983) — QTc formulas",
    ]
    for src in sources:
        story.append(Paragraph(f"• {src}", S_small))
    story.append(Spacer(1, 8))

    # Disclaimer
    story.append(HRFlowable(width=pw, thickness=0.5, color=_MUTED, spaceAfter=4))
    story.append(Paragraph(
        "<b>⚠ Disclaimer médical :</b> Ce rapport est généré par KANÉA CardioSense AI v2.0 "
        "à titre d'aide au diagnostic. Il ne remplace pas l'expertise d'un cardiologue. "
        "En cas d'urgence cardiaque, appeler le SAMU (15) ou les services d'urgence locaux. "
        "Toute décision thérapeutique doit être validée par un médecin qualifié. "
        "Conforme aux recommandations ESC/AHA/ACC 2021–2022.",
        _s("disc", fontSize=7, textColor=_MUTED, fontName="Helvetica", spaceAfter=6, leading=10),
    ))

    # Signature + QR
    qr_text = (
        f"KANEA-Cardio|ID:{report_id}|Patient:{patient_id}|"
        f"Diagnostic:{prediction}|Confiance:{confidence:.1%}|Urgence:{urgency}"
    )
    qr_img = _qr(qr_text, size=60)
    sig_data = [[
        Paragraph(
            f"<b>KANÉA CardioSense AI</b><br/>"
            f"Rapport ID : {report_id}<br/>"
            f"Généré le : {now_str}<br/>"
            f"Patient : {patient_id}<br/>"
            f"Examinateur : {examiner}",
            _s("sig", fontSize=8, fontName="Helvetica", textColor=_DARK),
        ),
        qr_img or Paragraph("QR non disponible", S_small),
    ]]
    story.append(Table(
        sig_data, colWidths=[pw * 0.75, pw * 0.25],
        style=TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#D5DCE8")),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]),
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def generate_cardio_report(
    result: dict[str, Any],
    output_path: str | None = None,
    **kwargs,
) -> bytes | None:
    """Interface publique pour générer et optionnellement sauvegarder le rapport PDF."""
    pdf_bytes = build_cardio_pdf_report(result, **kwargs)
    if pdf_bytes and output_path:
        with open(output_path, "wb") as fh:
            fh.write(pdf_bytes)
    return pdf_bytes
