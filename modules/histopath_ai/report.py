"""
HistoPath AI — Rapport PDF anatomopathologique A4 multi-pages
=============================================================
Contient : informations patient · diagnostic IA · grade tumoral ·
stade pathologique TNM · scores IHC (ER/PR/HER2/Ki-67) · biomarqueurs critiques ·
feature importance (explainability) · recommandations guidelines · QR code.

Format : A4 · multi-pages · imprimable · compatible hôpital.
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

# ── Palette KANEA HistoPath ───────────────────────────────────────────────────
_PURPLE = colors.HexColor("#9B59B6") if _RL_OK else None
_RED    = colors.HexColor("#C0392B") if _RL_OK else None
_ORANGE = colors.HexColor("#E67E22") if _RL_OK else None
_GREEN  = colors.HexColor("#27AE60") if _RL_OK else None
_BLUE   = colors.HexColor("#2980B9") if _RL_OK else None
_DARK   = colors.HexColor("#1A2B3C") if _RL_OK else None
_MUTED  = colors.HexColor("#5E7A8A") if _RL_OK else None
_LIGHT  = colors.HexColor("#EEF2F6") if _RL_OK else None
_WHITE  = colors.white               if _RL_OK else None
_CRIT   = colors.HexColor("#922B21") if _RL_OK else None

_URG_RL = {
    "EXTRÊME": colors.HexColor("#922B21") if _RL_OK else None,
    "Critique": colors.HexColor("#C0392B") if _RL_OK else None,
    "ÉLEVÉ":   colors.HexColor("#E74C3C") if _RL_OK else None,
    "Élevée":  colors.HexColor("#E74C3C") if _RL_OK else None,
    "MODÉRÉ":  colors.HexColor("#E67E22") if _RL_OK else None,
    "Modérée": colors.HexColor("#E67E22") if _RL_OK else None,
    "INFO":    colors.HexColor("#2980B9") if _RL_OK else None,
    "Faible":  colors.HexColor("#27AE60") if _RL_OK else None,
}

_URG_HEX = {
    "EXTRÊME": "922B21", "Critique": "C0392B",
    "ÉLEVÉ": "E74C3C",   "Élevée": "E74C3C",
    "MODÉRÉ": "E67E22",  "Modérée": "E67E22",
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


def _badge(text: str, hex_color: str = "9B59B6") -> str:
    return (
        f'<font color="#{hex_color}"><b>{text}</b></font>'
    )


def _urgency_color(urgency: str) -> str:
    return _URG_HEX.get(urgency, "5E7A8A")


def _score_row(label: str, value: str, note: str = "") -> list:
    return [label, value, note]


# ═══════════════════════════════════════════════════════════════════════════════
# Fonction principale
# ═══════════════════════════════════════════════════════════════════════════════

def build_histopath_pdf_report(
    result: dict[str, Any],
    patient_id: str = "KANEA-HP",
    examiner: str = "KANÉA System",
    institution: str = "Anatomopathologie — KANÉA",
    patient_name: str = "",
    patient_age: str = "",
    patient_sex: str = "",
    sample_id: str = "",
    sample_type: str = "",
) -> bytes | None:
    """
    Génère un rapport PDF anatomopathologique A4 multi-pages.
    Retourne les bytes du PDF ou None si ReportLab non disponible.
    """
    if not _RL_OK:
        return None

    buf = io.BytesIO()
    W, H = A4
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"HistoPath AI — {patient_id}",
        author="KANÉA Medical AI Platform",
    )

    story = []
    pw = W - 36 * mm  # page width usable

    # ── Styles ────────────────────────────────────────────────────────────────
    S_title   = _s("title",   fontSize=18, textColor=_PURPLE, fontName="Helvetica-Bold",
                   alignment=TA_CENTER, spaceAfter=2)
    S_sub     = _s("sub",     fontSize=9,  textColor=_MUTED,  fontName="Helvetica",
                   alignment=TA_CENTER, spaceAfter=6)
    S_h1      = _s("h1",      fontSize=12, textColor=_DARK,   fontName="Helvetica-Bold",
                   spaceBefore=8, spaceAfter=4)
    S_h2      = _s("h2",      fontSize=10, textColor=_PURPLE, fontName="Helvetica-Bold",
                   spaceBefore=4, spaceAfter=3)
    S_body    = _s("body",    fontSize=8,  textColor=_DARK,   fontName="Helvetica",
                   spaceAfter=3, leading=12)
    S_small   = _s("small",   fontSize=7,  textColor=_MUTED,  fontName="Helvetica",
                   spaceAfter=2)
    S_center  = _s("center",  fontSize=8,  textColor=_DARK,   fontName="Helvetica",
                   alignment=TA_CENTER)
    S_warn    = _s("warn",    fontSize=8,  textColor=_RED,    fontName="Helvetica-Bold",
                   spaceAfter=3)

    # ── Helpers de tables ─────────────────────────────────────────────────────
    def _tbl(data, col_widths, style_cmds=None):
        cmds = [
            ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_LIGHT, _WHITE]),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#D5DCE8")),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ]
        if style_cmds:
            cmds.extend(style_cmds)
        return Table(data, colWidths=col_widths, style=TableStyle(cmds))

    def _header_tbl(data, col_widths):
        cmds = [
            ("BACKGROUND",  (0, 0), (-1, 0), _PURPLE),
            ("TEXTCOLOR",   (0, 0), (-1, 0), _WHITE),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT, _WHITE]),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#D5DCE8")),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ]
        return Table(data, colWidths=col_widths, style=TableStyle(cmds))

    now_str     = datetime.now().strftime("%d/%m/%Y à %H:%M")
    report_id   = str(uuid.uuid4())[:8].upper()
    cancer_type = result.get("cancer_type", "Histopathologie")
    prediction  = result.get("prediction", "—")
    confidence  = result.get("confidence", 0.0)
    urgency     = result.get("clinical_profile", {}).get("urgency", "—")
    action      = result.get("clinical_profile", {}).get("action", "—")
    surv5       = result.get("clinical_profile", {}).get("five_year_survival", "—")
    urg_color   = _URG_RL.get(urgency, _ORANGE)
    urg_hex     = _urgency_color(urgency)
    scores      = result.get("clinical_scores", {})
    findings    = result.get("critical_findings", [])
    feat_imp    = result.get("explainability", {}).get("feature_importance", {})
    ihc         = result.get("ihc_summary", {})
    safety      = result.get("clinical_safety", {})

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — EN-TÊTE + DIAGNOSTIC PRINCIPAL
    # ══════════════════════════════════════════════════════════════════════════

    # Logo / titre
    story.append(Paragraph("🔬 HistoPath AI — Rapport Anatomopathologique", S_title))
    story.append(Paragraph(
        f"KANÉA Medical AI Platform | {institution} | {now_str} | ID {report_id}",
        S_sub,
    ))
    story.append(HRFlowable(width=pw, thickness=2, color=_PURPLE, spaceAfter=6))

    # Informations patient
    story.append(Paragraph("Informations Patient & Échantillon", S_h1))
    pat_data = [
        ["ID Patient", patient_id or "—", "Préleveur", examiner],
        ["Nom / Prénom", patient_name or "—", "Âge", patient_age or "—"],
        ["Sexe", patient_sex or "—", "ID Échantillon", sample_id or report_id],
        ["Type de cancer", cancer_type, "Type tissu", sample_type or result.get("input_params", {}).get("tissue_type", "—")],
        ["Date rapport", now_str, "Version IA", result.get("module_version", "v1.0")],
    ]
    story.append(_tbl(
        pat_data,
        col_widths=[pw * 0.20, pw * 0.30, pw * 0.20, pw * 0.30],
        style_cmds=[
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (0, -1), _PURPLE),
            ("TEXTCOLOR", (2, 0), (2, -1), _PURPLE),
        ],
    ))
    story.append(Spacer(1, 6))

    # ── Bandeau diagnostic principal ─────────────────────────────────────────
    diag_color = _URG_RL.get(urgency, _ORANGE)
    diag_hex   = urg_hex

    diag_data = [[
        Paragraph(
            f'<font color="#{diag_hex}"><b>DIAGNOSTIC IA : {prediction}</b></font>',
            _s("d1", fontSize=12, fontName="Helvetica-Bold", textColor=_DARK,
               alignment=TA_LEFT),
        ),
        Paragraph(
            f'<b>Confiance : {confidence:.1%}</b><br/>'
            f'Urgence : <font color="#{diag_hex}"><b>{urgency}</b></font>',
            _s("d2", fontSize=9, fontName="Helvetica", textColor=_DARK, alignment=TA_RIGHT),
        ),
    ]]
    diag_tbl = Table(
        diag_data, colWidths=[pw * 0.70, pw * 0.30],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(f"#{diag_hex}22")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING",   (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
            ("BOX", (0, 0), (-1, -1), 1.5, diag_color or _ORANGE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]),
    )
    story.append(diag_tbl)
    story.append(Spacer(1, 4))

    # Action recommandée
    story.append(Paragraph(f"<b>Action recommandée :</b> {action}", S_body))
    story.append(Paragraph(f"<b>Survie à 5 ans estimée :</b> {surv5}", S_body))

    # ── Alerte sécurité ──────────────────────────────────────────────────────
    if safety.get("level") in ("warning", "critical"):
        sev_color = _RED if safety["level"] == "critical" else _ORANGE
        sev_hex   = "C0392B" if safety["level"] == "critical" else "E67E22"
        story.append(Paragraph(
            f'<font color="#{sev_hex}"><b>⚠ {safety.get("message", "")}</b></font>',
            S_warn,
        ))

    story.append(HRFlowable(width=pw, thickness=0.5, color=_MUTED, spaceBefore=6, spaceAfter=4))

    # ── Distribution des probabilités ────────────────────────────────────────
    story.append(Paragraph("Distribution des Probabilités Diagnostiques", S_h2))
    probs = result.get("probabilities", {})
    if probs:
        prob_sorted = sorted(probs.items(), key=lambda x: -x[1])
        prob_data   = [["Classe Diagnostique", "Probabilité", "Barre"]]
        for cls, p in prob_sorted:
            bar_w = int(p * 30)
            bar   = "█" * bar_w + "░" * (30 - bar_w)
            prob_data.append([cls, f"{p:.1%}", bar[:20]])
        story.append(_header_tbl(prob_data, col_widths=[pw * 0.50, pw * 0.15, pw * 0.35]))
    story.append(Spacer(1, 6))

    # ── Récidive multi-horizon ────────────────────────────────────────────────
    rec_risk = result.get("clinical_profile", {}).get("recurrence_risk", {})
    if rec_risk:
        story.append(Paragraph("Risque de Récidive Estimé (Score d'Agressivité)", S_h2))
        rec_data = [["Horizon", "Risque estimé"]]
        labels   = {"1an": "À 1 an", "3ans": "À 3 ans", "5ans": "À 5 ans", "10ans": "À 10 ans"}
        for k, v in rec_risk.items():
            rec_data.append([labels.get(k, k), f"{v:.1%}"])
        story.append(_header_tbl(rec_data, col_widths=[pw * 0.40, pw * 0.60]))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — SCORES CLINIQUES
    # ══════════════════════════════════════════════════════════════════════════

    story.append(Paragraph("🔬 Scores Anatomopathologiques", S_h1))
    story.append(HRFlowable(width=pw, thickness=1, color=_PURPLE, spaceAfter=5))

    def _render_score_section(title: str, score_dict: dict, color=None):
        if not score_dict:
            return
        story.append(Paragraph(title, _s("sh", fontSize=9, fontName="Helvetica-Bold",
                                          textColor=color or _PURPLE, spaceBefore=5, spaceAfter=2)))
        rows = []
        skip_keys = {"interpretation", "recommendation"}
        for k, v in score_dict.items():
            if k in skip_keys or v is None:
                continue
            label = k.replace("_", " ").title()
            if isinstance(v, bool):
                val = "Oui" if v else "Non"
            elif isinstance(v, list):
                val = ", ".join(str(x) for x in v) if v else "—"
            elif isinstance(v, float):
                val = f"{v:.4f}" if v != int(v) else str(int(v))
            else:
                val = str(v)
            rows.append([label, val])
        if rows:
            story.append(_tbl(
                rows, col_widths=[pw * 0.45, pw * 0.55],
                style_cmds=[("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                             ("TEXTCOLOR", (0, 0), (0, -1), _DARK)],
            ))
        # Interprétation
        interp = score_dict.get("interpretation", "")
        reco   = score_dict.get("recommendation", "")
        if interp:
            story.append(Paragraph(f"<i>Interprétation : {interp}</i>", S_small))
        if reco:
            story.append(Paragraph(f"<b>Recommandation :</b> {reco}", S_small))
        story.append(Spacer(1, 4))

    # TNM
    if "tnm" in scores:
        _render_score_section("Stade TNM Pathologique (AJCC 8e édition)", scores["tnm"], _BLUE)

    # Nottingham (sein)
    if "nottingham" in scores:
        _render_score_section("Nottingham Histologic Grade (Elston-Ellis)", scores["nottingham"], _PURPLE)

    # Ki-67
    if "ki67" in scores:
        _render_score_section("Ki-67 Index — Prolifération Cellulaire", scores["ki67"], _ORANGE)

    # HER2
    if "her2" in scores:
        _render_score_section("HER2 Score (ASCO/CAP 2018)", scores["her2"], _BLUE)

    # Récepteurs hormonaux
    if "hormone_receptors" in scores:
        _render_score_section("Récepteurs Hormonaux ER / PR (ASCO/CAP 2020)", scores["hormone_receptors"], _GREEN)

    # Sous-type moléculaire
    if "molecular_subtype" in scores:
        _render_score_section("Sous-type Moléculaire — St. Gallen 2021", scores["molecular_subtype"], _PURPLE)

    # Gleason / ISUP
    if "gleason" in scores:
        _render_score_section("Gleason Score + ISUP Grade Group", scores["gleason"], _RED)

    # Tumor Budding
    if "tumor_budding" in scores:
        _render_score_section("Tumor Budding Score (ITBCC 2016)", scores["tumor_budding"], _ORANGE)

    # CIN / Bethesda
    if "cin" in scores:
        _render_score_section("CIN Grading + Bethesda System 2014", scores["cin"], _RED)

    # Edmondson-Steiner (HCC)
    if "edmondson_steiner" in scores:
        _render_score_section("Edmondson-Steiner Grade (HCC)", scores["edmondson_steiner"], _ORANGE)

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — IHC + FINDINGS CRITIQUES + EXPLAINABILITY
    # ══════════════════════════════════════════════════════════════════════════

    story.append(Paragraph("🧫 Immunohistochimie & Findings Critiques", S_h1))
    story.append(HRFlowable(width=pw, thickness=1, color=_PURPLE, spaceAfter=5))

    # IHC Summary
    if ihc:
        story.append(Paragraph("Bilan Immunohistochimique", S_h2))
        ihc_rows = [["Marqueur", "Résultat"]]
        for k, v in ihc.items():
            if v and v != "Non évalué":
                ihc_rows.append([k.upper(), str(v)])
        story.append(_header_tbl(ihc_rows, col_widths=[pw * 0.35, pw * 0.65]))
        story.append(Spacer(1, 6))

    # Findings critiques
    if findings:
        story.append(Paragraph("Findings Anatomopathologiques Critiques", S_h2))
        for f in findings:
            sev     = f.get("severity", "INFO")
            hex_c   = f.get("color", "#E67E22").replace("#", "")
            finding = f.get("finding", "—")
            detail  = f.get("detail", "")
            story.append(Paragraph(
                f'<font color="#{hex_c}"><b>[{sev}]</b></font> <b>{finding}</b> — {detail}',
                S_body,
            ))
        story.append(Spacer(1, 6))
    else:
        story.append(Paragraph(
            '<font color="#27AE60"><b>Aucun finding critique détecté.</b></font>',
            S_body,
        ))
        story.append(Spacer(1, 4))

    story.append(HRFlowable(width=pw, thickness=0.5, color=_MUTED, spaceAfter=4))

    # ── Explainability — Feature Importance ──────────────────────────────────
    story.append(Paragraph("🤖 Explainability — Importance des Variables (SHAP-inspired)", S_h2))
    if feat_imp:
        fi_sorted = sorted(feat_imp.items(), key=lambda x: -x[1])[:12]
        fi_data   = [["Variable", "Importance (%)", "Contribution visuelle"]]
        for feature, score in fi_sorted:
            bar_len = int(score / 100 * 25)
            bar     = "█" * bar_len + "░" * (25 - bar_len)
            fi_data.append([feature, f"{score:.1f}%", bar[:20]])
        story.append(_header_tbl(fi_data, col_widths=[pw * 0.45, pw * 0.18, pw * 0.37]))
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
    story.append(HRFlowable(width=pw, thickness=1, color=_PURPLE, spaceAfter=5))

    # Recommandations par score
    story.append(Paragraph("Recommandations Issues des Scores Cliniques", S_h2))
    reco_data = [["Score", "Recommandation"]]
    for score_key, score_val in scores.items():
        if isinstance(score_val, dict) and score_val.get("recommendation"):
            label = score_key.replace("_", " ").title()
            reco_data.append([label, score_val["recommendation"]])
    if len(reco_data) > 1:
        story.append(_header_tbl(reco_data, col_widths=[pw * 0.22, pw * 0.78]))
    story.append(Spacer(1, 6))

    # Action principale
    story.append(Paragraph("Action Principale Recommandée", S_h2))
    action_data = [[
        Paragraph(
            f'<font color="#{urg_hex}"><b>{urgency.upper()}</b></font> — {action}',
            _s("act", fontSize=9, fontName="Helvetica", textColor=_DARK),
        )
    ]]
    story.append(Table(
        action_data,
        colWidths=[pw],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(f"#{urg_hex}18")),
            ("BOX", (0, 0), (-1, -1), 1, urg_color or _PURPLE),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]),
    ))
    story.append(Spacer(1, 8))

    # Références guidelines
    story.append(Paragraph("Références & Guidelines", S_h2))
    guidelines = result.get("guidelines_ref", "")
    story.append(Paragraph(guidelines, S_small))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Sources Bibliographiques", S_h2))
    sources = [
        "College of American Pathologists (CAP) Protocol 2023",
        "WHO Classification of Tumours, 5th Edition (2022)",
        "AJCC Cancer Staging Manual, 8th Edition (2017)",
        "ESMO Clinical Practice Guidelines 2023 (Breast, CRC, Prostate, Lung)",
        "ASCO/CAP HER2 Testing Guidelines (Wolff et al., JCO 2018)",
        "St. Gallen Expert Consensus 2021 (Molecular Subtypes)",
        "Elston & Ellis, Histopathology 1991 (Nottingham Grade)",
        "Epstein et al., Eur Urol 2016 (ISUP Grade Group)",
        "ITBCC 2016 — Lugli et al., Mod Pathol 2017 (Tumor Budding)",
        "Bethesda System 2014 (Cervical Cytology)",
        "Edmondson & Steiner, Cancer 1954 (HCC Grading)",
        "DESTINY-Breast04 — Modi et al., NEJM 2022 (HER2-Low, T-DXd)",
    ]
    for src in sources:
        story.append(Paragraph(f"• {src}", S_small))
    story.append(Spacer(1, 8))

    # ── Disclaimer médical ────────────────────────────────────────────────────
    story.append(HRFlowable(width=pw, thickness=0.5, color=_MUTED, spaceAfter=4))
    story.append(Paragraph(
        "<b>⚠ Disclaimer médical :</b> Ce rapport est généré par un système d'intelligence "
        "artificielle (KANÉA HistoPath AI v1.0) à titre d'aide au diagnostic. "
        "Il ne remplace en aucun cas l'expertise d'un médecin anatomopathologiste qualifié. "
        "Toute décision thérapeutique doit être validée en réunion de concertation pluridisciplinaire (RCP). "
        "Conforme aux recommandations ESMO/ASCO/NCCN/CAP 2023.",
        _s("disc", fontSize=7, textColor=_MUTED, fontName="Helvetica", spaceAfter=6, leading=10),
    ))

    # ── QR Code + signature ──────────────────────────────────────────────────
    qr_text = (
        f"KANEA-HistoPath|ID:{report_id}|Patient:{patient_id}|"
        f"Diagnostic:{prediction}|Confiance:{confidence:.1%}|Cancer:{cancer_type}"
    )
    qr_img = _qr(qr_text, size=60)

    sig_data = [[
        Paragraph(
            f"<b>KANÉA HistoPath AI</b><br/>"
            f"Rapport ID : {report_id}<br/>"
            f"Généré le : {now_str}<br/>"
            f"Patient : {patient_id}<br/>"
            f"Examinateur : {examiner}",
            _s("sig", fontSize=8, fontName="Helvetica", textColor=_DARK),
        ),
        qr_img or Paragraph("QR non disponible", S_small),
    ]]
    story.append(Table(
        sig_data,
        colWidths=[pw * 0.75, pw * 0.25],
        style=TableStyle([
            ("VALIGN",   (0, 0), (-1, -1), "TOP"),
            ("GRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#D5DCE8")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ]),
    ))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story)
    buf.seek(0)
    return buf.read()


def generate_histopath_report(
    result: dict[str, Any],
    output_path: str | None = None,
    **kwargs,
) -> bytes | None:
    """
    Interface publique pour générer et optionnellement sauvegarder le rapport PDF.
    """
    pdf_bytes = build_histopath_pdf_report(result, **kwargs)
    if pdf_bytes and output_path:
        with open(output_path, "wb") as fh:
            fh.write(pdf_bytes)
    return pdf_bytes
