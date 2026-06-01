"""
HematoVision AI — Rapport médical PDF A4 multi-pages
=====================================================
Contient : données patient, NFS, diagnostic IA, scores OMS/ELN/ISS/IPSS-R,
parasitémie, biomarqueurs critiques, alertes, explainability, recommandations.
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

# ── Palette ──────────────────────────────────────────────────────────────────
_CRIMSON  = colors.HexColor("#8B0000") if _RL_OK else None
_RED      = colors.HexColor("#C0392B") if _RL_OK else None
_ORANGE   = colors.HexColor("#E67E22") if _RL_OK else None
_GREEN    = colors.HexColor("#27AE60") if _RL_OK else None
_BLUE     = colors.HexColor("#2E4A8E") if _RL_OK else None
_TEAL     = colors.HexColor("#1A7A6E") if _RL_OK else None
_DARK     = colors.HexColor("#1C2833") if _RL_OK else None
_MUTED    = colors.HexColor("#566573") if _RL_OK else None
_LIGHT    = colors.HexColor("#EBF5FB") if _RL_OK else None
_WHITE    = colors.white               if _RL_OK else None
_PURPLE   = colors.HexColor("#6C3483") if _RL_OK else None

_URGENCY_RL = {
    "Critique": colors.HexColor("#C0392B") if _RL_OK else None,
    "Élevée":   colors.HexColor("#E74C3C") if _RL_OK else None,
    "Modérée":  colors.HexColor("#E67E22") if _RL_OK else None,
    "Faible":   colors.HexColor("#27AE60") if _RL_OK else None,
}
_URGENCY_HEX = {
    "Critique": "C0392B",
    "Élevée":   "E74C3C",
    "Modérée":  "E67E22",
    "Faible":   "27AE60",
}


def _s(name: str, **kw) -> Any:
    return ParagraphStyle(name, **kw)


def _qr_img(text: str, size: int = 55) -> Any | None:
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


def build_hemato_pdf_report(
    result: dict[str, Any],
    patient_id: str = "KANEA-AUTO",
    examiner: str = "KANÉA System",
    institution: str = "Hématologie — KANÉA",
) -> bytes | None:
    """Génère le rapport PDF HematoVision AI A4."""
    if not _RL_OK:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=14*mm, bottomMargin=14*mm,
                            leftMargin=17*mm, rightMargin=17*mm)
    W = A4[0] - 34*mm

    s_h1 = _s("H1", fontSize=14, fontName="Helvetica-Bold", textColor=_WHITE, leading=17)
    s_h2 = _s("H2", fontSize=10, fontName="Helvetica-Bold", textColor=_DARK, leading=13, spaceAfter=3)
    s_h3 = _s("H3", fontSize=9,  fontName="Helvetica-Bold", textColor=_TEAL, leading=11)
    s_b  = _s("B",  fontSize=8.5,fontName="Helvetica",      textColor=_DARK, leading=11)
    s_m  = _s("M",  fontSize=7.5,fontName="Helvetica",      textColor=_MUTED,leading=10)
    s_r  = _s("R",  fontSize=7.5,fontName="Helvetica",      textColor=_MUTED,alignment=TA_RIGHT,leading=10)

    story: list[Any] = []

    # ── Données principales ──────────────────────────────────────────────────
    pred        = result.get("prediction", "—")
    conf        = result.get("confidence", 0.0)
    urgency     = result.get("clinical_profile", {}).get("urgency", "—")
    color_rl    = _URGENCY_RL.get(urgency, _MUTED)
    urg_hex     = _URGENCY_HEX.get(urgency, "566573")
    action      = result.get("recommended_action", "—")
    request_id  = result.get("request_id", str(uuid.uuid4()))[:16]
    date_str    = datetime.now().strftime("%d/%m/%Y à %H:%M")
    model_ver   = result.get("model_version", "v2.0")

    # ── HEADER ───────────────────────────────────────────────────────────────
    qr = _qr_img(f"KANEA-HEMATO-{request_id}", 55)
    header_left = Paragraph(
        f'<font size="15"><b>KANÉA — HematoVision AI {model_ver}</b></font><br/>'
        f'<font size="8.5">Module hématologie · Cytologie sanguine · Classification OMS/ELN/ICC</font>',
        s_h1,
    )
    hd_data = [[header_left, qr if qr else Paragraph(request_id[:8], s_r)]]
    hd_t = Table(hd_data, colWidths=[W * 0.75, W * 0.25])
    hd_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), _DARK),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(hd_t)
    story.append(Spacer(1, 4*mm))

    # ── 1. RENSEIGNEMENTS ────────────────────────────────────────────────────
    story.append(Paragraph("1. Renseignements du dossier", s_h2))
    info = [
        ["N° dossier", patient_id, "Date", date_str],
        ["Examinateur", examiner,   "Institution", institution],
        ["Modèle IA", result.get("model_architecture","—")[:45], "Version", model_ver],
        ["Analyse (ms)", str(result.get("processing_ms",0)), "ID requête", request_id],
    ]
    info_t = Table(info, colWidths=[W*0.15, W*0.35, W*0.14, W*0.36])
    info_t.setStyle(TableStyle([
        ("FONTNAME",      (0,0),(-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("FONTNAME",      (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTNAME",      (2,0),(2,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (0,0),(0,-1),  _MUTED),
        ("TEXTCOLOR",     (2,0),(2,-1),  _MUTED),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [_LIGHT, _WHITE]),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#D5E8F0")),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
    ]))
    story.append(info_t)
    story.append(Spacer(1, 4*mm))

    # ── 2. DIAGNOSTIC IA PRINCIPAL ───────────────────────────────────────────
    story.append(Paragraph("2. Diagnostic IA principal", s_h2))
    pred_data = [
        [Paragraph("<b>Diagnostic</b>", s_m),
         Paragraph(f"<b><font size='12' color='#{urg_hex}'>{pred}</font></b>", s_b),
         Paragraph("<b>Confiance</b>", s_m),
         Paragraph(f"<b><font size='11' color='#{urg_hex}'>{conf:.1%}</font></b>", s_b)],
        [Paragraph("<b>Urgence</b>", s_m),
         Paragraph(f"<b><font color='#{urg_hex}'>{urgency}</font></b>", s_b),
         Paragraph("<b>Référence</b>", s_m),
         Paragraph("<font size='7.5'>WHO 2022 · ICC 2022 · ELN 2022</font>", s_m)],
        [Paragraph("<b>Action</b>", s_m),
         Paragraph(f"<font size='8'>{action[:100]}</font>", s_b),
         Paragraph("", s_m), Paragraph("", s_m)],
    ]
    pred_t = Table(pred_data, colWidths=[W*0.14, W*0.36, W*0.14, W*0.36])
    pred_t.setStyle(TableStyle([
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [colors.HexColor("#F7F9FC"), _WHITE]),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#D5E8F0")),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("LINEBEFORE",    (0,0),(0,-1),  4, color_rl),
        ("SPAN",          (1,2),(3,2)),
    ]))
    story.append(pred_t)
    story.append(Spacer(1, 4*mm))

    # ── 3. NFS — RÉSUMÉ HÉMOGRAMME ──────────────────────────────────────────
    nfs = result.get("nfs_summary", {})
    story.append(Paragraph("3. NFS — Numération Formule Sanguine", s_h2))
    nfs_interp = nfs.get("interpretation", "—")
    nfs_reco   = nfs.get("recommendation", "—")
    nfs_sev    = nfs.get("anemia_severity", "—")
    nfs_flags  = nfs.get("critical_flags", [])

    story.append(Paragraph(
        f"<b>Interprétation NFS :</b> {nfs_interp}  |  "
        f"<b>Sévérité anémie :</b> {nfs_sev}", s_b,
    ))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(f"<b>Recommandation :</b> {nfs_reco}", s_b))
    story.append(Spacer(1, 2*mm))

    if nfs_flags:
        flag_rows = [["Paramètre", "Valeur", "Statut", "Détail"]]
        for f in nfs_flags[:8]:
            flag_rows.append([
                f.get("param", "—"),
                str(f.get("value", "—")),
                f.get("status", "—"),
                f.get("detail", "—")[:60],
            ])
        flag_t = Table(flag_rows, colWidths=[W*0.15, W*0.15, W*0.18, W*0.52])
        flag_t.setStyle(TableStyle([
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 7.5),
            ("BACKGROUND",    (0,0),(-1,0), _TEAL),
            ("TEXTCOLOR",     (0,0),(-1,0), _WHITE),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [_LIGHT, _WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#D5E8F0")),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ]))
        story.append(flag_t)
    story.append(Spacer(1, 4*mm))

    # ── 4. SCORES CLINIQUES ──────────────────────────────────────────────────
    story.append(Paragraph("4. Scores cliniques hématologiques", s_h2))
    scores = result.get("clinical_scores", {})
    sc_rows = [["Score", "Résultat", "Catégorie / Stade", "Recommandation"]]

    eln = scores.get("eln_aml", {})
    if eln:
        sc_rows.append(["ELN AML 2022",
                         eln.get("risk_category","—"),
                         f"RC attendue : {eln.get('cr_rate','—')} · OS 3 ans : {eln.get('os_3yr','—')}",
                         eln.get("recommendation","—")[:60]])

    ann = scores.get("ann_arbor", {})
    if ann:
        sc_rows.append(["Ann Arbor / Lugano",
                         ann.get("stage","—"),
                         f"SG 5 ans : {ann.get('five_year_os','—')}",
                         ann.get("recommendation","—")[:60]])

    ipi = scores.get("ipi", {})
    if ipi:
        sc_rows.append(["IPI Score",
                         f"{ipi.get('score','—')}/5 — {ipi.get('risk_group','—')}",
                         f"RC : {ipi.get('cr_rate','—')} · OS 5 ans : {ipi.get('five_year_os','—')}",
                         ipi.get("recommendation","—")[:60]])

    iss = scores.get("iss_riss", {})
    if iss:
        sc_rows.append(["ISS / R-ISS",
                         f"{iss.get('iss_label','—')} / {iss.get('r_iss_label','—')}",
                         f"OS : {iss.get('five_year_os_iss','—')} (ISS) · {iss.get('five_year_os_riss','—')} (R-ISS)",
                         iss.get("recommendation","—")[:60]])

    ipss = scores.get("ipss_r", {})
    if ipss:
        sc_rows.append(["IPSS-R",
                         f"Score {ipss.get('score','—')} — {ipss.get('risk_category','—')}",
                         f"Survie médiane : {ipss.get('median_survival_yr','—')}",
                         ipss.get("recommendation","—")[:60]])

    dic = scores.get("isth_dic", {})
    if dic:
        sc_rows.append(["ISTH DIC",
                         f"Score {dic.get('score','—')} — {'CIVD manifeste' if dic.get('overt_dic') else 'Non manifeste'}",
                         f"Mortalité : {dic.get('mortality_risk','—')}",
                         dic.get("recommendation","—")[:60]])

    if len(sc_rows) > 1:
        sc_t = Table(sc_rows, colWidths=[W*0.16, W*0.22, W*0.30, W*0.32])
        sc_t.setStyle(TableStyle([
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 7.5),
            ("BACKGROUND",    (0,0),(-1,0), _BLUE),
            ("TEXTCOLOR",     (0,0),(-1,0), _WHITE),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [_LIGHT, _WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#D5E8F0")),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ]))
        story.append(sc_t)
    else:
        story.append(Paragraph("Aucun score spécialisé calculé (contexte général).", s_m))
    story.append(Spacer(1, 4*mm))

    # ── 5. PARASITÉMIE PALUDISME ─────────────────────────────────────────────
    malaria = scores.get("malaria", {})
    if malaria:
        story.append(Paragraph("5. Analyse parasitologique — Paludisme", s_h2))
        mal_data = [
            ["Parasitémie", f"{malaria.get('parasitemia_pct',0):.2f}%",
             "Espèce", malaria.get("plasmodium_species","—")],
            ["Grade", malaria.get("grade","—"),
             "Sévérité", malaria.get("severity","—")],
            ["Urgence", malaria.get("treatment_urgency","—"),
             "Recommandation", malaria.get("recommendation","—")[:55]],
        ]
        mal_t = Table(mal_data, colWidths=[W*0.13, W*0.20, W*0.13, W*0.54])
        mal_t.setStyle(TableStyle([
            ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
            ("FONTNAME",      (2,0),(2,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 8),
            ("TEXTCOLOR",     (0,0),(0,-1),  _MUTED),
            ("TEXTCOLOR",     (2,0),(2,-1),  _MUTED),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [colors.HexColor("#FEF9E7"), _WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#FDEBD0")),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ]))
        story.append(mal_t)
        story.append(Spacer(1, 4*mm))

    # ── 6. PROBABILITÉS DIAGNOSTIQUES ───────────────────────────────────────
    story.append(Paragraph("6. Probabilités diagnostiques", s_h2))
    probs = result.get("probabilities", {})
    if probs:
        top_probs = sorted(probs.items(), key=lambda x: -x[1])[:8]
        prob_rows = [["Diagnostic", "Probabilité", "Barre"]]
        for dx, p_val in top_probs:
            bar = "█" * int(p_val * 40) + "░" * (40 - int(p_val * 40))
            prob_rows.append([dx[:50], f"{p_val:.1%}", bar])
        prob_t = Table(prob_rows, colWidths=[W*0.43, W*0.11, W*0.46])
        prob_t.setStyle(TableStyle([
            ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTNAME",      (2,1),(2,-1),  "Courier"),
            ("FONTSIZE",      (0,0),(-1,-1), 7.5),
            ("BACKGROUND",    (0,0),(-1,0),  _PURPLE),
            ("TEXTCOLOR",     (0,0),(-1,0),  _WHITE),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [_LIGHT, _WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#D5E8F0")),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ]))
        story.append(prob_t)
    story.append(Spacer(1, 4*mm))

    # ── 7. ALERTES CLINIQUES ─────────────────────────────────────────────────
    alerts = result.get("alerts", [])
    if alerts:
        story.append(Paragraph("7. Alertes cliniques", s_h2))
        alert_bg    = {"CRITICAL": "#FDECEA", "HIGH": "#FEF0E7", "WARNING": "#FEF9E7", "INFO": "#EBF4FD"}
        alert_border= {"CRITICAL": "#922B21",  "HIGH": "#E74C3C", "WARNING": "#E67E22", "INFO": "#2E86DE"}
        for a in alerts[:8]:
            lvl = a.get("level", "INFO")
            alert_data = [[
                Paragraph(
                    f"<b>{a.get('icon','')} [{lvl}] {a.get('title','—')}</b><br/>"
                    f"<font size='7.5'>{a.get('message','—')}</font><br/>"
                    f"<font size='7'><i>Action : {a.get('clinical_action','—')[:90]}</i></font>",
                    _s(f"AL{lvl}", fontSize=8, fontName="Helvetica", textColor=_DARK, leading=11),
                )
            ]]
            at = Table(alert_data, colWidths=[W])
            at.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), colors.HexColor(alert_bg.get(lvl,"#EBF4FD"))),
                ("LINEBEFORE",   (0,0),(-1,-1), 4, colors.HexColor(alert_border.get(lvl,"#2E86DE"))),
                ("TOPPADDING",   (0,0),(-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                ("LEFTPADDING",  (0,0),(-1,-1), 8),
            ]))
            story.append(at)
            story.append(Spacer(1, 2*mm))
        story.append(Spacer(1, 2*mm))

    # ── 8. EXPLAINABILITY ────────────────────────────────────────────────────
    expl = result.get("explainability", {})
    top3 = expl.get("top_3_drivers", [])
    fi   = expl.get("feature_importance", {})
    if top3 or fi:
        story.append(Paragraph("8. Explainability IA — Facteurs déterminants", s_h2))
        if top3:
            top3_str = " · ".join(f"<b>{k}</b> ({v:.0f}%)" for k, v in top3)
            story.append(Paragraph(f"Top 3 facteurs : {top3_str}", s_b))
            story.append(Spacer(1, 2*mm))
        if fi:
            fi_rows = [["Variable biologique", "Importance", "Contribution"]]
            for feat, imp in sorted(fi.items(), key=lambda x: -x[1])[:10]:
                bar = "█" * int(imp / 100 * 40) + "░" * (40 - int(imp / 100 * 40))
                fi_rows.append([feat, f"{imp:.1f}%", bar])
            fi_t = Table(fi_rows, colWidths=[W*0.30, W*0.12, W*0.58])
            fi_t.setStyle(TableStyle([
                ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
                ("FONTNAME",      (2,1),(2,-1),  "Courier"),
                ("FONTSIZE",      (0,0),(-1,-1), 7.5),
                ("BACKGROUND",    (0,0),(-1,0),  _TEAL),
                ("TEXTCOLOR",     (0,0),(-1,0),  _WHITE),
                ("ROWBACKGROUNDS",(0,1),(-1,-1), [_LIGHT, _WHITE]),
                ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#D5E8F0")),
                ("TOPPADDING",    (0,0),(-1,-1), 3),
                ("BOTTOMPADDING", (0,0),(-1,-1), 3),
                ("LEFTPADDING",   (0,0),(-1,-1), 5),
            ]))
            story.append(fi_t)
        story.append(Spacer(1, 4*mm))

    # ── 9. SÉCURITÉ ET RECOMMANDATION FINALE ────────────────────────────────
    story.append(Paragraph("9. Recommandation clinique et sécurité IA", s_h2))
    safety = result.get("clinical_safety", {})
    safety_msg = safety.get("message", "")
    if safety_msg:
        safety_lvl = safety.get("level","ok")
        saf_color  = {"critical":"#FDECEA","warning":"#FEF0E7","ok":"#EAFAF1"}.get(safety_lvl,"#EAFAF1")
        saf_border = {"critical":"#C0392B","warning":"#E67E22","ok":"#27AE60"}.get(safety_lvl,"#27AE60")
        saf_data   = [[Paragraph(f"<b>Sécurité IA :</b> {safety_msg}", s_b)]]
        saf_t = Table(saf_data, colWidths=[W])
        saf_t.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,-1), colors.HexColor(saf_color)),
            ("LINEBEFORE",  (0,0),(-1,-1), 4, colors.HexColor(saf_border)),
            ("TOPPADDING", (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING", (0,0),(-1,-1), 8),
        ]))
        story.append(saf_t)
        story.append(Spacer(1, 2*mm))

    story.append(Paragraph(
        f"<b>Action recommandée :</b> {action}", s_b,
    ))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f"<b>Référence :</b> {result.get('guidelines_ref','—')}", s_m,
    ))
    story.append(Spacer(1, 4*mm))

    # ── FOOTER ───────────────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#D5E8F0")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "⚠ Outil d'aide à la décision médicale — ne remplace pas le jugement clinique d'un hématologue. "
        "Toute décision thérapeutique doit être validée par un médecin spécialiste. "
        "HematoVision AI v2.0 — KANÉA 2026 — WHO 2022 · ICC 2022 · ELN 2022 · ASH/EHA Guidelines.",
        s_m,
    ))

    doc.build(story)
    return buf.getvalue()


def build_hemato_html_report(
    result: dict[str, Any],
    patient_id: str = "KANEA",
) -> str:
    """Rapport HTML léger HematoVision AI (fallback sans ReportLab)."""
    pred     = result.get("prediction", "—")
    conf     = result.get("confidence", 0.0)
    urgency  = result.get("clinical_profile", {}).get("urgency", "—")
    action   = result.get("recommended_action", "—")
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    urg_hex  = "#" + _URGENCY_HEX.get(urgency, "566573")
    nfs      = result.get("nfs_summary", {})
    alerts   = result.get("alerts", [])
    scores   = result.get("clinical_scores", {})
    probs    = result.get("probabilities", {})

    probs_rows = "".join(
        f"<tr><td>{dx[:50]}</td><td>{p:.1%}</td></tr>"
        for dx, p in sorted(probs.items(), key=lambda x: -x[1])[:6]
    )
    alert_rows = "".join(
        f'<div style="border-left:4px solid {a.get("clinical_action","#999") and "#C0392B"};'
        f'padding:5px 10px;margin:4px 0;background:#fef9f0;font-size:10px;">'
        f'<b>{a.get("icon","")} [{a.get("level","")}] {a.get("title","")}</b><br/>'
        f'{a.get("message","")}</div>'
        for a in alerts[:6]
    )
    nfs_flags_html = "".join(
        f"<tr><td>{f.get('param','')}</td><td>{f.get('value','')}</td><td>{f.get('status','')}</td></tr>"
        for f in nfs.get("critical_flags", [])[:6]
    )

    # Scores block
    eln  = scores.get("eln_aml", {})
    iss  = scores.get("iss_riss", {})
    ipss = scores.get("ipss_r", {})
    ann  = scores.get("ann_arbor", {})
    mal  = scores.get("malaria", {})
    scores_html = ""
    if eln:
        scores_html += f"<tr><th>ELN AML 2022</th><td>{eln.get('risk_category','—')}</td><td>{eln.get('recommendation','—')[:70]}</td></tr>"
    if iss:
        scores_html += f"<tr><th>ISS / R-ISS</th><td>{iss.get('iss_label','—')}</td><td>{iss.get('recommendation','—')[:70]}</td></tr>"
    if ipss:
        scores_html += f"<tr><th>IPSS-R</th><td>Score {ipss.get('score','—')} — {ipss.get('risk_category','—')}</td><td>{ipss.get('recommendation','—')[:70]}</td></tr>"
    if ann:
        scores_html += f"<tr><th>Ann Arbor</th><td>{ann.get('stage','—')}</td><td>{ann.get('recommendation','—')[:70]}</td></tr>"
    if mal:
        scores_html += f"<tr><th>Paludisme</th><td>{mal.get('grade','—')}</td><td>{mal.get('recommendation','—')[:70]}</td></tr>"

    return f"""<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8">
<title>HematoVision AI — {patient_id}</title>
<style>
body{{font-family:Arial,sans-serif;font-size:11px;color:#1C2833;margin:0;padding:12px}}
h1{{background:#1C2833;color:#fff;padding:8px 14px;font-size:13px;margin:0}}
h2{{font-size:10px;color:#1A7A6E;text-transform:uppercase;margin:10px 0 4px;border-bottom:1px solid #D5E8F0;padding-bottom:2px}}
table{{width:100%;border-collapse:collapse;margin-bottom:6px}}
th{{background:#2E4A8E;color:#fff;padding:3px 6px;font-size:9px;text-align:left}}
td{{padding:3px 6px;border:1px solid #D5E8F0;font-size:9px}}
tr:nth-child(even){{background:#EBF5FB}}
.disc{{background:#FFF8E1;border-left:4px solid #F39C12;padding:6px 10px;font-size:8px;margin-top:10px}}
.info{{background:#F0F4F8;padding:5px 10px;margin:0;font-size:9px}}
</style></head><body>
<h1>KANÉA — HematoVision AI v2.0 · Rapport hématologique</h1>
<p class="info">Dossier : {patient_id} · {date_str} · ID : {result.get("request_id","—")[:16]}</p>
<h2>Diagnostic IA principal</h2>
<p><b style="color:{urg_hex};font-size:13px">{pred}</b>
&nbsp;&nbsp; Confiance : <b>{conf:.1%}</b> &nbsp;&nbsp; Urgence : <b style="color:{urg_hex}">{urgency}</b></p>
<p><b>Action :</b> {action}</p>
<h2>NFS — Résumé hémogramme</h2>
<p>{nfs.get("interpretation","—")}</p>
{('<table><tr><th>Paramètre</th><th>Valeur</th><th>Statut</th></tr>' + nfs_flags_html + '</table>') if nfs_flags_html else ''}
<h2>Probabilités diagnostiques</h2>
<table><tr><th>Diagnostic</th><th>Probabilité</th></tr>{probs_rows}</table>
{('<h2>Scores cliniques</h2><table><tr><th>Score</th><th>Résultat</th><th>Recommandation</th></tr>' + scores_html + '</table>') if scores_html else ''}
<h2>Alertes cliniques</h2>{alert_rows if alert_rows else '<p>Aucune alerte critique.</p>'}
<div class="disc">⚠ Outil d'aide à la décision médicale — validation médicale obligatoire.
HematoVision AI v2.0 — KANÉA 2026 — WHO 2022 · ICC 2022 · ELN 2022 · ASH/EHA.</div>
</body></html>"""
