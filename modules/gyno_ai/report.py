"""
GynoCare AI — Rapport médical PDF A4 multi-pages
=================================================
Contient : données patiente, diagnostic IA, FIGO staging, scores ESGO/O-RADS/rASRM,
marqueurs tumoraux, ROMA score, alertes cliniques, explainability, recommandations.
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
        HRFlowable,
    )
    from reportlab.lib.enums import TA_RIGHT
    _RL_OK = True
except ImportError:
    _RL_OK = False

try:
    import qrcode
    _QR_OK = True
except ImportError:
    _QR_OK = False

# ── Palette ──────────────────────────────────────────────────────────────────
_ROSE    = colors.HexColor("#9B2D6F") if _RL_OK else None
_RED     = colors.HexColor("#C0392B") if _RL_OK else None
_ORANGE  = colors.HexColor("#E67E22") if _RL_OK else None
_GREEN   = colors.HexColor("#1E8449") if _RL_OK else None
_BLUE    = colors.HexColor("#1A5276") if _RL_OK else None
_TEAL    = colors.HexColor("#0E7373") if _RL_OK else None
_DARK    = colors.HexColor("#1C2833") if _RL_OK else None
_MUTED   = colors.HexColor("#566573") if _RL_OK else None
_LIGHT   = colors.HexColor("#FDFEFE") if _RL_OK else None
_LILAC   = colors.HexColor("#F4ECF7") if _RL_OK else None
_WHITE   = colors.white               if _RL_OK else None

_URGENCY_RL = {
    "Critique": colors.HexColor("#C0392B") if _RL_OK else None,
    "Élevée":   colors.HexColor("#E74C3C") if _RL_OK else None,
    "Modérée":  colors.HexColor("#E67E22") if _RL_OK else None,
    "Faible":   colors.HexColor("#27AE60") if _RL_OK else None,
}
_URGENCY_HEX = {
    "Critique": "C0392B", "Élevée": "E74C3C",
    "Modérée":  "E67E22", "Faible": "27AE60",
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


def build_gyno_pdf_report(
    result: dict[str, Any],
    patient_id: str = "KANEA-AUTO",
    examiner: str = "KANÉA System",
    institution: str = "Gynécologie-Oncologie — KANÉA",
) -> bytes | None:
    """Génère le rapport PDF GynoCare AI A4."""
    if not _RL_OK:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=14*mm, bottomMargin=14*mm,
                            leftMargin=17*mm, rightMargin=17*mm)
    W = A4[0] - 34*mm

    s_h1 = _s("H1", fontSize=14, fontName="Helvetica-Bold", textColor=_WHITE, leading=17)
    s_h2 = _s("H2", fontSize=10, fontName="Helvetica-Bold", textColor=_DARK, leading=13, spaceAfter=3)
    s_b  = _s("B",  fontSize=8.5,fontName="Helvetica",      textColor=_DARK, leading=11)
    s_m  = _s("M",  fontSize=7.5,fontName="Helvetica",      textColor=_MUTED,leading=10)
    s_r  = _s("R",  fontSize=7.5,fontName="Helvetica",      textColor=_MUTED,alignment=TA_RIGHT,leading=10)

    story: list[Any] = []

    pred       = result.get("prediction", "—")
    conf       = result.get("confidence", 0.0)
    urgency    = result.get("clinical_profile", {}).get("urgency", "—")
    color_rl   = _URGENCY_RL.get(urgency, _MUTED)
    urg_hex    = _URGENCY_HEX.get(urgency, "566573")
    action     = result.get("recommended_action", "—")
    request_id = result.get("request_id", str(uuid.uuid4()))[:16]
    date_str   = datetime.now().strftime("%d/%m/%Y à %H:%M")
    model_ver  = result.get("model_version", "v2.0")

    # ── HEADER ───────────────────────────────────────────────────────────────
    qr = _qr_img(f"KANEA-GYNO-{request_id}", 55)
    hd_t = Table(
        [[Paragraph(
            f'<font size="15"><b>KANÉA — GynoCare AI {model_ver}</b></font><br/>'
            f'<font size="8.5">Module gynécologie · Oncologie gynécologique · FIGO/ESGO/NCCN</font>',
            s_h1,
        ), qr if qr else Paragraph(request_id[:8], s_r)]],
        colWidths=[W*0.75, W*0.25],
    )
    hd_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), _ROSE),
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
    info_t = Table([
        ["N° dossier", patient_id,  "Date", date_str],
        ["Examinatrice", examiner,  "Institution", institution],
        ["Modèle IA", result.get("model_architecture","—")[:45], "Version", model_ver],
        ["Analyse (ms)", str(result.get("processing_ms",0)), "ID requête", request_id],
    ], colWidths=[W*0.15, W*0.35, W*0.14, W*0.36])
    info_t.setStyle(TableStyle([
        ("FONTNAME",      (0,0),(-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("FONTNAME",      (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTNAME",      (2,0),(2,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (0,0),(0,-1),  _MUTED),
        ("TEXTCOLOR",     (2,0),(2,-1),  _MUTED),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [_LILAC, _WHITE]),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#E8D5F0")),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
    ]))
    story.append(info_t)
    story.append(Spacer(1, 4*mm))

    # ── 2. DIAGNOSTIC IA PRINCIPAL ───────────────────────────────────────────
    story.append(Paragraph("2. Diagnostic IA principal", s_h2))
    pred_t = Table([
        [Paragraph("<b>Diagnostic</b>", s_m),
         Paragraph(f"<b><font size='12' color='#{urg_hex}'>{pred}</font></b>", s_b),
         Paragraph("<b>Confiance</b>", s_m),
         Paragraph(f"<b><font size='11' color='#{urg_hex}'>{conf:.1%}</font></b>", s_b)],
        [Paragraph("<b>Urgence</b>", s_m),
         Paragraph(f"<b><font color='#{urg_hex}'>{urgency}</font></b>", s_b),
         Paragraph("<b>Référence</b>", s_m),
         Paragraph("<font size='7.5'>FIGO 2018/2023 · ESGO 2020</font>", s_m)],
        [Paragraph("<b>Action</b>", s_m),
         Paragraph(f"<font size='8'>{action[:100]}</font>", s_b),
         Paragraph("", s_m), Paragraph("", s_m)],
    ], colWidths=[W*0.14, W*0.36, W*0.14, W*0.36])
    pred_t.setStyle(TableStyle([
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [colors.HexColor("#FDF8FF"), _WHITE]),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#E8D5F0")),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("LINEBEFORE",    (0,0),(0,-1),  4, color_rl),
        ("SPAN",          (1,2),(3,2)),
    ]))
    story.append(pred_t)
    story.append(Spacer(1, 4*mm))

    # ── 3. MARQUEURS TUMORAUX ────────────────────────────────────────────────
    tm = result.get("tumor_markers_summary", {})
    roma = result.get("roma_summary", {})
    story.append(Paragraph("3. Marqueurs tumoraux gynécologiques", s_h2))
    tm_flags = tm.get("flags", [])
    tm_interp = tm.get("interpretation", "—")
    story.append(Paragraph(f"<b>Interprétation :</b> {tm_interp}", s_b))
    story.append(Spacer(1, 2*mm))

    if tm_flags:
        flag_rows = [["Marqueur", "Valeur", "Statut", "Détail"]]
        for f in tm_flags[:6]:
            flag_rows.append([
                f.get("marker","—"), str(f.get("value","—")),
                f.get("status","—"), f.get("detail","—")[:60],
            ])
        fl_t = Table(flag_rows, colWidths=[W*0.12, W*0.15, W*0.18, W*0.55])
        fl_t.setStyle(TableStyle([
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 7.5),
            ("BACKGROUND",    (0,0),(-1,0), _ROSE),
            ("TEXTCOLOR",     (0,0),(-1,0), _WHITE),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [_LILAC, _WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#E8D5F0")),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ]))
        story.append(fl_t)
        story.append(Spacer(1, 2*mm))

    if roma:
        roma_pct = roma.get("score_pct", 0)
        roma_hr  = roma.get("high_risk", False)
        roma_hex = "C0392B" if roma_hr else "27AE60"
        story.append(Paragraph(
            f"<b>ROMA Score :</b> <font color='#{roma_hex}'><b>{roma_pct:.1f}%</b></font> — "
            f"{'⚠ RISQUE ÉLEVÉ' if roma_hr else '✓ Risque faible'}",
            s_b,
        ))
        story.append(Spacer(1, 1*mm))
        story.append(Paragraph(roma.get("recommendation","—"), s_m))
    story.append(Spacer(1, 4*mm))

    # ── 4. SCORES CLINIQUES FIGO ─────────────────────────────────────────────
    story.append(Paragraph("4. Stadification FIGO et scores cliniques", s_h2))
    scores = result.get("clinical_scores", {})
    sc_rows = [["Score / Classification", "Résultat / Stade", "Pronostic", "Recommandation"]]

    figo_cx = scores.get("figo_cervix", {})
    if figo_cx:
        sc_rows.append(["FIGO Col 2018",
                         f"{figo_cx.get('figo_stage','—')} — {figo_cx.get('substage','—')}",
                         f"SG 5 ans : {figo_cx.get('five_year_os','—')}",
                         figo_cx.get("treatment","—")[:60]])

    figo_end = scores.get("figo_endometrium", {})
    if figo_end:
        sc_rows.append(["FIGO Endomètre 2023",
                         f"{figo_end.get('figo_stage','—')} — ESGO : {figo_end.get('esgo_risk_group','—')}",
                         f"SG 5 ans : {figo_end.get('five_year_os','—')}",
                         figo_end.get("adjuvant_treatment","—")[:60]])

    figo_ov = scores.get("figo_ovary", {})
    if figo_ov:
        sc_rows.append(["FIGO Ovaire 2014",
                         f"{figo_ov.get('figo_stage','—')} — {figo_ov.get('substage','—')}",
                         f"SG 5 ans : {figo_ov.get('five_year_os','—')} · {figo_ov.get('brca_status','—')}",
                         figo_ov.get("treatment","—")[:60]])

    orads = scores.get("orads", {})
    if orads:
        sc_rows.append(["O-RADS",
                         f"{orads.get('category','—')}",
                         f"Malignité : {orads.get('malignancy_risk','—')}",
                         orads.get("recommendation","—")[:60]])

    endo_sc = scores.get("endometriosis", {})
    if endo_sc:
        sc_rows.append(["rASRM Endométriose",
                         f"Stade {endo_sc.get('stage','—')} ({endo_sc.get('stage_label','—')}) — Score {endo_sc.get('score','—')}",
                         endo_sc.get("fertility_impact","—")[:50],
                         endo_sc.get("recommendation","—")[:60]])

    sopk = scores.get("sopk", {})
    if sopk:
        sc_rows.append(["Rotterdam SOPK",
                         sopk.get("phenotype","—")[:50],
                         sopk.get("metabolic_risk","—")[:50],
                         sopk.get("recommendation","—")[:60]])

    if len(sc_rows) > 1:
        sc_t = Table(sc_rows, colWidths=[W*0.20, W*0.25, W*0.25, W*0.30])
        sc_t.setStyle(TableStyle([
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 7.5),
            ("BACKGROUND",    (0,0),(-1,0), _BLUE),
            ("TEXTCOLOR",     (0,0),(-1,0), _WHITE),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [_LILAC, _WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#E8D5F0")),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ]))
        story.append(sc_t)
    else:
        story.append(Paragraph("Aucun score spécialisé calculé (contexte général).", s_m))
    story.append(Spacer(1, 4*mm))

    # ── 5. PROBABILITÉS DIAGNOSTIQUES ───────────────────────────────────────
    story.append(Paragraph("5. Probabilités diagnostiques (15 classes)", s_h2))
    probs = result.get("probabilities", {})
    if probs:
        prob_rows = [["Diagnostic", "Probabilité", "Barre"]]
        for dx, pv in sorted(probs.items(), key=lambda x: -x[1])[:8]:
            bar = "█" * int(pv*40) + "░" * (40 - int(pv*40))
            prob_rows.append([dx[:52], f"{pv:.1%}", bar])
        pr_t = Table(prob_rows, colWidths=[W*0.45, W*0.10, W*0.45])
        pr_t.setStyle(TableStyle([
            ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTNAME",      (2,1),(2,-1),  "Courier"),
            ("FONTSIZE",      (0,0),(-1,-1), 7.5),
            ("BACKGROUND",    (0,0),(-1,0),  _ROSE),
            ("TEXTCOLOR",     (0,0),(-1,0),  _WHITE),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [_LILAC, _WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#E8D5F0")),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ]))
        story.append(pr_t)
    story.append(Spacer(1, 4*mm))

    # ── 6. ALERTES CLINIQUES ─────────────────────────────────────────────────
    alerts = result.get("alerts", [])
    if alerts:
        story.append(Paragraph("6. Alertes cliniques", s_h2))
        alert_bg     = {"CRITICAL":"#FDECEA","HIGH":"#FEF0E7","WARNING":"#FEF9E7","INFO":"#EBF4FD"}
        alert_border = {"CRITICAL":"#922B21","HIGH":"#E74C3C","WARNING":"#E67E22","INFO":"#2E86DE"}
        for a in alerts[:8]:
            lvl = a.get("level","INFO")
            at = Table([[
                Paragraph(
                    f"<b>{a.get('icon','')} [{lvl}] {a.get('title','—')}</b><br/>"
                    f"<font size='7.5'>{a.get('message','—')}</font><br/>"
                    f"<font size='7'><i>Action : {a.get('clinical_action','—')[:90]}</i></font>",
                    _s(f"AL{lvl}", fontSize=8, fontName="Helvetica", textColor=_DARK, leading=11),
                )
            ]], colWidths=[W])
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

    # ── 7. EXPLAINABILITY ────────────────────────────────────────────────────
    expl = result.get("explainability", {})
    top3 = expl.get("top_3_drivers", [])
    fi   = expl.get("feature_importance", {})
    if top3 or fi:
        story.append(Paragraph("7. Explainability IA — Facteurs déterminants", s_h2))
        if top3:
            t3 = " · ".join(f"<b>{k}</b> ({v:.0f}%)" for k, v in top3)
            story.append(Paragraph(f"Top 3 facteurs : {t3}", s_b))
            story.append(Spacer(1, 2*mm))
        if fi:
            fi_rows = [["Biomarqueur / Paramètre", "Importance", "Contribution"]]
            for feat, imp in sorted(fi.items(), key=lambda x: -x[1])[:10]:
                bar = "█" * int(imp/100*40) + "░" * (40 - int(imp/100*40))
                fi_rows.append([feat, f"{imp:.1f}%", bar])
            fi_t = Table(fi_rows, colWidths=[W*0.32, W*0.12, W*0.56])
            fi_t.setStyle(TableStyle([
                ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
                ("FONTNAME",      (2,1),(2,-1),  "Courier"),
                ("FONTSIZE",      (0,0),(-1,-1), 7.5),
                ("BACKGROUND",    (0,0),(-1,0),  _TEAL),
                ("TEXTCOLOR",     (0,0),(-1,0),  _WHITE),
                ("ROWBACKGROUNDS",(0,1),(-1,-1), [_LILAC, _WHITE]),
                ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#E8D5F0")),
                ("TOPPADDING",    (0,0),(-1,-1), 3),
                ("BOTTOMPADDING", (0,0),(-1,-1), 3),
                ("LEFTPADDING",   (0,0),(-1,-1), 5),
            ]))
            story.append(fi_t)
        story.append(Spacer(1, 4*mm))

    # ── 8. SÉCURITÉ ET RECOMMANDATION FINALE ────────────────────────────────
    story.append(Paragraph("8. Recommandation clinique et sécurité IA", s_h2))
    safety = result.get("clinical_safety", {})
    safety_msg = safety.get("message", "")
    if safety_msg:
        lvl = safety.get("level","ok")
        saf_bg  = {"critical":"#FDECEA","warning":"#FEF0E7","ok":"#EAFAF1"}.get(lvl,"#EAFAF1")
        saf_brd = {"critical":"#C0392B","warning":"#E67E22","ok":"#27AE60"}.get(lvl,"#27AE60")
        saf_t = Table([[Paragraph(f"<b>Sécurité IA :</b> {safety_msg}", s_b)]], colWidths=[W])
        saf_t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), colors.HexColor(saf_bg)),
            ("LINEBEFORE",   (0,0),(-1,-1), 4, colors.HexColor(saf_brd)),
            ("TOPPADDING",   (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
            ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ]))
        story.append(saf_t)
        story.append(Spacer(1, 2*mm))

    story.append(Paragraph(f"<b>Action recommandée :</b> {action}", s_b))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f"<b>Références :</b> {result.get('guidelines_ref','—')}", s_m,
    ))
    story.append(Spacer(1, 4*mm))

    # ── FOOTER ───────────────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#E8D5F0")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "⚠ Outil d'aide à la décision médicale — ne remplace pas le jugement clinique d'un gynécologue-oncologue. "
        "Toute décision thérapeutique doit être validée par un médecin spécialiste. "
        "GynoCare AI v2.0 — KANÉA 2026 — FIGO 2018/2023 · ESGO 2020 · NCCN 2023.",
        s_m,
    ))

    doc.build(story)
    return buf.getvalue()


def build_gyno_html_report(
    result: dict[str, Any],
    patient_id: str = "KANEA",
) -> str:
    """Rapport HTML léger GynoCare AI (fallback sans ReportLab)."""
    pred    = result.get("prediction", "—")
    conf    = result.get("confidence", 0.0)
    urgency = result.get("clinical_profile", {}).get("urgency", "—")
    action  = result.get("recommended_action", "—")
    date_str= datetime.now().strftime("%d/%m/%Y %H:%M")
    urg_hex = "#" + _URGENCY_HEX.get(urgency, "566573")
    tm      = result.get("tumor_markers_summary", {})
    roma    = result.get("roma_summary", {})
    alerts  = result.get("alerts", [])
    scores  = result.get("clinical_scores", {})
    probs   = result.get("probabilities", {})

    probs_rows = "".join(
        f"<tr><td>{dx[:52]}</td><td>{pv:.1%}</td></tr>"
        for dx, pv in sorted(probs.items(), key=lambda x: -x[1])[:7]
    )
    alert_rows = "".join(
        f'<div style="border-left:4px solid #C0392B;padding:5px 10px;'
        f'margin:4px 0;background:#fef9f0;font-size:10px;">'
        f'<b>{a.get("icon","")} [{a.get("level","")}] {a.get("title","")}</b><br/>'
        f'{a.get("message","")}</div>'
        for a in alerts[:6]
    )
    tm_flags_html = "".join(
        f"<tr><td>{f.get('marker','')}</td><td>{f.get('value','')}</td><td>{f.get('status','')}</td></tr>"
        for f in tm.get("flags", [])[:5]
    )
    sc_html = ""
    for key, label in [("figo_cervix","FIGO Col"),("figo_endometrium","FIGO Endomètre"),
                        ("figo_ovary","FIGO Ovaire"),("orads","O-RADS"),("sopk","SOPK")]:
        sc = scores.get(key, {})
        if sc:
            stage = sc.get("figo_stage") or sc.get("category") or sc.get("diagnosis","—")
            reco  = sc.get("recommendation","—")[:70]
            sc_html += f"<tr><th>{label}</th><td>{stage}</td><td>{reco}</td></tr>"

    roma_html = ""
    if roma:
        rc = "#C0392B" if roma.get("high_risk") else "#27AE60"
        roma_html = (f'<p><b>ROMA Score :</b> <b style="color:{rc}">{roma.get("score_pct",0):.1f}%</b> — '
                     f'{"⚠ Risque élevé" if roma.get("high_risk") else "✓ Risque faible"}</p>')

    return f"""<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8">
<title>GynoCare AI — {patient_id}</title>
<style>
body{{font-family:Arial,sans-serif;font-size:11px;color:#1C2833;margin:0;padding:12px}}
h1{{background:#9B2D6F;color:#fff;padding:8px 14px;font-size:13px;margin:0}}
h2{{font-size:10px;color:#0E7373;text-transform:uppercase;margin:10px 0 4px;border-bottom:1px solid #E8D5F0;padding-bottom:2px}}
table{{width:100%;border-collapse:collapse;margin-bottom:6px}}
th{{background:#1A5276;color:#fff;padding:3px 6px;font-size:9px;text-align:left}}
td{{padding:3px 6px;border:1px solid #E8D5F0;font-size:9px}}
tr:nth-child(even){{background:#F4ECF7}}
.disc{{background:#FFF8E1;border-left:4px solid #F39C12;padding:6px 10px;font-size:8px;margin-top:10px}}
.info{{background:#F9F3FC;padding:5px 10px;margin:0;font-size:9px}}
</style></head><body>
<h1>KANÉA — GynoCare AI v2.0 · Rapport gynécologique</h1>
<p class="info">Dossier : {patient_id} · {date_str} · ID : {result.get("request_id","—")[:16]}</p>
<h2>Diagnostic IA principal</h2>
<p><b style="color:{urg_hex};font-size:13px">{pred}</b>
&nbsp;&nbsp; Confiance : <b>{conf:.1%}</b> &nbsp;&nbsp; Urgence : <b style="color:{urg_hex}">{urgency}</b></p>
<p><b>Action :</b> {action}</p>
<h2>Marqueurs tumoraux</h2>
<p>{tm.get("interpretation","—")}</p>
{roma_html}
{('<table><tr><th>Marqueur</th><th>Valeur</th><th>Statut</th></tr>' + tm_flags_html + '</table>') if tm_flags_html else ''}
<h2>Probabilités diagnostiques</h2>
<table><tr><th>Diagnostic</th><th>Probabilité</th></tr>{probs_rows}</table>
{('<h2>Scores cliniques FIGO</h2><table><tr><th>Score</th><th>Résultat</th><th>Recommandation</th></tr>' + sc_html + '</table>') if sc_html else ''}
<h2>Alertes cliniques</h2>{alert_rows if alert_rows else '<p>Aucune alerte critique.</p>'}
<div class="disc">⚠ Outil d'aide à la décision médicale — validation médicale obligatoire.
GynoCare AI v2.0 — KANÉA 2026 — FIGO 2018/2023 · ESGO 2020 · NCCN 2023.</div>
</body></html>"""
