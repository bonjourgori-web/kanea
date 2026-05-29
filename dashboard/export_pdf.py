"""
KANÉA — Export PDF des rapports d'analyse
==========================================
Génère un rapport PDF clinique à partir d'un résultat de prédiction.
Fonctionne sans dépendance externe lourde (utilise uniquement reportlab
ou, en fallback, une page HTML convertie en bytes téléchargeable).

Usage depuis le dashboard Streamlit :
    from dashboard.export_pdf import build_pdf_report
    pdf_bytes = build_pdf_report(result, module="malaria")
    st.download_button("Télécharger le rapport", pdf_bytes, "rapport_kanea.pdf")
"""
from __future__ import annotations

import base64
import datetime
import io
import uuid
from pathlib import Path
from typing import Any

_KANEA_ROOT = Path(__file__).resolve().parents[1]


def _fallback_html_report(result: dict[str, Any], module: str) -> bytes:
    """Génère un rapport HTML (fallback si reportlab non installé)."""
    now  = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    pred = result.get("prediction") or result.get("prediction", {})
    conf = result.get("confidence")
    status = result.get("status", "—")

    rows_html = ""
    for k, v in result.items():
        if k in ("explainability",) or (isinstance(v, str) and len(v) > 200):
            continue
        rows_html += f"<tr><td style='padding:6px 10px;color:#5E7A8A;font-weight:600'>{k}</td><td style='padding:6px 10px'>{v}</td></tr>"

    expl = result.get("explainability", {})
    shap_html = ""
    if isinstance(expl, dict) and expl.get("top_features"):
        shap_html = "<h3 style='color:#20B2AA'>Importance des features</h3><table>"
        for feat in expl["top_features"]:
            pct = feat['importance'] * 100
            shap_html += (
                f"<tr><td style='padding:4px 10px'>{feat['feature']}</td>"
                f"<td><div style='width:{pct*3:.0f}px;height:12px;"
                f"background:#20B2AA;border-radius:4px'></div></td>"
                f"<td style='padding:4px 8px'>{pct:.1f}%</td></tr>"
            )
        shap_html += "</table>"

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Rapport KANEA</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #1A2B3C; }}
  h1   {{ color: #20B2AA; border-bottom: 2px solid #20B2AA; padding-bottom: 8px; }}
  h2   {{ color: #1A2B3C; margin-top: 24px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
  tr:nth-child(even) {{ background: #F7F9FC; }}
  .badge {{ display:inline-block; padding:4px 12px; border-radius:999px;
            font-weight:700; font-size:0.9rem; }}
  .badge-ok  {{ background:#E9F7EF; color:#27AE60; }}
  .badge-err {{ background:#FDECEA; color:#E74C3C; }}
  .badge-warn{{ background:#FEF0E7; color:#E67E22; }}
  .footer {{ margin-top:40px; font-size:0.75rem; color:#8AABB8; border-top:1px solid #DDE8EE; padding-top:8px; }}
</style>
</head>
<body>
  <h1>🧬 KANEA — Rapport d'analyse</h1>
  <p><strong>Module :</strong> {module} &nbsp;|&nbsp;
     <strong>Date :</strong> {now} &nbsp;|&nbsp;
     <strong>Statut :</strong>
     <span class="badge {'badge-ok' if status == 'model_loaded' else 'badge-warn'}">{status}</span>
  </p>

  <h2>Résultat</h2>
  {"<p><strong>Prédiction :</strong> " + str(pred) + "</p>" if pred else ""}
  {"<p><strong>Confiance :</strong> " + f"{conf:.1%}" + "</p>" if conf else ""}

  <h2>Données complètes</h2>
  <table>{rows_html}</table>

  {shap_html}

  <div class="footer">
    KANEA v1.0 · Knowledge Anthropology & Neural Engine for Africa ·
    Outil d'aide à la décision — pas un diagnostic officiel. Supervision médicale requise.
  </div>
</body>
</html>"""
    return html.encode("utf-8")


def _qr_code_image(data: str, size_px: int = 120) -> io.BytesIO | None:
    """Génère un QR code en mémoire (PIL Image → BytesIO PNG)."""
    try:
        import qrcode
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M,
                           box_size=4, border=2)
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        img = img.resize((size_px, size_px))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except Exception:
        return None


def _b64_to_image_buffer(b64_str: str, size: tuple[int, int] = (160, 160)) -> io.BytesIO | None:
    """Convertit un base64 PNG en BytesIO redimensionné."""
    try:
        from PIL import Image as PILImage
        raw = base64.b64decode(b64_str)
        img = PILImage.open(io.BytesIO(raw)).convert("RGB").resize(size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except Exception:
        return None


def _build_breast_cancer_pdf(result: dict[str, Any], patient_id: str | None) -> bytes:
    """Rapport médical A4 BreastCancer AI v3.0 — EfficientNet-B0 + CAM + grade/TNM/ER/Ki67 + QR code."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Image as RLImage, Paragraph, SimpleDocTemplate,
        Spacer, Table, TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=1.8*cm, bottomMargin=2*cm,
                            leftMargin=2.2*cm, rightMargin=2.2*cm)
    styles  = getSampleStyleSheet()
    teal    = colors.HexColor("#20B2AA")
    red_c   = colors.HexColor("#C0392B")
    ink     = colors.HexColor("#1A2B3C")
    muted   = colors.HexColor("#5E7A8A")

    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=teal, fontSize=17, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=ink, fontSize=12,
                        spaceBefore=12, spaceAfter=3)
    bd = ParagraphStyle("bd", parent=styles["Normal"], textColor=ink, fontSize=9.5, leading=14)
    sm = ParagraphStyle("sm", parent=styles["Normal"], textColor=muted, fontSize=8, leading=12)

    now       = datetime.datetime.now()
    report_id = f"BC-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    pred      = result.get("prediction") or "—"
    conf      = result.get("confidence") or 0.0
    probs     = result.get("probabilities") or {}
    img_name  = result.get("input_image") or "—"
    grade     = result.get("tumor_grade") or {}
    tnm       = result.get("tnm_stage") or {}
    recep     = result.get("receptor_status") or {}
    ki67      = result.get("ki67") or {}
    subtyp    = result.get("tumor_subtype") or {}
    safety    = result.get("clinical_safety") or {}
    reco      = result.get("recommendations") or {}

    pred_colors = {"Normal": "#27AE60", "Benign": "#F39C12", "Malignant": "#C0392B"}
    pred_hex    = pred_colors.get(pred, "#5E7A8A")
    urgency     = reco.get("urgency", "—")

    qr_data = (f"KANEA/BreastCancerAI | ID:{report_id} | "
               f"Résultat:{pred} | Conf:{conf:.1%} | "
               f"Stage:{tnm.get('clinical_stage','—')} | "
               f"Date:{now.strftime('%Y-%m-%d %H:%M')}")
    qr_buf = _qr_code_image(qr_data, size_px=100)

    story = [
        Paragraph("KANEA — BreastCancer AI", h1),
        Paragraph("Rapport d'analyse oncologique automatisée — EfficientNet-B0", sm),
        HRFlowable(width="100%", thickness=2, color=teal, spaceAfter=6),
        Paragraph(
            f"<b>N° rapport :</b> {report_id} &nbsp;&nbsp; "
            f"<b>Date :</b> {now.strftime('%d/%m/%Y %H:%M')} &nbsp;&nbsp; "
            f"<b>Image :</b> {img_name}"
            + (f" &nbsp;&nbsp; <b>Patient :</b> {patient_id}" if patient_id else ""),
            bd,
        ),
        Spacer(1, 0.3*cm),
    ]

    # Résultat principal + QR
    pred_label = {"Normal": "NORMAL — Aucun signe de malignité",
                  "Benign": "BÉNIN — Lésion non cancéreuse",
                  "Malignant": "MALIN — Signe potentiel de cancer"}.get(pred, pred)
    result_txt = [
        Paragraph("1. Résultat de l'analyse", h2),
        Paragraph(f"<font color='{pred_hex}'><b>{pred_label}</b></font>",
                  ParagraphStyle("res", parent=bd, fontSize=11)),
        Spacer(1, 0.1*cm),
        Paragraph(f"<b>Confiance :</b> {conf:.1%}  &nbsp;|&nbsp; <b>Urgence :</b> <font color='{pred_hex}'>{urgency}</font>", bd),
        Paragraph(f"<b>Probabilités :</b> Normal {probs.get('Normal',0):.1%}  "
                  f"| Bénin {probs.get('Benign',0):.1%}  "
                  f"| Malin {probs.get('Malignant',0):.1%}", bd),
    ]
    if qr_buf:
        qr_img  = RLImage(qr_buf, width=2.5*cm, height=2.5*cm)
        side    = Table([[result_txt, qr_img]], colWidths=[13.5*cm, 2.8*cm])
        side.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("ALIGN",(1,0),(1,0),"CENTER")]))
        story.append(side)
    else:
        story.extend(result_txt)
    story.append(Spacer(1, 0.3*cm))

    # Analyses cliniques
    if pred == "Malignant" and (grade or tnm or recep or ki67):
        story.append(Paragraph("2. Analyses cliniques avancées", h2))
        clinical_rows = [["Paramètre", "Valeur", "Interprétation clinique"]]

        if grade:
            g = grade.get("dominant_grade", "—")
            clinical_rows.append(["Grade tumoral", g, grade.get("clinical_note", "—")])
        if tnm:
            stage_str = f"{tnm.get('T','?')} {tnm.get('N','?')} {tnm.get('M','?')} — {tnm.get('clinical_stage','—')}"
            clinical_rows.append(["Stade TNM", stage_str, tnm.get("stage_note", "—")])
        if subtyp:
            clinical_rows.append(["Sous-type", subtyp.get("dominant_subtype","—"),
                                   subtyp.get("epidemiological_note","—")])
        if recep:
            r_str = (f"ER {recep.get('ER_percentage',0)}% ({recep.get('ER_status','?')})  "
                     f"PR {recep.get('PR_percentage',0)}% ({recep.get('PR_status','?')})  "
                     f"HER2 {recep.get('HER2_status','?')}")
            clinical_rows.append(["Récepteurs ER/PR/HER2", r_str,
                                   f"Phénotype : {recep.get('phenotype','—')}"])
        if ki67:
            clinical_rows.append(["Ki67 (prolifération)", f"{ki67.get('percentage',0)}%",
                                   ki67.get("interpretation","—")])

        if len(clinical_rows) > 1:
            ct = Table(clinical_rows, colWidths=[4*cm, 5*cm, 7.2*cm])
            ct.setStyle(TableStyle([
                ("BACKGROUND",    (0,0),(-1,0), teal),
                ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
                ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
                ("FONTSIZE",      (0,0),(-1,-1), 8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#F7F9FC")]),
                ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#DDE8EE")),
                ("TOPPADDING",    (0,0),(-1,-1), 4),
                ("BOTTOMPADDING", (0,0),(-1,-1), 4),
                ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ]))
            story += [ct, Spacer(1, 0.15*cm)]
            story.append(Paragraph(
                "<i>* Grade, TNM, récepteurs, Ki67 : estimations algorithmiques (priors SEER + "
                "analyse CAM). Confirmation par biopsie et IHC obligatoire.</i>",
                ParagraphStyle("note", parent=sm, textColor=muted, fontSize=7.5),
            ))
            story.append(Spacer(1, 0.2*cm))

    # Alerte sécurité
    if safety.get("level") in ("critical", "warning", "low_confidence"):
        story.append(Paragraph(
            f"<b>Alerte :</b> {safety.get('message','')}",
            ParagraphStyle("alert", parent=bd, textColor=red_c, fontSize=9),
        ))
        story.append(Spacer(1, 0.15*cm))

    # CAM
    expl    = result.get("explainability") or {}
    cam_b64 = expl.get("heatmap_b64") if isinstance(expl, dict) else None
    sec_num = 3 if pred == "Malignant" else 2
    if cam_b64:
        cam_buf = _b64_to_image_buffer(cam_b64, size=(200, 200))
        if cam_buf:
            story += [
                Paragraph(f"{sec_num}. Carte d'activation CAM — Zones suspectes", h2),
                Paragraph("Rouge = forte activation (zone suspecte), Bleu = faible activation.", sm),
                Spacer(1, 0.2*cm),
                RLImage(cam_buf, width=5*cm, height=5*cm),
                Spacer(1, 0.2*cm),
            ]
            sec_num += 1

    # Recommandations
    actions = reco.get("actions") or []
    if actions:
        story.append(Paragraph(f"{sec_num}. Recommandations cliniques", h2))
        story.append(Paragraph(f"<b>Urgence :</b> <font color='{pred_hex}'>{urgency}</font>  "
                                f"&nbsp;|&nbsp;  <b>Suivi :</b> {reco.get('follow_up','—')}", bd))
        story.append(Spacer(1, 0.1*cm))
        for i, action in enumerate(actions, 1):
            story.append(Paragraph(f"{i}. {action}", bd))

    # Pied de page
    story += [
        Spacer(1, 0.4*cm),
        HRFlowable(width="100%", thickness=0.5, color=muted),
        Paragraph(
            f"KANEA v3.0 · BreastCancer AI · EfficientNet-B0 · "
            f"Outil d'aide à la décision — validation médicale obligatoire · "
            f"Rapport {report_id} · {now.strftime('%d/%m/%Y')}",
            sm,
        ),
    ]
    doc.build(story)
    return buf.getvalue()


def _build_nutrition_pdf(result: dict[str, Any], patient_id: str | None) -> bytes:
    """Rapport médical A4 NutriTrack AI — SHAP + Z-scores + recommandations + QR code."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Image as RLImage, Paragraph, SimpleDocTemplate,
        Spacer, Table, TableStyle,
    )

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               topMargin=1.8*cm, bottomMargin=2*cm,
                               leftMargin=2.2*cm, rightMargin=2.2*cm)
    styles = getSampleStyleSheet()
    teal   = colors.HexColor("#20B2AA")
    ink    = colors.HexColor("#1A2B3C")
    muted  = colors.HexColor("#5E7A8A")

    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=teal, fontSize=17, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=ink, fontSize=12,
                        spaceBefore=12, spaceAfter=3)
    bd = ParagraphStyle("bd", parent=styles["Normal"], textColor=ink, fontSize=9.5, leading=14)
    sm = ParagraphStyle("sm", parent=styles["Normal"], textColor=muted, fontSize=8, leading=12)

    now       = datetime.datetime.now()
    report_id = f"NUT-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    pred      = result.get("prediction") or "—"
    conf      = result.get("confidence") or 0.0
    risk      = result.get("risk_level") or "—"
    inputs    = result.get("inputs_summary") or {}
    derived   = result.get("derived_features") or {}
    reco      = result.get("recommendations") or {}

    # Couleur urgence
    urgency_color_map = {
        "CRITIQUE": "#C0392B", "ÉLEVÉE": "#E67E22",
        "MODÉRÉE": "#F39C12", "FAIBLE": "#27AE60",
    }
    urgency      = reco.get("urgency", "—")
    urge_hex   = urgency_color_map.get(urgency, "#5E7A8A")
    pred_label = reco.get("label_fr", pred)

    # QR code
    qr_data = (f"KANEA/NutriTrack | ID:{report_id} | "
               f"Statut:{pred} | Risque:{risk} | Conf:{conf:.1%} | "
               f"Date:{now.strftime('%Y-%m-%d %H:%M')}")
    qr_buf = _qr_code_image(qr_data, size_px=100)

    story = [
        Paragraph("KANEA — NutriTrack AI", h1),
        Paragraph("Rapport d'analyse nutritionnelle automatisée", sm),
        HRFlowable(width="100%", thickness=2, color=teal, spaceAfter=6),
        Paragraph(
            f"<b>N° rapport :</b> {report_id} &nbsp;&nbsp; "
            f"<b>Date :</b> {now.strftime('%d/%m/%Y %H:%M')}"
            + (f" &nbsp;&nbsp; <b>Patient :</b> {patient_id}" if patient_id else ""),
            bd,
        ),
        Spacer(1, 0.3*cm),
    ]

    # Résultat + QR code
    result_txt = [
        Paragraph("1. Statut nutritionnel", h2),
        Paragraph(
            f"<font color='{urge_hex}'><b>{pred_label}</b></font>",
            ParagraphStyle("res", parent=bd, fontSize=11),
        ),
        Spacer(1, 0.1*cm),
        Paragraph(f"<b>Confiance :</b> {conf:.1%}  &nbsp;|&nbsp; <b>Niveau de risque :</b> {risk.upper()}", bd),
        Paragraph(f"<b>Urgence :</b> <font color='{urge_hex}'>{urgency}</font>", bd),
    ]
    if qr_buf:
        qr_img = RLImage(qr_buf, width=2.5*cm, height=2.5*cm)
        side   = Table([[result_txt, qr_img]], colWidths=[13.5*cm, 2.8*cm])
        side.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN",  (1, 0), (1, 0),  "CENTER"),
        ]))
        story.append(side)
    else:
        story.extend(result_txt)
    story.append(Spacer(1, 0.3*cm))

    # Données anthropométriques
    story.append(Paragraph("2. Données anthropométriques", h2))
    anthr_rows = [["Mesure", "Valeur", "Interprétation OMS"]]

    bmi = derived.get("bmi")
    waz = derived.get("waz")
    haz = derived.get("haz")
    whz = derived.get("whz")
    rs  = derived.get("nutrition_risk_score")

    anthr_data = [
        ("Âge",     f"{inputs.get('age_months', '—')} mois",  ""),
        ("Poids",   f"{inputs.get('weight_kg', '—')} kg",      ""),
        ("Taille",  f"{inputs.get('height_cm', '—')} cm",      ""),
        ("MUAC",    f"{inputs.get('muac_cm', '—')} cm",
            "< 11.5 cm = MAS | 11.5–12.5 cm = MAM" if inputs.get("muac_cm") else ""),
        ("IMC",     f"{bmi:.1f} kg/m²" if bmi else "—",
            "Normal: 18.5–25 | Maigreur: < 18.5 | Obésité: > 30"),
        ("WAZ",     f"{waz:.2f}" if waz else "—",    "< -3 = MAS | -3 à -2 = MAM"),
        ("HAZ",     f"{haz:.2f}" if haz else "—",    "< -3 = Retard sévère | -3 à -2 = Retard"),
        ("WHZ",     f"{whz:.2f}" if whz else "—",    "< -3 = MAS | -3 à -2 = MAM"),
        ("Score risque", f"{rs}/10" if rs is not None else "—",
            "0–2 = Faible | 3–5 = Modéré | 6–8 = Élevé | 9–10 = Critique"),
    ]
    for label, val, interp in anthr_data:
        anthr_rows.append([label, val, interp])

    at = Table(anthr_rows, colWidths=[4*cm, 3.5*cm, 8.7*cm])
    at.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), teal),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE8EE")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story += [at, Spacer(1, 0.3*cm)]

    # SHAP top features
    expl       = result.get("explainability") or {}
    top_feats  = expl.get("top_features") or []
    shap_img_b64 = expl.get("shap_img_b64")

    if top_feats or shap_img_b64:
        story.append(Paragraph("3. Facteurs prédictifs (SHAP)", h2))
        story.append(Paragraph(
            f"Méthode : {expl.get('method', 'feature_importance')} — "
            f"Variables ayant le plus influencé la prédiction :",
            sm,
        ))
        story.append(Spacer(1, 0.15*cm))

        if shap_img_b64:
            shap_buf = _b64_to_image_buffer(shap_img_b64, size=(320, 190))
            if shap_buf:
                story.append(RLImage(shap_buf, width=8*cm, height=4.8*cm))
        elif top_feats:
            feat_rows = [["Variable", "Importance", "Direction"]]
            for f in top_feats[:6]:
                feat_rows.append([
                    f["feature"],
                    f"{f['importance']:.4f}",
                    "Augmente risque" if f.get("direction") == "+" else "Réduit risque",
                ])
            ft = Table(feat_rows, colWidths=[5*cm, 3.5*cm, 7.7*cm])
            ft.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), teal),
                ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
                ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE8EE")),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(ft)
        story.append(Spacer(1, 0.2*cm))

    # Recommandations cliniques
    actions = reco.get("actions") or []
    if actions:
        story.append(Paragraph("4. Recommandations cliniques", h2))
        story.append(Paragraph(
            f"<b>Urgence :</b> <font color='{urge_hex}'>{urgency}</font>  &nbsp;|&nbsp;  "
            f"<b>Suivi :</b> {reco.get('monitoring', '—')}",
            bd,
        ))
        story.append(Spacer(1, 0.1*cm))
        for i, action in enumerate(actions, 1):
            story.append(Paragraph(f"{i}. {action}", bd))
        story.append(Spacer(1, 0.1*cm))
        if reco.get("diet_advice"):
            story.append(Paragraph(f"<b>Conseil nutritionnel :</b> {reco['diet_advice']}", bd))
        story.append(Paragraph(
            f"<b>Seuil OMS de référence :</b> {reco.get('oms_threshold', '—')}",
            sm,
        ))

    # Pied de page
    story += [
        Spacer(1, 0.4*cm),
        HRFlowable(width="100%", thickness=0.5, color=muted),
        Paragraph(
            f"KANEA v3.0 · NutriTrack AI · Outil d'aide à la décision — "
            f"Validation clinique obligatoire · Rapport {report_id} · "
            f"{now.strftime('%d/%m/%Y')}",
            sm,
        ),
    ]
    doc.build(story)
    return buf.getvalue()


def _build_malaria_pdf(result: dict[str, Any], patient_id: str | None) -> bytes:
    """Rapport médical A4 dédié MalariaScan AI avec QR code et carte CAM."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Image as RLImage, Paragraph, SimpleDocTemplate,
        Spacer, Table, TableStyle,
    )

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               topMargin=1.8*cm, bottomMargin=2*cm,
                               leftMargin=2.2*cm, rightMargin=2.2*cm)
    styles = getSampleStyleSheet()
    teal   = colors.HexColor("#20B2AA")
    red    = colors.HexColor("#C0392B")
    ink    = colors.HexColor("#1A2B3C")
    muted  = colors.HexColor("#5E7A8A")

    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=teal, fontSize=17, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=ink, fontSize=12,
                        spaceBefore=12, spaceAfter=3)
    bd = ParagraphStyle("bd", parent=styles["Normal"], textColor=ink, fontSize=9.5, leading=14)
    sm = ParagraphStyle("sm", parent=styles["Normal"], textColor=muted, fontSize=8, leading=12)

    now        = datetime.datetime.now()
    report_id  = f"MAL-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    pred       = result.get("prediction") or "—"
    conf       = result.get("confidence") or 0.0
    probs      = result.get("probabilities") or {}
    image_name = result.get("input_image") or "—"
    is_pos     = pred == "Parasitized"
    pred_label = "POSITIF — Plasmodium détecté" if is_pos else "NÉGATIF — Aucun parasite détecté"
    reco       = (
        "Initier immédiatement un traitement antipaludéen (ACT) selon les protocoles nationaux. "
        "Contrôle parasitologique à J3 recommandé."
        if is_pos else
        "Aucun traitement antipaludéen requis. Suivi clinique si la symptomatologie persiste."
    )

    # QR code encodant les données clés du rapport
    qr_data = (f"KANEA/MalariaScan | ID:{report_id} | "
               f"Résultat:{pred} | Confiance:{conf:.1%} | "
               f"Date:{now.strftime('%Y-%m-%d %H:%M')}")
    qr_buf  = _qr_code_image(qr_data, size_px=100)

    story = [
        # En-tête
        Paragraph("KANEA — MalariaScan AI", h1),
        Paragraph("Rapport d'analyse parasitologique automatisée", sm),
        HRFlowable(width="100%", thickness=2, color=teal, spaceAfter=6),

        # Identifiants
        Paragraph(
            f"<b>N° rapport :</b> {report_id} &nbsp;&nbsp; "
            f"<b>Date :</b> {now.strftime('%d/%m/%Y %H:%M')} &nbsp;&nbsp; "
            f"<b>Image :</b> {image_name}"
            + (f" &nbsp;&nbsp; <b>Patient :</b> {patient_id}" if patient_id else ""),
            bd,
        ),
        Spacer(1, 0.3*cm),
    ]

    # Résultat principal + QR code en tableau côte à côte
    result_text = [
        Paragraph("1. Résultat de l'analyse", h2),
        Paragraph(
            f"<font color='{'#C0392B' if is_pos else '#27AE60'}'><b>{pred_label}</b></font>",
            ParagraphStyle("res", parent=bd, fontSize=11),
        ),
        Spacer(1, 0.15*cm),
        Paragraph(f"<b>Confiance :</b> {conf:.1%}", bd),
        Paragraph(f"<b>Probabilité Parasitized :</b> {probs.get('Parasitized', 0):.1%}", bd),
        Paragraph(f"<b>Probabilité Uninfected :</b> {probs.get('Uninfected', 0):.1%}", bd),
    ]
    if qr_buf:
        qr_img  = RLImage(qr_buf, width=2.5*cm, height=2.5*cm)
        side_tbl = Table([[result_text, qr_img]], colWidths=[13.5*cm, 2.8*cm])
        side_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN",  (1, 0), (1, 0),  "CENTER"),
        ]))
        story.append(side_tbl)
    else:
        story.extend(result_text)

    story.append(Spacer(1, 0.3*cm))

    # Données cliniques avancées (espèce, stade, parasitémie)
    parasitemia = result.get("parasitemia") or {}
    species     = result.get("species_prediction") or {}
    stage       = result.get("parasite_stage") or {}
    safety      = result.get("clinical_safety") or {}

    if is_pos and (parasitemia or species or stage):
        story.append(Paragraph("2. Analyses cliniques avancées", h2))

        # Tableau parasitémie + espèce + stade
        clinical_rows = [["Paramètre", "Valeur", "Interprétation clinique"]]
        if parasitemia:
            pct = parasitemia.get("percentage", 0)
            sev = parasitemia.get("severity", "—")
            inf = parasitemia.get("infected_cells", 0)
            tot = parasitemia.get("total_cells", 120)
            # Interprétation OMS de la sévérité
            if pct < 1:
                interp_para = "Faible risque — surveillance standard"
            elif pct < 5:
                interp_para = "Risque modéré — traitement ACT recommandé"
            elif pct < 10:
                interp_para = "Risque sévère — hospitalisation requise"
            else:
                interp_para = "URGENCE — hyperparasitémie, soins intensifs"
            clinical_rows.append([
                "Parasitémie estimée",
                f"{pct}%  ({sev})",
                f"{inf}/{tot} hématies · {interp_para}",
            ])

        if species:
            dom   = species.get("dominant_species", "—")
            pf    = species.get("plasmodium_falciparum", 0)
            pv    = species.get("plasmodium_vivax", 0)
            pm    = species.get("plasmodium_malariae", 0)
            po    = species.get("plasmodium_ovale", 0)
            # Fiche clinique espèce dominante
            species_notes = {
                "Plasmodium falciparum": "Forme la plus létale — accès pernicieux possible, résistances ACT à surveiller",
                "Plasmodium vivax":      "Rechutes possibles (hypnozoïtes) — ajouter primaquine au traitement",
                "Plasmodium malariae":   "Évolution chronique — syndrome néphrotique possible à long terme",
                "Plasmodium ovale":      "Rechutes possibles (hypnozoïtes) — traitement radical recommandé",
                "Plasmodium knowlesi":   "Zoonose rare — suivi strict en zone endémique Asie du Sud-Est",
            }
            sp_note = species_notes.get(dom, "Espèce à confirmer par expert")
            clinical_rows.append([
                "Espèce (estimation)",
                f"{dom}\n(Pf={pf:.0%} Pv={pv:.0%} Pm={pm:.0%} Po={po:.0%})",
                sp_note,
            ])

        if stage:
            st_dom   = stage.get("dominant_stage", "—")
            st_ring  = stage.get("ring", 0)
            st_troph = stage.get("trophozoite", 0)
            st_schi  = stage.get("schizont", 0)
            st_gam   = stage.get("gametocyte", 0)
            stage_notes = {
                "Ring":        "Stade précoce — traitement ACT généralement efficace",
                "Trophozoite": "Stade intermédiaire — métabolisme parasitaire actif",
                "Schizont":    "Stade avancé — risque de séquestration endothéliale (Pf)",
                "Gametocyte":  "Forme transmissible — risque de propagation vectorielle",
            }
            st_note = stage_notes.get(st_dom, "—")
            clinical_rows.append([
                "Stade parasitaire",
                f"{st_dom}",
                f"Ring {st_ring:.0%} · Troph {st_troph:.0%} · Schi {st_schi:.0%} · Gam {st_gam:.0%}\n{st_note}",
            ])

        if len(clinical_rows) > 1:
            ct = Table(clinical_rows, colWidths=[4.2*cm, 4.5*cm, 7.5*cm])
            ct.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), teal),
                ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, 0), 8),
                ("FONTSIZE",      (0, 1), (-1, -1), 8),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
                ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE8EE")),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ]))
            story += [ct, Spacer(1, 0.15*cm)]
            story.append(Paragraph(
                "<i>* Espèce et stade : estimations algorithmiques (priors épidémiologiques OMS Afrique + "
                "analyse spatiale CAM). Confirmation par microscopie optique ou PCR obligatoire avant "
                "décision thérapeutique.</i>",
                ParagraphStyle("note", parent=sm, textColor=muted, fontSize=7.5),
            ))
            story.append(Spacer(1, 0.2*cm))

    # Alerte sécurité clinique
    if safety.get("level") in ("warning", "low_confidence"):
        story += [
            Paragraph(
                f"<b>Alerte sécurité :</b> {safety.get('message', '')}",
                ParagraphStyle("alert", parent=bd, textColor=red, fontSize=9),
            ),
            Spacer(1, 0.15*cm),
        ]

    # Carte CAM
    expl    = result.get("explainability") or {}
    cam_b64 = expl.get("heatmap_b64") if isinstance(expl, dict) else None
    sec_num = 3 if (is_pos and (parasitemia or species or stage)) else 2
    if cam_b64:
        cam_buf = _b64_to_image_buffer(cam_b64, size=(200, 200))
        if cam_buf:
            story += [
                Paragraph(f"{sec_num}. Carte d'activation CAM — Localisation du parasite", h2),
                Paragraph(
                    "Rouge = forte activation (zone suspecte), Bleu = faible activation.",
                    sm,
                ),
                Spacer(1, 0.2*cm),
                RLImage(cam_buf, width=5*cm, height=5*cm),
                Spacer(1, 0.2*cm),
            ]
            sec_num += 1

    # Méthodologie
    story += [
        Paragraph(f"{sec_num}. Méthodologie", h2),
        Paragraph(
            "ResNet34 entraîné sur NIH Malaria Cell Images (27 560 images). "
            "Prétraitement 224×224 px, normalisation ImageNet. ONNX Runtime CPU. "
            "CAM = Class Activation Map (équivalent Grad-CAM pour ResNet+GAP).",
            bd,
        ),
        Paragraph(
            "Accuracy 92.5% · AUC-ROC 0.969 · Version v3.0",
            bd,
        ),
        Spacer(1, 0.15*cm),
    ]

    # Recommandation clinique
    story += [
        Paragraph(f"{sec_num + 1}. Recommandation clinique", h2),
        Paragraph(reco, bd),
        Spacer(1, 0.15*cm),
        Paragraph(
            "<b>Important :</b> Outil d'aide à la décision — pas un diagnostic officiel. "
            "Validation par biologiste qualifié obligatoire avant toute décision thérapeutique.",
            ParagraphStyle("warn", parent=bd, textColor=red, fontSize=8.5),
        ),
    ]

    # Pied de page
    story += [
        Spacer(1, 0.4*cm),
        HRFlowable(width="100%", thickness=0.5, color=muted),
        Paragraph(
            f"KANEA v2.1 · MalariaScan AI · Knowledge Anthropology &amp; Neural Engine for Africa · "
            f"Rapport {report_id} · {now.strftime('%d/%m/%Y')}",
            sm,
        ),
    ]

    doc.build(story)
    return buf.getvalue()


def build_pdf_report(
    result: dict[str, Any],
    module: str = "analyse",
    patient_id: str | None = None,
) -> tuple[bytes, str]:
    """
    Génère un rapport téléchargeable à partir d'un résultat KANEA.

    Returns:
        (bytes, mime_type) — contenu du fichier et son type MIME.
        Priorité : PDF (reportlab) → HTML (fallback universel).
    """
    try:
        # Module malaria : rapport clinique dédié avec QR code + CAM
        if module == "malaria":
            pdf_bytes = _build_malaria_pdf(result, patient_id)
            return pdf_bytes, "application/pdf"

        # Module nutrition : rapport NutriTrack AI avec SHAP + recommandations
        if module in ("nutrition", "biometry"):
            pdf_bytes = _build_nutrition_pdf(result, patient_id)
            return pdf_bytes, "application/pdf"

        # Module cancer du sein : rapport BreastCancer AI
        if module in ("breast_cancer", "cancer_sein"):
            pdf_bytes = _build_breast_cancer_pdf(result, patient_id)
            return pdf_bytes, "application/pdf"

        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )

        buf    = io.BytesIO()
        doc    = SimpleDocTemplate(buf, pagesize=A4,
                                   topMargin=2*cm, bottomMargin=2*cm,
                                   leftMargin=2.2*cm, rightMargin=2.2*cm)
        styles = getSampleStyleSheet()
        teal   = colors.HexColor("#20B2AA")
        ink    = colors.HexColor("#1A2B3C")
        muted  = colors.HexColor("#5E7A8A")

        title_style = ParagraphStyle("title", parent=styles["Heading1"],
                                     textColor=teal, fontSize=18, spaceAfter=4)
        h2_style    = ParagraphStyle("h2", parent=styles["Heading2"],
                                     textColor=ink, fontSize=13, spaceBefore=14, spaceAfter=4)
        body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                     textColor=ink, fontSize=10, leading=14)
        small_style = ParagraphStyle("small", parent=styles["Normal"],
                                     textColor=muted, fontSize=8, leading=12)

        now    = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        pred   = result.get("prediction")
        conf   = result.get("confidence")
        status = result.get("status", "—")

        story = [
            Paragraph("KANEA — Rapport d'analyse", title_style),
            HRFlowable(width="100%", thickness=2, color=teal, spaceAfter=8),
            Paragraph(f"<b>Module :</b> {module} &nbsp;&nbsp; <b>Date :</b> {now}"
                      + (f" &nbsp;&nbsp; <b>Patient :</b> {patient_id}" if patient_id else ""),
                      body_style),
            Paragraph(f"<b>Statut :</b> {status}", body_style),
            Spacer(1, 0.4*cm),
        ]

        if pred:
            story += [
                Paragraph("Résultat", h2_style),
                Paragraph(f"<b>Prédiction :</b> {pred}", body_style),
            ]
            if conf:
                story.append(Paragraph(f"<b>Confiance :</b> {conf:.1%}", body_style))
            story.append(Spacer(1, 0.3*cm))

        # Tableau des données
        story.append(Paragraph("Données d'entrée", h2_style))
        summary = result.get("inputs_summary") or {}
        derived = result.get("derived_features") or {}
        table_data = [["Paramètre", "Valeur"]]
        for k, v in {**summary, **derived}.items():
            if v is not None:
                table_data.append([str(k), str(v)])

        if len(table_data) > 1:
            t = Table(table_data, colWidths=[7*cm, 9*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), teal),
                ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE8EE")),
                ("TOPPADDING",  (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.3*cm))

        # SHAP / feature importance
        expl = result.get("explainability", {})
        if isinstance(expl, dict) and expl.get("top_features"):
            story.append(Paragraph("Importance des features", h2_style))
            fi_data = [["Feature", "Importance"]]
            for feat in expl["top_features"]:
                fi_data.append([feat["feature"], f"{feat['importance']*100:.1f}%"])
            fi_t = Table(fi_data, colWidths=[10*cm, 6*cm])
            fi_t.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), teal),
                ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE8EE")),
                ("TOPPADDING",  (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(fi_t)
            story.append(Spacer(1, 0.3*cm))

        # Disclaimer
        story += [
            HRFlowable(width="100%", thickness=0.5, color=muted, spaceBefore=12),
            Paragraph(
                "KANEA v1.0 · Outil d'aide à la décision — pas un diagnostic officiel. "
                "Interprétation sous supervision médicale requise.",
                small_style,
            ),
        ]

        doc.build(story)
        return buf.getvalue(), "application/pdf"

    except ImportError:
        # Fallback HTML si reportlab non installé
        return _fallback_html_report(result, module), "text/html"
