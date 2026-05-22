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

import datetime
import io
from typing import Any


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
