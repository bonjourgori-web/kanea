"""
pdf_generator.py — Générateur de rapport médico-légal PDF professionnel.

Utilise ReportLab (reportlab.platypus) pour créer des rapports A4 complets
avec en-tête KANÉA, tableaux de prédictions, graphiques intégrés,
QR code et signature numérique SHA256.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── ReportLab ────────────────────────────────────────────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
        Table, TableStyle, Image, HRFlowable, PageBreak, KeepTogether,
        SimpleDocTemplate,
    )
    from reportlab.platypus.flowables import Flowable
    _RL_OK = True
except ImportError:
    _RL_OK = False
    log.warning("reportlab non disponible — génération PDF désactivée")

# ── QR Code ──────────────────────────────────────────────────────────────────
try:
    import qrcode
    from io import BytesIO
    _QR_OK = True
except ImportError:
    _QR_OK = False
    log.info("qrcode non disponible — fallback texte pour QR")

# ── PIL ───────────────────────────────────────────────────────────────────────
try:
    from PIL import Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


# ── Palette de couleurs ───────────────────────────────────────────────────────
class _C:
    TEAL       = colors.HexColor("#20B2AA")
    DARK_TEAL  = colors.HexColor("#148F85")
    CORAL      = colors.HexColor("#FF6B6B")
    BLUE       = colors.HexColor("#45B7D1")
    DARK_GRAY  = colors.HexColor("#2D3436")
    MED_GRAY   = colors.HexColor("#636E72")
    LIGHT_GRAY = colors.HexColor("#DFE6E9")
    WHITE      = colors.white
    BLACK      = colors.black
    PALE_TEAL  = colors.HexColor("#E0F5F5")
    PALE_RED   = colors.HexColor("#FFF5F5")


def _build_styles() -> dict:
    """Construit les styles de paragraphe ReportLab."""
    base = getSampleStyleSheet()
    styles = {}

    styles["title"] = ParagraphStyle(
        "KaneaTitle",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=_C.TEAL,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    styles["subtitle"] = ParagraphStyle(
        "KaneaSubtitle",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=12,
        textColor=_C.MED_GRAY,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    styles["section"] = ParagraphStyle(
        "KaneaSection",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=_C.DARK_TEAL,
        spaceBefore=14,
        spaceAfter=6,
        borderPad=4,
    )
    styles["body"] = ParagraphStyle(
        "KaneaBody",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=_C.DARK_GRAY,
        leading=14,
        spaceAfter=4,
    )
    styles["small"] = ParagraphStyle(
        "KaneaSmall",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=_C.MED_GRAY,
        leading=11,
    )
    styles["disclaimer"] = ParagraphStyle(
        "KaneaDisclaimer",
        parent=base["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        textColor=_C.MED_GRAY,
        leading=11,
        alignment=TA_JUSTIFY,
    )
    styles["highlight"] = ParagraphStyle(
        "KaneaHighlight",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=_C.WHITE,
        leading=15,
    )
    styles["footer"] = ParagraphStyle(
        "KaneaFooter",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=_C.MED_GRAY,
        alignment=TA_CENTER,
    )
    return styles


def _make_qr_image(data: str, size_cm: float = 3.5) -> Optional[Any]:
    """Génère un QR code ReportLab Image. Retourne None si impossible."""
    if not (_QR_OK and _PIL_OK and _RL_OK):
        return None
    try:
        from io import BytesIO
        qr = qrcode.QRCode(version=1, box_size=10, border=2)
        qr.add_data(data)
        qr.make(fit=True)
        img_pil = qr.make_image(fill_color="black", back_color="white")
        buf = BytesIO()
        img_pil.save(buf, format="PNG")
        buf.seek(0)
        return Image(buf, width=size_cm * cm, height=size_cm * cm)
    except Exception as exc:
        log.warning("QR code generation failed: %s", exc)
        return None


def _hash_content(content: str) -> str:
    """Calcule le hash SHA256 du contenu pour signature numérique."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _table_style_base() -> list:
    """Style de base pour les tableaux ReportLab."""
    return [
        ("BACKGROUND",   (0, 0), (-1, 0), _C.TEAL),
        ("TEXTCOLOR",    (0, 0), (-1, 0), _C.WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 10),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_C.WHITE, _C.PALE_TEAL]),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("GRID",         (0, 0), (-1, -1), 0.5, _C.LIGHT_GRAY),
        ("ROWHEIGHT",    (0, 0), (-1, -1), 22),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]


class BioidPDFReport:
    """
    Générateur de rapport médico-légal PDF pour le module BioID AI de KANÉA.
    """

    DISCLAIMER_TEXT = (
        "AVERTISSEMENT MÉDICO-LÉGAL — Ce rapport a été généré automatiquement par le système "
        "BioID AI (KANÉA) à titre d'aide à l'expertise forensique. Les estimations biologiques "
        "fournies (sexe, âge, ascendance, stature) sont basées sur des modèles statistiques "
        "et des données anthropométriques. Elles ne constituent pas un diagnostic médical "
        "définitif et doivent être interprétées par un expert qualifié en anthropologie forensique. "
        "L'utilisation de ce rapport à des fins judiciaires ou médicales relève de la responsabilité "
        "exclusive de l'expert signataire. KANÉA / Côte d'Ivoire — Tous droits réservés."
    )

    def generate(
        self,
        case_data: dict[str, Any],
        predictions: dict[str, Any],
        charts: list[str],
        output_path: Path,
    ) -> str:
        """
        Génère le rapport PDF complet.

        Paramètres
        ----------
        case_data : dict
            Informations du dossier : case_id, examiner, timestamp, etc.
        predictions : dict
            Résultats des prédictions BioID.
        charts : list[str]
            Chemins des fichiers PNG à inclure.
        output_path : Path
            Chemin de destination du PDF.

        Retourne le chemin du PDF généré, ou "" si erreur.
        """
        if not _RL_OK:
            log.error("reportlab non disponible — impossible de générer le PDF")
            return ""

        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            styles   = _build_styles()
            story    = []
            doc_id   = str(uuid.uuid4()).upper()
            now      = datetime.now()
            ts_str   = now.strftime("%d/%m/%Y à %H:%M:%S")
            case_id  = str(case_data.get("case_id", "N/A"))
            examiner = str(case_data.get("examiner", "Expert KANÉA"))

            # ── EN-TÊTE ───────────────────────────────────────────────────────
            story.append(Spacer(1, 0.3 * cm))

            # Logo KANÉA (texte stylisé si pas d'image)
            story.append(Paragraph("● KANÉA", ParagraphStyle(
                "Logo", fontName="Helvetica-Bold", fontSize=28,
                textColor=_C.TEAL, alignment=TA_CENTER,
            )))
            story.append(Paragraph(
                "Plateforme d'Intelligence Artificielle Médicale — Côte d'Ivoire",
                styles["subtitle"],
            ))
            story.append(HRFlowable(width="100%", thickness=2, color=_C.TEAL, spaceAfter=6))

            story.append(Paragraph(
                "RAPPORT MÉDICO-LÉGAL — BIOID AI",
                styles["title"],
            ))
            story.append(Paragraph(
                "Estimation du Profil Biologique Forensique",
                styles["subtitle"],
            ))
            story.append(HRFlowable(width="100%", thickness=1, color=_C.LIGHT_GRAY, spaceAfter=10))
            story.append(Spacer(1, 0.4 * cm))

            # ── INFORMATIONS DU DOSSIER ───────────────────────────────────────
            story.append(Paragraph("1. Informations du Dossier", styles["section"]))

            dossier_data = [
                ["Champ", "Valeur"],
                ["Identifiant du dossier",    case_id],
                ["Examinateur",               examiner],
                ["Date et heure d'analyse",   ts_str],
                ["ID unique du rapport",       doc_id],
                ["Système",                   "BioID AI v2.0 — KANÉA"],
                ["Statut",                    str(case_data.get("status", "Analysé"))],
            ]
            t_dossier = Table(dossier_data, colWidths=[7 * cm, 11 * cm])
            t_style = _table_style_base()
            t_style.append(("ALIGN", (0, 1), (0, -1), "LEFT"))
            t_style.append(("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"))
            t_dossier.setStyle(TableStyle(t_style))
            story.append(t_dossier)
            story.append(Spacer(1, 0.5 * cm))

            # ── RÉSUMÉ MÉDICO-LÉGAL ───────────────────────────────────────────
            story.append(Paragraph("2. Résumé Médico-Légal", styles["section"]))

            preds = predictions
            sex          = str(preds.get("biological_sex", "Non déterminé"))
            sex_conf     = float(preds.get("sex_confidence", 0.0))
            age          = float(preds.get("age_at_death", 0))
            age_range    = str(preds.get("age_range", f"{max(0, age-8):.0f}–{age+8:.0f} ans"))
            ancestry     = str(preds.get("ancestry", "Non déterminé"))
            anc_conf     = float(preds.get("ancestry_confidence", 0.0))
            stature      = float(preds.get("stature_cm", 0))
            stat_method  = str(preds.get("stature_method", "Trotter & Gleser 1958"))

            summary_lines = [
                f"<b>Sexe biologique estimé :</b> {sex} (confiance : {sex_conf:.1%})",
                f"<b>Âge au décès estimé :</b> {age:.0f} ans — Intervalle : {age_range}",
                f"<b>Ascendance biogéographique :</b> {ancestry} (confiance : {anc_conf:.1%})",
                f"<b>Stature estimée :</b> {stature:.1f} cm — Méthode : {stat_method}" if stature > 0
                else "<b>Stature :</b> Données insuffisantes (mesures post-crâniennes manquantes)",
            ]

            # Encadré coloré
            summary_table_data = [[Paragraph(
                "<br/>".join(summary_lines),
                ParagraphStyle("SummaryBox", fontName="Helvetica", fontSize=10,
                               textColor=_C.DARK_GRAY, leading=16, spaceAfter=0),
            )]]
            summary_table = Table(summary_table_data, colWidths=[18 * cm])
            summary_table.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), _C.PALE_TEAL),
                ("BOX",           (0, 0), (-1, -1), 1.5, _C.TEAL),
                ("TOPPADDING",    (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("LEFTPADDING",   (0, 0), (-1, -1), 14),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 0.6 * cm))

            # ── TABLEAU DES PRÉDICTIONS DÉTAILLÉES ────────────────────────────
            story.append(Paragraph("3. Détail des Prédictions", styles["section"]))

            pred_table_data = [
                ["Paramètre", "Estimation", "Intervalle / Probabilité", "Méthode"],
                [
                    "Sexe biologique",
                    sex,
                    f"{sex_conf:.1%}",
                    "VotingClassifier RF+GB",
                ],
                [
                    "Âge au décès",
                    f"{age:.0f} ans",
                    age_range,
                    "VotingRegressor RF+GB+EN",
                ],
                [
                    "Ascendance",
                    ancestry,
                    f"{anc_conf:.1%}",
                    "Random Forest + PCA AIMs",
                ],
                [
                    "Stature",
                    f"{stature:.1f} cm" if stature > 0 else "N/D",
                    f"±5 cm (95% IC)" if stature > 0 else "—",
                    stat_method if stature > 0 else "Données insuffisantes",
                ],
            ]

            t_preds = Table(pred_table_data, colWidths=[4.5 * cm, 3.5 * cm, 5 * cm, 5 * cm])
            t_preds.setStyle(TableStyle(_table_style_base()))
            story.append(t_preds)
            story.append(Spacer(1, 0.6 * cm))

            # ── GRAPHIQUES ────────────────────────────────────────────────────
            valid_charts = [c for c in charts if c and Path(c).exists()]
            if valid_charts:
                story.append(Paragraph("4. Visualisations Analytiques", styles["section"]))

                for i, chart_path in enumerate(valid_charts):
                    try:
                        chart_img = Image(chart_path)
                        # Redimensionner pour tenir sur la page
                        max_w = 16 * cm
                        max_h = 12 * cm
                        w, h = chart_img.drawWidth, chart_img.drawHeight
                        ratio = min(max_w / w, max_h / h)
                        chart_img.drawWidth  = w * ratio
                        chart_img.drawHeight = h * ratio
                        story.append(KeepTogether([chart_img, Spacer(1, 0.4 * cm)]))
                    except Exception as e:
                        log.warning("Impossible d'inclure le graphique %s : %s", chart_path, e)

            # ── QR CODE ───────────────────────────────────────────────────────
            story.append(Spacer(1, 0.5 * cm))
            story.append(HRFlowable(width="100%", thickness=1, color=_C.LIGHT_GRAY))
            story.append(Spacer(1, 0.3 * cm))

            qr_data = (
                f"KANEA-BIOID|case={case_id}|id={doc_id}"
                f"|sex={sex}|age={age:.0f}|ancestry={ancestry}"
            )
            qr_img = _make_qr_image(qr_data, size_cm=3.5)

            qr_section = [
                Paragraph("Vérification du Rapport", styles["section"]),
            ]
            if qr_img:
                qr_table = Table(
                    [[qr_img, Paragraph(
                        f"<b>ID Rapport :</b> {doc_id}<br/>"
                        f"<b>Dossier :</b> {case_id}<br/>"
                        f"<b>Généré le :</b> {ts_str}<br/>"
                        f"<b>Système :</b> BioID AI v2.0 — KANÉA",
                        styles["body"],
                    )]],
                    colWidths=[4 * cm, 14 * cm],
                )
                qr_table.setStyle(TableStyle([
                    ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]))
                qr_section.append(qr_table)
            else:
                qr_section.append(Paragraph(
                    f"ID rapport : {doc_id} | Dossier : {case_id} | {ts_str}",
                    styles["body"],
                ))

            story.extend(qr_section)
            story.append(Spacer(1, 0.5 * cm))

            # ── SIGNATURE NUMÉRIQUE ───────────────────────────────────────────
            sig_content = (
                f"{case_id}|{doc_id}|{ts_str}|{sex}|{age}|{ancestry}|{stature}"
            )
            sig_hash = _hash_content(sig_content)
            story.append(Paragraph(
                f"<b>Signature numérique SHA-256 :</b>",
                styles["body"],
            ))
            story.append(Paragraph(sig_hash, styles["small"]))
            story.append(Spacer(1, 0.6 * cm))

            # ── DISCLAIMER ────────────────────────────────────────────────────
            story.append(HRFlowable(width="100%", thickness=1, color=_C.LIGHT_GRAY))
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph("Avertissement légal", styles["section"]))
            story.append(Paragraph(self.DISCLAIMER_TEXT, styles["disclaimer"]))
            story.append(Spacer(1, 1 * cm))

            # ── CONSTRUCTION DU PDF ───────────────────────────────────────────
            page_w, page_h = A4
            total_pages = [0]  # mutable pour les callbacks

            def on_first_page(canvas, doc):
                total_pages[0] = doc.page
                _draw_footer(canvas, doc, styles)

            def on_later_pages(canvas, doc):
                _draw_footer(canvas, doc, styles)

            pdf = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=2 * cm,
                leftMargin=2 * cm,
                topMargin=2 * cm,
                bottomMargin=2.5 * cm,
                title=f"BioID AI — Dossier {case_id}",
                author="KANÉA BioID AI",
                subject="Rapport médico-légal forensique",
            )
            pdf.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)

            log.info("PDF généré : %s", output_path)
            return str(output_path)

        except Exception as exc:
            log.error("BioidPDFReport.generate error: %s", exc, exc_info=True)
            return ""


def _draw_footer(canvas: Any, doc: Any, styles: dict) -> None:
    """Dessine le pied de page sur chaque page."""
    if not _RL_OK:
        return
    try:
        page_w, page_h = A4
        canvas.saveState()

        # Ligne de séparation
        canvas.setStrokeColor(_C.LIGHT_GRAY)
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, 1.8 * cm, page_w - 2 * cm, 1.8 * cm)

        # Texte du pied de page
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_C.MED_GRAY)
        canvas.drawString(
            2 * cm, 1.3 * cm,
            "KANÉA — Plateforme IA Médicale | BioID AI v2.0 | Confidentiel"
        )
        canvas.drawRightString(
            page_w - 2 * cm, 1.3 * cm,
            f"Page {doc.page}"
        )
        canvas.restoreState()
    except Exception as exc:
        log.warning("_draw_footer error: %s", exc)


# ── Fonction utilitaire de haut niveau ───────────────────────────────────────
def generate_bioid_report(
    case_data: dict[str, Any],
    predictions: dict[str, Any],
    charts: list[str],
    output_dir: Optional[Path] = None,
) -> str:
    """
    Génère un rapport PDF BioID complet.

    Fonction de haut niveau wrappant BioidPDFReport.generate().
    Retourne le chemin du PDF ou "" si erreur.
    """
    try:
        if output_dir is None:
            output_dir = Path("reports")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        case_id   = str(case_data.get("case_id", "case_unknown"))
        ts_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"bioid_report_{case_id}_{ts_suffix}.pdf"
        output_path = output_dir / filename

        generator = BioidPDFReport()
        return generator.generate(case_data, predictions, charts, output_path)

    except Exception as exc:
        log.error("generate_bioid_report error: %s", exc)
        return ""
