"""
DermAI — Rapport PDF médical dermatologique A4
===============================================
Contient : diagnostic IA, ABCDE, Breslow, TNM, Grad-CAM, stade, recommandations.
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
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        Image as RLImage,
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

_TEAL  = colors.HexColor("#20B2AA") if _RL_OK else None
_RED   = colors.HexColor("#C0392B") if _RL_OK else None
_GREEN = colors.HexColor("#27AE60") if _RL_OK else None
_DARK  = colors.HexColor("#1A2B3C") if _RL_OK else None
_MUTED = colors.HexColor("#5E7A8A") if _RL_OK else None
_LIGHT = colors.HexColor("#EEF2F6") if _RL_OK else None
_WHITE = colors.white               if _RL_OK else None

_URG_HEX = {"Critique":"C0392B","Urgence":"922B21","Élevée":"E74C3C","Modérée":"E67E22","Faible":"27AE60"}
_URG_RL  = {"Critique":colors.HexColor("#C0392B") if _RL_OK else None,
             "Urgence": colors.HexColor("#922B21") if _RL_OK else None,
             "Élevée":  colors.HexColor("#E74C3C") if _RL_OK else None,
             "Modérée": colors.HexColor("#E67E22") if _RL_OK else None,
             "Faible":  colors.HexColor("#27AE60") if _RL_OK else None}


def _s(name, **kw):
    return ParagraphStyle(name, **kw)


def _qr(text, size=55):
    if not _QR_OK or not _RL_OK: return None
    try:
        qr = _qrc.QRCode(box_size=2, border=2)
        qr.add_data(text); qr.make(fit=True)
        img = qr.make_image(); buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        return RLImage(buf, width=size, height=size)
    except Exception: return None


def _b64img(b64, w=180, h=180):
    if not b64 or not _RL_OK: return None
    try:
        buf = io.BytesIO(base64.b64decode(b64))
        return RLImage(buf, width=w, height=h)
    except Exception: return None


def build_derm_pdf_report(
    result: dict[str, Any],
    patient_id: str = "KANEA-AUTO",
    examiner: str = "KANÉA System",
    institution: str = "Dermatologie — KANÉA",
) -> bytes | None:
    if not _RL_OK: return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=14*mm, bottomMargin=14*mm,
                            leftMargin=17*mm, rightMargin=17*mm)
    W = A4[0] - 34*mm

    s_h1 = _s("H1", fontSize=14, fontName="Helvetica-Bold", textColor=_WHITE, leading=17)
    s_h2 = _s("H2", fontSize=10.5, fontName="Helvetica-Bold", textColor=_DARK, leading=13, spaceAfter=3)
    s_b  = _s("B", fontSize=8.5, fontName="Helvetica", textColor=_DARK, leading=11)
    s_m  = _s("M", fontSize=7.5, fontName="Helvetica", textColor=_MUTED, leading=10)

    story: list[Any] = []
    pred     = result.get("prediction","—")
    conf     = result.get("confidence", 0)
    urgency  = result.get("clinical_profile",{}).get("urgency","—")
    urg_hex  = _URG_HEX.get(urgency,"5E7A8A")
    urg_rl   = _URG_RL.get(urgency, _MUTED)
    action   = result.get("recommended_action","—")
    request_id = result.get("request_id", str(uuid.uuid4()))[:16]
    date_str = datetime.now().strftime("%d/%m/%Y à %H:%M")

    qr = _qr(f"KANEA-DERM-{request_id}", 55)
    hdr_left = Paragraph(
        f'<font size="15"><b>KANÉA — DermAI v2.0</b></font><br/>'
        f'<font size="8.5">Rapport dermatologique IA — 25 pathologies cutanées</font>', s_h1)
    header_data = [[hdr_left, qr if qr else Paragraph(f'<font size="7.5">{request_id}<br/>{date_str}</font>',
                   _s("HR",fontSize=7.5,fontName="Helvetica",textColor=_WHITE,alignment=TA_RIGHT,leading=10))]]
    hdr_t = Table(header_data, colWidths=[W*0.72, W*0.28])
    hdr_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),_DARK),
                               ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
                               ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),8),
                               ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(hdr_t); story.append(Spacer(1,4*mm))

    # 1. Renseignements
    story.append(Paragraph("1. Renseignements du dossier", s_h2))
    info_t = Table([
        ["N° dossier",patient_id,"Date",date_str],
        ["Examinateur",examiner,"Institution",institution],
        ["Modèle IA","EfficientNet-B4 — ISIC+HAM10000","Version","v2.0"],
    ], colWidths=[W*0.16,W*0.34,W*0.14,W*0.36])
    info_t.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),8),
                                ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
                                ("TEXTCOLOR",(0,0),(0,-1),_MUTED),("TEXTCOLOR",(2,0),(2,-1),_MUTED),
                                ("ROWBACKGROUNDS",(0,0),(-1,-1),[_LIGHT,_WHITE]),
                                ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#DDE8EE")),
                                ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
                                ("LEFTPADDING",(0,0),(-1,-1),6)]))
    story.append(info_t); story.append(Spacer(1,4*mm))

    # 2. Résultat principal
    story.append(Paragraph("2. Résultat de l'analyse IA", s_h2))
    profile = result.get("clinical_profile",{})
    pred_data = [
        [Paragraph("<b>Diagnostic IA</b>",s_m), Paragraph(f"<b><font size='12' color='#{urg_hex}'>{pred}</font></b>",s_b),
         Paragraph("<b>Confiance</b>",s_m), Paragraph(f"<b><font size='11' color='#{urg_hex}'>{conf:.1%}</font></b>",s_b)],
        [Paragraph("<b>Urgence</b>",s_m), Paragraph(f"<b><font color='#{urg_hex}'>{urgency}</font></b>",s_b),
         Paragraph("<b>CIM-10</b>",s_m), Paragraph(profile.get("icd10","—"),s_b)],
        [Paragraph("<b>Catégorie</b>",s_m), Paragraph(profile.get("category","—"),s_b),
         Paragraph("<b>Dermoscopie</b>",s_m), Paragraph(f'<font size="7.5">{profile.get("dermoscopy","—")[:60]}</font>',s_m)],
        [Paragraph("<b>Action</b>",s_m), Paragraph(f'<font size="8">{action[:80]}</font>',s_b),
         Paragraph("<b>Guidelines</b>",s_m), Paragraph(f'<font size="7.5">{profile.get("guidelines","—")[:50]}</font>',s_m)],
    ]
    pred_t = Table(pred_data, colWidths=[W*0.14,W*0.36,W*0.14,W*0.36])
    pred_t.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),8.5),
                                ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#F7F9FC"),_WHITE]),
                                ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#DDE8EE")),
                                ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
                                ("LEFTPADDING",(0,0),(-1,-1),7),("LINEBEFORE",(0,0),(0,-1),4,urg_rl)]))
    story.append(pred_t); story.append(Spacer(1,4*mm))

    # 3. Scores cliniques
    scores = result.get("clinical_scores",{})
    abcde = scores.get("abcde",{})
    breslow = scores.get("breslow_clark",{})
    tnm = scores.get("tnm",{})
    if abcde or breslow or tnm:
        story.append(Paragraph("3. Scores cliniques dermatologiques", s_h2))
        sc_rows = [["Score","Résultat","Interprétation","Action"]]
        if abcde:
            sc_rows.append(["ABCDE",f"TDS {abcde.get('total_score','—')}",
                            f"{abcde.get('risk','—')} — {abcde.get('probability','—')}",
                            abcde.get('recommendation','—')[:60]])
        if breslow:
            sc_rows.append(["Breslow/Clark",f"{breslow.get('breslow_mm','—')} mm ({breslow.get('t_stage','—')})",
                            f"{breslow.get('category','—')} — Clark {breslow.get('clark_level','—')}",
                            f"Survie 5 ans {breslow.get('survival','—')} — Marges {breslow.get('margins_cm','—')} cm"])
        if tnm:
            sc_rows.append(["TNM (AJCC 8e)",f"T{tnm.get('T','?')} N{tnm.get('N','?')} M{tnm.get('M','?')}",
                            f"Stade {tnm.get('stage','—')} — Survie {tnm.get('survival','—')}",
                            tnm.get('treatment','—')[:60]])
        sc_t = Table(sc_rows, colWidths=[W*0.14,W*0.22,W*0.34,W*0.30])
        sc_t.setStyle(TableStyle([("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.5),
                                  ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#D91E7A")),
                                  ("TEXTCOLOR",(0,0),(-1,0),_WHITE),
                                  ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LIGHT,_WHITE]),
                                  ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#DDE8EE")),
                                  ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
                                  ("LEFTPADDING",(0,0),(-1,-1),5),("WORDWRAP",(0,0),(-1,-1),True)]))
        story.append(sc_t); story.append(Spacer(1,4*mm))

    # 4. Grad-CAM
    heatmap_b64 = result.get("explainability",{}).get("heatmap_b64")
    if heatmap_b64:
        story.append(Paragraph("4. Explainability — Grad-CAM dermoscopique", s_h2))
        hm_img = _b64img(heatmap_b64, 175, 175)
        if hm_img:
            hm_t = Table([[hm_img,
                           Paragraph(f"<b>Méthode :</b> Gradient-weighted CAM (EfficientNet-B4)<br/>"
                                     f"<b>Lésion :</b> {pred}<br/>"
                                     f"<b>Pattern :</b> {profile.get('dermoscopy','—')}<br/><br/>"
                                     f"<i>La heatmap rouge localise les zones dermoscopiques "
                                     f"ayant le plus contribué au diagnostic.</i>", s_b)]],
                         colWidths=[180, W-180])
            hm_t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(1,0),(1,0),10)]))
            story.append(hm_t); story.append(Spacer(1,4*mm))

    # 5. Différentiel
    diff = result.get("differential_diagnosis",[])
    if diff:
        story.append(Paragraph("5. Diagnostic différentiel", s_h2))
        diff_rows = [["Classe","Probabilité","CIM-10","Catégorie"]] + [
            [d["class"],f"{d['probability']:.1%}",d.get("icd10","—"),d.get("category","—")]
            for d in diff]
        diff_t = Table(diff_rows, colWidths=[W*0.40,W*0.15,W*0.15,W*0.30])
        diff_t.setStyle(TableStyle([("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
                                    ("BACKGROUND",(0,0),(-1,0),_TEAL),("TEXTCOLOR",(0,0),(-1,0),_WHITE),
                                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LIGHT,_WHITE]),
                                    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#DDE8EE")),
                                    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
                                    ("LEFTPADDING",(0,0),(-1,-1),6)]))
        story.append(diff_t); story.append(Spacer(1,4*mm))

    # 6. Quantification
    quant = result.get("quantification",{})
    if quant:
        story.append(Paragraph("6. Quantification de la lésion", s_h2))
        q_rows = [["Paramètre","Valeur","Interprétation"],
                  ["Surface estimée atteinte",f"{quant.get('lesion_coverage_pct',0):.1f}%","< 1 cm² faible · > 5 cm² étendue"],
                  ["Index d'asymétrie",f"{quant.get('asymmetry_index',0):.3f}","< 0.05 = symétrique · > 0.10 = asymétrique"],
                  ["Hétérogénéité couleur",f"{quant.get('color_heterogeneity',0):.3f}","< 0.10 = uniforme · > 0.20 = polychrome"],
                  ["Ratio zones sombres",f"{quant.get('dark_zone_ratio',0):.3f}","Indicateur mélanisation"]]
        q_t = Table(q_rows, colWidths=[W*0.30,W*0.20,W*0.50])
        q_t.setStyle(TableStyle([("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
                                 ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#8E44AD")),("TEXTCOLOR",(0,0),(-1,0),_WHITE),
                                 ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LIGHT,_WHITE]),
                                 ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#DDE8EE")),
                                 ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
                                 ("LEFTPADDING",(0,0),(-1,-1),6)]))
        story.append(q_t); story.append(Spacer(1,4*mm))

    # Safety + footer
    safety = result.get("clinical_safety",{})
    if safety.get("level") in ("warning","critical"):
        a_col = colors.HexColor("#922B21") if safety["level"]=="critical" else colors.HexColor("#E67E22")
        a_bg  = colors.HexColor("#FDECEA") if safety["level"]=="critical" else colors.HexColor("#FEF0E7")
        al_t  = Table([[Paragraph(f"⚠ {safety.get('message','')}",s_b)]],colWidths=[W])
        al_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),a_bg),("LINEBEFORE",(0,0),(-1,-1),4,a_col),
                                  ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
                                  ("LEFTPADDING",(0,0),(-1,-1),10)]))
        story.append(al_t); story.append(Spacer(1,3*mm))

    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#DDE8EE")))
    story.append(Spacer(1,2*mm))
    story.append(Paragraph("⚠ Outil d'aide à la décision médicale — ne remplace pas le diagnostic d'un dermatologue. "
                           "Toute lésion suspecte doit être évaluée par un spécialiste. DermAI v2.0 — KANÉA 2026 — "
                           "EfficientNet-B4 — ISIC 2020 + HAM10000 + BCN20000.", s_m))
    doc.build(story)
    return buf.getvalue()


def build_derm_html_report(result: dict[str, Any], patient_id: str = "KANEA") -> str:
    pred = result.get("prediction","—"); conf = result.get("confidence",0)
    urgency = result.get("clinical_profile",{}).get("urgency","—")
    action = result.get("recommended_action","—")
    urg_hex = "#" + _URG_HEX.get(urgency,"5E7A8A")
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    diff = result.get("differential_diagnosis",[])
    diff_rows = "".join(f"<tr><td>{d['class']}</td><td>{d['probability']:.1%}</td><td>{d.get('icd10','—')}</td></tr>" for d in diff)
    return f"""<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8"><title>DermAI — {patient_id}</title>
<style>body{{font-family:Arial,sans-serif;font-size:11px;color:#1A2B3C;margin:0;padding:12px}}
h1{{background:#1A2B3C;color:#fff;padding:8px 12px;font-size:13px}}
h2{{font-size:10px;color:#D91E7A;text-transform:uppercase;margin:10px 0 4px}}
table{{width:100%;border-collapse:collapse;margin-bottom:6px}}
th{{background:#D91E7A;color:#fff;padding:3px 6px;font-size:9px;text-align:left}}
td{{padding:3px 6px;border:1px solid #DDE8EE;font-size:9px}}
tr:nth-child(even){{background:#F7F9FC}}
.disc{{background:#FFF8E1;border-left:4px solid #F39C12;padding:6px 10px;font-size:8px}}</style>
</head><body>
<h1>KANÉA — DermAI v2.0 · Rapport dermatologique</h1>
<p style="background:#F0F4F8;padding:5px 10px;margin:0;font-size:9px">Dossier : {patient_id} · {date_str}</p>
<h2>Résultat IA</h2>
<p><b style="color:{urg_hex};font-size:13px">{pred}</b>
&nbsp; Confiance : <b>{conf:.1%}</b> &nbsp; Urgence : <b style="color:{urg_hex}">{urgency}</b></p>
<p><b>Action :</b> {action}</p>
<h2>Diagnostic différentiel</h2>
<table><tr><th>Classe</th><th>Probabilité</th><th>CIM-10</th></tr>{diff_rows}</table>
<div class="disc">⚠ Outil d'aide à la décision — validation dermatologique obligatoire. KANÉA 2026.</div>
</body></html>"""
