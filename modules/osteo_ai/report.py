"""
OsteoDetect AI — Rapport PDF rhumatologique A4 multi-pages
===========================================================
Kellgren-Lawrence · WOMAC · DAS28 · CDAI/SDAI · T-Score · FRAX ·
BASDAI · BASFI · ASDAS · findings critiques · SHAP · QR code.
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

_BROWN  = colors.HexColor("#795548") if _RL_OK else None
_ORANGE = colors.HexColor("#E67E22") if _RL_OK else None
_RED    = colors.HexColor("#C0392B") if _RL_OK else None
_CRIT   = colors.HexColor("#922B21") if _RL_OK else None
_GREEN  = colors.HexColor("#27AE60") if _RL_OK else None
_BLUE   = colors.HexColor("#2980B9") if _RL_OK else None
_PURPLE = colors.HexColor("#8E44AD") if _RL_OK else None
_DARK   = colors.HexColor("#1A2B3C") if _RL_OK else None
_MUTED  = colors.HexColor("#5E7A8A") if _RL_OK else None
_LIGHT  = colors.HexColor("#EEF2F6") if _RL_OK else None
_WHITE  = colors.white               if _RL_OK else None

_URG_RL  = {"Critique":_CRIT,"Élevée":colors.HexColor("#E74C3C") if _RL_OK else None,
             "Modérée":_ORANGE,"Faible":_GREEN}
_URG_HEX = {"Critique":"922B21","Élevée":"E74C3C","Modérée":"E67E22","Faible":"27AE60"}


def _s(n, **kw): return ParagraphStyle(n, **kw)
def _qr(text, size=55):
    if not (_QR_OK and _RL_OK): return None
    try:
        from reportlab.platypus import Image as RLI
        q = _qrc.QRCode(box_size=2, border=2)
        q.add_data(text); q.make(fit=True)
        img = q.make_image(); buf = io.BytesIO(); img.save(buf,"PNG"); buf.seek(0)
        return RLI(buf, width=size, height=size)
    except Exception: return None


def build_osteo_pdf_report(
    result: dict[str, Any],
    patient_id: str = "KANEA-OS",
    examiner: str = "KANÉA System",
    institution: str = "Rhumatologie — KANÉA",
    patient_name: str = "",
    patient_age: str = "",
    patient_sex: str = "",
    clinical_context: str = "",
) -> bytes | None:
    if not _RL_OK: return None

    buf = io.BytesIO()
    W, H = A4
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm,
        title=f"OsteoDetect AI — {patient_id}", author="KANÉA Medical AI Platform")
    story = []; pw = W - 36*mm

    S_title = _s("t",  fontSize=17, textColor=_BROWN, fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=2)
    S_sub   = _s("su", fontSize=9,  textColor=_MUTED, fontName="Helvetica", alignment=TA_CENTER, spaceAfter=6)
    S_h1    = _s("h1", fontSize=12, textColor=_DARK,  fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
    S_h2    = _s("h2", fontSize=10, textColor=_BROWN, fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=3)
    S_body  = _s("bd", fontSize=8,  textColor=_DARK,  fontName="Helvetica", spaceAfter=3, leading=12)
    S_small = _s("sm", fontSize=7,  textColor=_MUTED, fontName="Helvetica", spaceAfter=2)
    S_warn  = _s("wn", fontSize=8,  textColor=_RED,   fontName="Helvetica-Bold", spaceAfter=3)

    def _tbl(data, cws, extra=None):
        cmds = [("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),8),
                ("ROWBACKGROUNDS",(0,0),(-1,-1),[_LIGHT,_WHITE]),
                ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#D5DCE8")),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),5),
                ("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),3),
                ("BOTTOMPADDING",(0,0),(-1,-1),3)]
        if extra: cmds.extend(extra)
        return Table(data, colWidths=cws, style=TableStyle(cmds))

    def _htbl(data, cws):
        cmds = [("BACKGROUND",(0,0),(-1,0),_BROWN),("TEXTCOLOR",(0,0),(-1,0),_WHITE),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LIGHT,_WHITE]),
                ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#D5DCE8")),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),5),
                ("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),3),
                ("BOTTOMPADDING",(0,0),(-1,-1),3)]
        return Table(data, colWidths=cws, style=TableStyle(cmds))

    now_str    = datetime.now().strftime("%d/%m/%Y à %H:%M")
    report_id  = str(uuid.uuid4())[:8].upper()
    prediction = result.get("prediction","—")
    confidence = result.get("confidence", 0.0)
    profile    = result.get("clinical_profile",{})
    urgency    = profile.get("urgency","—")
    action     = profile.get("action","—")
    urg_color  = _URG_RL.get(urgency, _ORANGE)
    urg_hex    = _URG_HEX.get(urgency,"E67E22")
    scores     = result.get("clinical_scores",{})
    critical   = result.get("critical_findings",[])
    feat_imp   = result.get("explainability",{}).get("feature_importance",{})
    summary    = result.get("clinical_summary",{})
    safety     = result.get("clinical_safety",{})
    frax_risk  = profile.get("fracture_risk_pct","—")

    # ── PAGE 1 ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("🦴 OsteoDetect AI — Rapport Rhumatologique", S_title))
    story.append(Paragraph(f"KANÉA Medical AI Platform | {institution} | {now_str} | ID {report_id}", S_sub))
    story.append(HRFlowable(width=pw, thickness=2, color=_BROWN, spaceAfter=6))

    story.append(Paragraph("Informations Patient", S_h1))
    story.append(_tbl([
        ["ID Patient",patient_id or "—","Examinateur",examiner],
        ["Nom / Prénom",patient_name or "—","Âge",patient_age or "—"],
        ["Sexe",patient_sex or "—","Contexte",clinical_context or "—"],
        ["Date rapport",now_str,"Version IA",result.get("model_version","v1.0")],
    ],[pw*.20,pw*.30,pw*.20,pw*.30],extra=[
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
        ("TEXTCOLOR",(0,0),(0,-1),_BROWN),("TEXTCOLOR",(2,0),(2,-1),_BROWN)]))
    story.append(Spacer(1,6))

    # Bandeau diagnostic
    story.append(Table([[
        Paragraph(f'<font color="#{urg_hex}"><b>DIAGNOSTIC : {prediction}</b></font>',
                  _s("d1",fontSize=11,fontName="Helvetica-Bold",textColor=_DARK)),
        Paragraph(f'<b>Confiance : {confidence:.1%}</b><br/>'
                  f'Urgence : <font color="#{urg_hex}"><b>{urgency}</b></font><br/>'
                  f'Risque fracture 10 ans : <b>{frax_risk}%</b>',
                  _s("d2",fontSize=9,fontName="Helvetica",textColor=_DARK,alignment=TA_RIGHT)),
    ]],[pw*.65,pw*.35],style=TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor(f"#{urg_hex}20")),
        ("BOX",(0,0),(-1,-1),1.5,urg_color or _BROWN),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE")])))
    story.append(Spacer(1,3))
    story.append(Paragraph(f"<b>Action recommandée :</b> {action}", S_body))

    if safety.get("level") in ("warning","critical"):
        hx = "C0392B" if safety["level"] == "critical" else "E67E22"
        story.append(Paragraph(f'<font color="#{hx}"><b>⚠ {safety.get("message","")}</b></font>',S_warn))

    story.append(HRFlowable(width=pw,thickness=0.5,color=_MUTED,spaceBefore=4,spaceAfter=4))

    # Résumé clinique
    story.append(Paragraph("Résumé Clinique Ostéo-Articulaire", S_h2))
    sr = [["Paramètre","Valeur"]]
    for k_s,v_s in summary.items():
        sr.append([k_s,str(v_s)])
    story.append(_htbl(sr,[pw*.45,pw*.55]))
    story.append(Spacer(1,5))

    # Probabilités
    story.append(Paragraph("Distribution des Probabilités Diagnostiques", S_h2))
    probs = result.get("probabilities",{})
    if probs:
        pd = [["Pathologie","Probabilité","Barre"]]
        for cls,p in sorted(probs.items(),key=lambda x:-x[1])[:10]:
            bar="█"*int(p*26)+"░"*(26-int(p*26))
            pd.append([cls,f"{p:.1%}",bar[:20]])
        story.append(_htbl(pd,[pw*.52,pw*.14,pw*.34]))

    story.append(PageBreak())

    # ── PAGE 2 — SCORES ───────────────────────────────────────────────────────
    story.append(Paragraph("📊 Scores Rhumatologiques & Ostéo-Articulaires", S_h1))
    story.append(HRFlowable(width=pw,thickness=1,color=_BROWN,spaceAfter=5))

    def _render_score(title, sd, color=None):
        if not sd: return
        c = color or _BROWN
        story.append(Paragraph(title,_s("sh",fontSize=9,fontName="Helvetica-Bold",
                                         textColor=c,spaceBefore=5,spaceAfter=2)))
        skip = {"interpretation","recommendation","radiological_features"}
        rows = []
        for k_r,v_r in sd.items():
            if k_r in skip or v_r is None: continue
            label = k_r.replace("_"," ").title()
            if isinstance(v_r,bool):   val="Oui" if v_r else "Non"
            elif isinstance(v_r,list): val=", ".join(str(x) for x in v_r) if v_r else "—"
            elif isinstance(v_r,float):val=f"{v_r:.3f}" if v_r!=int(v_r) else str(int(v_r))
            else: val=str(v_r)
            rows.append([label,val])
        if rows:
            story.append(_tbl(rows,[pw*.45,pw*.55],extra=[
                ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
                ("TEXTCOLOR",(0,0),(0,-1),_DARK)]))
        rf = sd.get("radiological_features",[])
        if rf:
            story.append(Paragraph("<b>Features radiologiques :</b> "+", ".join(rf),S_small))
        interp = sd.get("interpretation","")
        reco   = sd.get("recommendation","")
        if interp: story.append(Paragraph(f"<i>{interp}</i>",S_small))
        if reco:   story.append(Paragraph(f"<b>Recommandation :</b> {reco}",S_small))
        story.append(Spacer(1,4))

    _render_score("Kellgren-Lawrence — Arthrose radiologique", scores.get("kellgren_lawrence",{}), _BROWN)
    _render_score("WOMAC Index — Arthrose fonctionnelle",      scores.get("womac",{}),             _ORANGE)
    _render_score("DAS28-CRP — Polyarthrite Rhumatoïde",       scores.get("das28",{}),             _PURPLE)
    _render_score("CDAI / SDAI — Activité PR",                 scores.get("cdai_sdai",{}),         _PURPLE)
    _render_score("T-Score DXA — Ostéoporose",                 scores.get("t_score",{}),           _BLUE)
    _render_score("FRAX — Risque fracturaire 10 ans (WHO)",    scores.get("frax",{}),              _RED)
    _render_score("BASDAI — Spondylarthrite Ankylosante",      scores.get("basdai",{}),            _BLUE)
    _render_score("BASFI — Fonction SPA",                      scores.get("basfi",{}),             _BLUE)
    _render_score("ASDAS-CRP — Activité SPA",                  scores.get("asdas",{}),             _BLUE)

    story.append(PageBreak())

    # ── PAGE 3 — FINDINGS + SHAP ───────────────────────────────────────────────
    story.append(Paragraph("⚠ Findings Critiques & Explainability", S_h1))
    story.append(HRFlowable(width=pw,thickness=1,color=_BROWN,spaceAfter=5))

    story.append(Paragraph("Findings Ostéo-Articulaires Critiques", S_h2))
    if critical:
        for c in critical:
            sev=c.get("severity","INFO"); hx=c.get("color","#E67E22").replace("#","")
            story.append(Paragraph(
                f'<font color="#{hx}"><b>[{sev}]</b></font> <b>{c.get("finding","—")}</b> — {c.get("detail","—")}',
                S_body))
        story.append(Spacer(1,5))
    else:
        story.append(Paragraph('<font color="#27AE60"><b>Aucun finding critique détecté.</b></font>',S_body))
        story.append(Spacer(1,4))

    story.append(HRFlowable(width=pw,thickness=0.5,color=_MUTED,spaceAfter=4))

    story.append(Paragraph("🤖 Explainability — Feature Importance (SHAP-inspired)", S_h2))
    if feat_imp:
        fi_data=[["Variable","Importance (%)","Contribution"]]
        for fk,sc in sorted(feat_imp.items(),key=lambda x:-x[1])[:14]:
            bar="█"*int(sc/100*22)+"░"*(22-int(sc/100*22))
            fi_data.append([fk,f"{sc:.1f}%",bar[:20]])
        story.append(_htbl(fi_data,[pw*.48,pw*.17,pw*.35]))
        story.append(Spacer(1,4))
    top5=result.get("explainability",{}).get("top_5_drivers",[])
    if top5:
        story.append(Paragraph(
            "<b>Top 5 variables décisives :</b> "+" · ".join(f"<b>{k}</b> ({v:.0f}%)" for k,v in top5),S_body))

    story.append(PageBreak())

    # ── PAGE 4 — RECOMMANDATIONS + QR ─────────────────────────────────────────
    story.append(Paragraph("📋 Recommandations Cliniques & Guidelines", S_h1))
    story.append(HRFlowable(width=pw,thickness=1,color=_BROWN,spaceAfter=5))

    story.append(Paragraph("Recommandations Issues des Scores", S_h2))
    rr=[["Score","Recommandation"]]
    for sk,sv in scores.items():
        if isinstance(sv,dict) and sv.get("recommendation"):
            rr.append([sk.replace("_"," ").upper(),sv["recommendation"]])
    if len(rr)>1: story.append(_htbl(rr,[pw*.20,pw*.80]))
    story.append(Spacer(1,6))

    story.append(Paragraph("Action Principale Recommandée", S_h2))
    story.append(Table([[
        Paragraph(f'<font color="#{urg_hex}"><b>{urgency.upper()}</b></font> — {action}',
                  _s("act",fontSize=9,fontName="Helvetica",textColor=_DARK))
    ]],[pw],style=TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor(f"#{urg_hex}18")),
        ("BOX",(0,0),(-1,-1),1,urg_color or _BROWN),
        ("LEFTPADDING",(0,0),(-1,-1),10),("TOPPADDING",(0,0),(-1,-1),6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6)])))
    story.append(Spacer(1,8))

    story.append(Paragraph("Références & Guidelines", S_h2))
    story.append(Paragraph(result.get("guidelines_ref",""),S_small))
    story.append(Spacer(1,5))

    story.append(Paragraph("Sources Bibliographiques", S_h2))
    for src in [
        "ACR/EULAR Rheumatoid Arthritis Classification Criteria 2010 (Aletaha et al., Arthritis Rheum)",
        "EULAR Recommendations for RA management 2022 (Smolen et al., Ann Rheum Dis)",
        "ASAS/EULAR Recommendations for SpA 2022 (van der Heijde et al.)",
        "IOF/ESCEO/ECTS Osteoporosis Guidelines 2023",
        "FRAX — Kanis et al., JBMR 2008 · WHO Collaborating Centre Sheffield",
        "Kellgren & Lawrence Grading — Ann Rheum Dis 1957",
        "WOMAC — Bellamy et al., J Rheumatol 1988",
        "DAS28-CRP — van der Heijde et al., 1995 · Prevoo et al., 2003",
        "BASDAI — Garrett et al., J Rheumatol 1994",
        "ASDAS-CRP — Lukas et al., Ann Rheum Dis 2009",
        "EWGSOP2 Sarcopénie — Cruz-Jentoft et al., Age Ageing 2019",
    ]:
        story.append(Paragraph(f"• {src}",S_small))
    story.append(Spacer(1,8))

    story.append(HRFlowable(width=pw,thickness=0.5,color=_MUTED,spaceAfter=4))
    story.append(Paragraph(
        "<b>⚠ Disclaimer médical :</b> Ce rapport est généré par KANÉA OsteoDetect AI v1.0 "
        "à titre d'aide au diagnostic rhumatologique. Il ne remplace pas l'expertise d'un "
        "rhumatologue ou orthopédiste. Toute décision thérapeutique (biothérapie, chirurgie, "
        "bisphosphonate) doit être validée par un médecin qualifié. "
        "Conforme aux recommandations ACR/EULAR/IOF 2022–2023.",
        _s("disc",fontSize=7,textColor=_MUTED,fontName="Helvetica",spaceAfter=6,leading=10)))

    qr_img = _qr(f"KANEA-Osteo|ID:{report_id}|Patient:{patient_id}|Prediction:{prediction}|Urgence:{urgency}", 60)
    story.append(Table([[
        Paragraph(f"<b>KANÉA OsteoDetect AI</b><br/>Rapport ID : {report_id}<br/>"
                  f"Généré le : {now_str}<br/>Patient : {patient_id}<br/>Examinateur : {examiner}",
                  _s("sig",fontSize=8,fontName="Helvetica",textColor=_DARK)),
        qr_img or Paragraph("QR non disponible",S_small),
    ]],[pw*.75,pw*.25],style=TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#D5DCE8")),
        ("LEFTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6)])))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def generate_osteo_report(result, output_path=None, **kwargs):
    pdf = build_osteo_pdf_report(result, **kwargs)
    if pdf and output_path:
        with open(output_path,"wb") as fh: fh.write(pdf)
    return pdf
