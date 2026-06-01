"""
GastroAI — Rapport PDF médical gastroentérologique A4
======================================================
Child-Pugh · MELD · CDAI · Mayo · Ranson · BISAP · Blatchford · Rockall · TNM · Paris
Grad-CAM endoscopique · QR Code · multi-pages.
"""
from __future__ import annotations
import base64, io, uuid
from datetime import datetime
from typing import Any

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable, Image as RLImage)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    _RL_OK = True
except ImportError:
    _RL_OK = False

try:
    import qrcode as _qrc; _QR_OK = True
except ImportError:
    _QR_OK = False

_NAVY  = colors.HexColor("#0D2137") if _RL_OK else None
_GREEN = colors.HexColor("#059669") if _RL_OK else None
_TEAL  = colors.HexColor("#0E7490") if _RL_OK else None
_RED   = colors.HexColor("#DC2626") if _RL_OK else None
_ORG   = colors.HexColor("#EA580C") if _RL_OK else None
_PURP  = colors.HexColor("#7C3AED") if _RL_OK else None
_LIGHT = colors.HexColor("#ECFDF5") if _RL_OK else None
_MUTED = colors.HexColor("#64748B") if _RL_OK else None
_WHITE = colors.white               if _RL_OK else None
_DARK  = colors.HexColor("#0F172A") if _RL_OK else None
_URG_C = {
    "Urgente":        colors.HexColor("#C0392B") if _RL_OK else None,
    "Élevée":         colors.HexColor("#E74C3C") if _RL_OK else None,
    "Modérée":        colors.HexColor("#E67E22") if _RL_OK else None,
    "Faible":         colors.HexColor("#27AE60") if _RL_OK else None,
    "Urgence absolue":colors.HexColor("#7B241C") if _RL_OK else None,
}


def _s(n, **kw): return ParagraphStyle(n, **kw)
def _qr(txt, sz=52):
    if not _QR_OK or not _RL_OK: return None
    try:
        q = _qrc.QRCode(box_size=2, border=2); q.add_data(txt); q.make(fit=True)
        img = q.make_image(); buf = io.BytesIO(); img.save(buf,"PNG"); buf.seek(0)
        return RLImage(buf, width=sz, height=sz)
    except: return None
def _b64img(b64, w=130, h=130):
    if not _RL_OK or not b64: return None
    try: return RLImage(io.BytesIO(base64.b64decode(b64)), width=w, height=h)
    except: return None


def generate_gastro_report(result: dict[str, Any], patient_info: dict | None = None) -> bytes | None:
    if not _RL_OK: return None
    patient_info = patient_info or {}
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14*mm, rightMargin=14*mm,
                              topMargin=11*mm, bottomMargin=11*mm)
    W    = A4[0] - 28*mm
    now  = datetime.now(); rid = str(uuid.uuid4())[:12].upper()

    pred    = result.get("prediction","—"); conf = result.get("confidence",0)
    severity= result.get("severity","—")
    urgency = result.get("overall_urgency", result.get("clinical_profile",{}).get("urgency","Faible"))
    profile = result.get("clinical_profile",{}); scores = result.get("clinical_scores",{})
    quant   = result.get("quantification",{}); diff = result.get("differential_diagnosis",[])
    safety  = result.get("clinical_safety",{}); xai  = result.get("explainability",{})
    hm_b64  = xai.get("heatmap_b64",""); urg_c = _URG_C.get(urgency, _ORG)

    B  = _s("B",  fontSize=8.5, textColor=_DARK, fontName="Helvetica",    spaceBefore=1, spaceAfter=1, leading=12)
    BD = _s("BD", fontSize=8.5, textColor=_DARK, fontName="Helvetica-Bold",spaceBefore=1, spaceAfter=1)
    H2 = _s("H2", fontSize=11,  textColor=_NAVY, fontName="Helvetica-Bold",spaceBefore=7, spaceAfter=3)
    H3 = _s("H3", fontSize=9.5, textColor=_TEAL, fontName="Helvetica-Bold",spaceBefore=4, spaceAfter=2)
    SM = _s("SM", fontSize=7.5, textColor=_MUTED,fontName="Helvetica",    spaceAfter=1)
    WN = _s("WN", fontSize=8.5, textColor=_WHITE,fontName="Helvetica-Bold",alignment=TA_CENTER)

    story = []

    # Header
    hdr = Table([[
        Paragraph("<b>KANEA</b>", _s("KH",fontSize=20,textColor=colors.HexColor("#34D399"),
                                    fontName="Helvetica-Bold",alignment=TA_LEFT)),
        Paragraph("GastroAI<br/><font size='9' color='#94A3B8'>Rapport gastroentérologique · Endoscopie IA</font>",
                  _s("GH",fontSize=13,textColor=_WHITE,fontName="Helvetica-Bold",alignment=TA_CENTER)),
        _qr(f"KANEA-GASTRO-{rid}",48) or Paragraph("",B),
    ]], colWidths=[W*0.22,W*0.56,W*0.22])
    hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),_NAVY),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
    story.append(hdr); story.append(Spacer(1,4))

    meta = Table([[Paragraph(f"<b>ID :</b> GASTRO-{rid}",SM),
                   Paragraph(f"<b>Date :</b> {now.strftime('%d/%m/%Y %H:%M')}",SM),
                   Paragraph("<b>Module :</b> GastroAI v1.0",SM),
                   Paragraph("<b>Modèle :</b> EfficientNet-B5 ONNX",SM)]],
                 colWidths=[W/4]*4)
    meta.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#F1F5F9")),
        ("FONTSIZE",(0,0),(-1,-1),7.5),("TEXTCOLOR",(0,0),(-1,-1),_MUTED),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",(0,0),(-1,-1),8),
        ("BOX",(0,0),(-1,-1),0.3,colors.HexColor("#CBD5E1"))]))
    story.append(meta); story.append(Spacer(1,5))

    # Patient
    pat = Table([
        [Paragraph(f"<b>Patient :</b> {patient_info.get('name','Anonyme')}",BD),
         Paragraph(f"<b>Âge :</b> {patient_info.get('age','—')} ans",B),
         Paragraph(f"<b>Sexe :</b> {patient_info.get('sex','—')}",B),
         Paragraph(f"<b>ID :</b> PAT-{rid[:6]}",B)],
        [Paragraph(f"<b>Modalité :</b> {patient_info.get('modality',profile.get('modality','—'))}",B),
         Paragraph(f"<b>Indication :</b> {patient_info.get('indication','Dépistage/diagnostic')}",B),
         Paragraph(f"<b>Opérateur :</b> {patient_info.get('operator','KANEA IA')}",B),
         Paragraph(f"<b>Service :</b> {patient_info.get('service','Gastroentérologie')}",B)],
    ], colWidths=[W*0.30,W*0.23,W*0.23,W*0.24])
    pat.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),8.5),("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),("LEFTPADDING",(0,0),(-1,-1),7),
        ("BOX",(0,0),(-1,-1),0.5,_GREEN),("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#E2E8F0")),
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#F8FAFC"))]))
    story.append(pat); story.append(Spacer(1,7))

    # Diagnostic principal
    diag = Table([[
        Table([
            [Paragraph("DIAGNOSTIC GASTROAI",_s("DH",fontSize=7.5,textColor=colors.HexColor("#94A3B8"),
                                                fontName="Helvetica-Bold",alignment=TA_CENTER))],
            [Paragraph(pred,_s("DP",fontSize=12,textColor=_WHITE,fontName="Helvetica-Bold",
                                alignment=TA_CENTER,leading=15))],
            [Paragraph(f"<b>Confiance IA : {conf:.1%}</b>",
                       _s("DC",fontSize=10,textColor=colors.HexColor("#34D399"),
                          fontName="Helvetica-Bold",alignment=TA_CENTER))],
        ],colWidths=[W*0.54]),
        Table([
            [Paragraph("SÉVÉRITÉ",_s("SH",fontSize=7.5,textColor=colors.HexColor("#94A3B8"),
                                    fontName="Helvetica-Bold",alignment=TA_CENTER))],
            [Paragraph(f"<b>{severity}</b>",_s("SV",fontSize=15,textColor=urg_c,
                                               fontName="Helvetica-Bold",alignment=TA_CENTER))],
            [Paragraph(f"Urgence : {urgency}",_s("SU",fontSize=9,textColor=_MUTED,
                                                  fontName="Helvetica",alignment=TA_CENTER))],
        ],colWidths=[W*0.38]),
    ]],colWidths=[W*0.60,W*0.40])
    diag.setStyle(TableStyle([("BACKGROUND",(0,0),(0,0),_NAVY),
        ("BACKGROUND",(1,0),(1,0),colors.HexColor("#F8FAFC")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),9),
        ("BOTTOMPADDING",(0,0),(-1,-1),9),("LEFTPADDING",(0,0),(-1,-1),10),
        ("BOX",(0,0),(-1,-1),1.0,_GREEN),("INNERGRID",(0,0),(-1,-1),0.5,colors.HexColor("#CBD5E1"))]))
    story.append(diag); story.append(Spacer(1,5))

    if safety.get("level") in ("warning","critical","emergency"):
        abg={"warning":"#92400E","critical":"#991B1B","emergency":"#7B241C"}.get(safety["level"],"#92400E")
        at=Table([[Paragraph(f"⚠ {safety['message']}",WN)]],colWidths=[W])
        at.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor(abg)),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LEFTPADDING",(0,0),(-1,-1),10)]))
        story.append(at); story.append(Spacer(1,4))

    # Profil clinique
    story.append(Paragraph("PROFIL CLINIQUE & EXAMENS",H2))
    story.append(HRFlowable(width=W,thickness=1.5,color=_GREEN,spaceAfter=3))
    findings_txt = " · ".join(profile.get("key_findings",[]))
    cl=Table([
        [Paragraph("<b>CIM-10</b>",BD),Paragraph(profile.get("icd10","—"),B),
         Paragraph("<b>Catégorie</b>",BD),Paragraph(profile.get("category","—"),B)],
        [Paragraph("<b>Urgence</b>",BD),Paragraph(profile.get("urgency","—"),B),
         Paragraph("<b>Modalité</b>",BD),Paragraph(profile.get("modality","—"),B)],
        [Paragraph("<b>Pattern</b>",BD),Paragraph(profile.get("pattern","—"),B),
         Paragraph("<b>Examens</b>",BD),Paragraph(profile.get("exam_type","—"),B)],
        [Paragraph("<b>Conduite</b>",BD),Paragraph(profile.get("action","—"),B),
         Paragraph("<b>Références</b>",BD),Paragraph(profile.get("guidelines","—"),SM)],
        [Paragraph("<b>Signes-clés</b>",BD),Paragraph(findings_txt,B),Paragraph("",B),Paragraph("",B)],
    ],colWidths=[W*0.18,W*0.32,W*0.18,W*0.32])
    cl.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#ECFDF5")),
        ("BACKGROUND",(2,0),(2,-1),colors.HexColor("#ECFDF5")),
        ("FONTSIZE",(0,0),(-1,-1),8.5),("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),6),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#CBD5E1")),
        ("BOX",(0,0),(-1,-1),0.5,_GREEN),("SPAN",(1,4),(3,4))]))
    story.append(cl); story.append(Spacer(1,7))

    # Scores hépatiques
    story.append(Paragraph("SCORES HÉPATIQUES — Child-Pugh · MELD · MELD-Na",H2))
    story.append(HRFlowable(width=W,thickness=1.5,color=_GREEN,spaceAfter=3))
    cp=scores.get("child_pugh",{}); ml=scores.get("meld",{})
    def _sr(l,v,d): return [Paragraph(f"<b>{l}</b>",BD),
                             Paragraph(str(v),_s("V",fontSize=9,textColor=_DARK,fontName="Helvetica-Bold")),
                             Paragraph(d,B)]
    hep=Table([
        _sr("Child-Pugh Score",f"{cp.get('score','—')}/15",f"Grade {cp.get('grade','—')} — Survie 1 an : {cp.get('survival_1y','—')} / 2 ans : {cp.get('survival_2y','—')}"),
        _sr("Transplantation","Oui" if cp.get("transplant") else "Non",cp.get("recommendation","—")),
        _sr("MELD Score",cp.get("meld",ml.get("meld","—")),f"Mortalité 90j : {ml.get('mortality_90d','—')} — Priorité : {ml.get('priority','—')}"),
        _sr("MELD-Na",ml.get("meld_na","—"),ml.get("recommendation","—")),
    ],colWidths=[W*0.22,W*0.18,W*0.60])
    hep.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),8.5),("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),6),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#ECFDF5"),_WHITE]),
        ("BOX",(0,0),(-1,-1),0.5,_GREEN)]))
    story.append(hep); story.append(Spacer(1,5))

    # Scores MICI
    story.append(Paragraph("MALADIES INFLAMMATOIRES — CDAI · Harvey-Bradshaw · Mayo Score",H2))
    story.append(HRFlowable(width=W,thickness=1.5,color=_GREEN,spaceAfter=3))
    cd=scores.get("cdai",{}); hbi=scores.get("hbi",{}); mayo=scores.get("mayo",{})
    mici=Table([
        _sr("CDAI Score",f"{cd.get('score','—')}",f"Activité : {cd.get('activity','—')} — Rémission : {'Oui' if cd.get('remission') else 'Non'}"),
        _sr("Biologique Crohn","Oui" if cd.get("biologic") else "Non",cd.get("recommendation","—")),
        _sr("Harvey-Bradshaw",f"{hbi.get('score','—')}",f"{hbi.get('activity','—')} — {hbi.get('recommendation','—')}"),
        _sr("Mayo Score (RCH)",f"{mayo.get('score','—')}/12 (partial: {mayo.get('partial','—')}/9)",
            f"Activité : {mayo.get('activity','—')} — Biologique : {'Oui' if mayo.get('biologic') else 'Non'}"),
        _sr("Conduite RCH","",mayo.get("recommendation","—")),
    ],colWidths=[W*0.22,W*0.18,W*0.60])
    mici.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),8.5),("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),6),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#F0FDF4"),_WHITE]),
        ("BOX",(0,0),(-1,-1),0.5,_TEAL)]))
    story.append(mici); story.append(Spacer(1,5))

    # Scores pancréatite + hémorragie
    story.append(Paragraph("PANCRÉATITE & HÉMORRAGIE — Ranson · BISAP · Blatchford · Rockall",H2))
    story.append(HRFlowable(width=W,thickness=1.5,color=_GREEN,spaceAfter=3))
    ran=scores.get("ranson",{}); bis=scores.get("bisap",{}); bla=scores.get("blatchford",{}); rok=scores.get("rockall",{})
    pan=Table([
        _sr("Ranson Score",f"{ran.get('score','—')}/11",f"Sévérité : {ran.get('severity','—')} — Mortalité : {ran.get('mortality','—')}"),
        _sr("USI Ranson","Oui" if ran.get("icu") else "Non",ran.get("recommendation","—")),
        _sr("BISAP Score",f"{bis.get('score','—')}/5",f"Sévérité : {bis.get('severity','—')} — Mortalité : {bis.get('mortality','—')}"),
        _sr("Glasgow-Blatchford",f"{bla.get('score','—')}",f"Risque : {bla.get('risk','—')} — Endoscopie : {bla.get('endoscopy','—')}"),
        _sr("Risque transfusion",bla.get("transfusion_risk","—"),bla.get("recommendation","—")),
        _sr("Rockall Score",f"Pré:{rok.get('pre_score','—')} / Complet:{rok.get('full_score','—')}",
            f"Récidive : {rok.get('rebleed','—')} — Mortalité : {rok.get('mortality','—')}"),
    ],colWidths=[W*0.22,W*0.18,W*0.60])
    pan.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),8.5),("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),6),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#FFF7ED"),_WHITE]),
        ("BOX",(0,0),(-1,-1),0.5,_ORG)]))
    story.append(pan); story.append(Spacer(1,5))

    # TNM + Paris
    story.append(Paragraph("TNM COLORECTAL & CLASSIFICATION DE PARIS",H2))
    story.append(HRFlowable(width=W,thickness=1.5,color=_GREEN,spaceAfter=3))
    tnm=scores.get("tnm_colorectal",{}); par=scores.get("paris",{})
    onco=Table([
        _sr("TNM AJCC 8e",f"T{tnm.get('T','?')} N{tnm.get('N','?')} M{tnm.get('M','?')}",f"Stade : {tnm.get('stage','—')} — Survie 5 ans : {tnm.get('survival_5y','—')}"),
        _sr("Traitement TNM","",tnm.get("treatment","—")),
        _sr("Classification Paris",par.get("classification","—"),f"{par.get('morphology','—')} — Risque cancer : {par.get('cancer_risk','—')}"),
        _sr("Résection recommandée","",par.get("resection","—")),
    ],colWidths=[W*0.22,W*0.18,W*0.60])
    onco.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),8.5),("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),6),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#F5F3FF"),_WHITE]),
        ("BOX",(0,0),(-1,-1),0.5,_PURP)]))
    story.append(onco); story.append(Spacer(1,6))

    # Grad-CAM + Différentiel
    story.append(Paragraph("VISUALISATION IA & DIAGNOSTIC DIFFÉRENTIEL",H2))
    story.append(HRFlowable(width=W,thickness=1.5,color=_GREEN,spaceAfter=3))
    rl_cam = _b64img(hm_b64,135,135)
    if rl_cam:
        cam_row=Table([[rl_cam,Table([
            [Paragraph("<b>Grad-CAM — Cartographie endoscopique IA</b>",H3)],
            [Paragraph(f"Zone anatomique : {xai.get('anatomic_zone','—')}",B)],
            [Paragraph("Grad-CAM adaptatif par zone digestive (muqueuse / lésion focale / hépatique / voies biliaires).",SM)],
            [Paragraph("Rouge = activation maximale · Bleu = activation faible.",SM)],
        ],colWidths=[W-148])]],colWidths=[145,W-145])
        cam_row.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#E2E8F0")),
            ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#F8FAFC")),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        story.append(cam_row); story.append(Spacer(1,5))

    dh=[Paragraph(h,_s("DH",fontSize=8,textColor=_WHITE,fontName="Helvetica-Bold",alignment=TA_CENTER))
        for h in ["Diagnostic","Probabilité","CIM-10","Catégorie","Urgence"]]
    dr=[dh]
    for i,d in enumerate(diff):
        dr.append([Paragraph(d.get("class","—"),B),
                   Paragraph(f"<b>{d.get('probability',0):.1%}</b>",
                              _s("P",fontSize=9,fontName="Helvetica-Bold",
                                 textColor=_GREEN if i==0 else _DARK,alignment=TA_CENTER)),
                   Paragraph(d.get("icd10","—"),SM),Paragraph(d.get("category","—"),SM),
                   Paragraph(d.get("urgency","—"),_s("U",fontSize=8,fontName="Helvetica",
                              textColor=_URG_C.get(d.get("urgency",""),_MUTED) or _MUTED,
                              alignment=TA_CENTER))])
    dt=Table(dr,colWidths=[W*0.36,W*0.12,W*0.12,W*0.22,W*0.18])
    dt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),_NAVY),
        ("FONTSIZE",(0,0),(-1,-1),8.5),("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),6),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#ECFDF5"),_WHITE]),
        ("BOX",(0,0),(-1,-1),0.5,_GREEN)]))
    story.append(dt); story.append(Spacer(1,6))

    # Quantification
    story.append(Paragraph("QUANTIFICATION LÉSIONNELLE",H2))
    story.append(HRFlowable(width=W,thickness=1.5,color=_GREEN,spaceAfter=3))
    qr2=[[Paragraph(h,_s("QH",fontSize=8,textColor=_WHITE,fontName="Helvetica-Bold"))
          for h in ["Paramètre","Valeur","Référence","Interprétation"]]]
    qd=[("Surface lésionnelle",f"{quant.get('lesion_coverage_pct',0):.1f}%","< 5% = minimal","Proportion muqueuse atteinte"),
        ("Ratio saignements",f"{quant.get('bleeding_ratio',0):.4f}","Urgence si > 0.04","Zones hémorragiques détectées"),
        ("Inflammation muqueuse",f"{quant.get('mucosal_inflammation',0):.4f}","IBD si > 0.07","Hyperhémie/érosions"),
        ("Ratio ulcérations",f"{quant.get('ulcer_ratio',0):.4f}","Signif. si > 0.03","Zones ulcérées"),
        ("Ratio polypes",f"{quant.get('polyp_ratio',0):.4f}","Dépistage si > 0.02","Lésions surélevées détectées"),
        ("Lésion focale",f"{quant.get('focal_lesion_ratio',0):.4f}","Masse si > 0.05","Zone hyperdense focale"),
        ("Brightess hépatique",f"{quant.get('hepatic_brightness',0):.3f}","Stéatose si > 0.40","Hyperéchogénicité hépatique")]
    for row in qd: qr2.append([Paragraph(str(c),B) for c in row])
    qt=Table(qr2,colWidths=[W*0.28,W*0.16,W*0.24,W*0.32])
    qt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),_TEAL),
        ("FONTSIZE",(0,0),(-1,-1),8.5),("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),6),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#ECFDF5"),_WHITE]),
        ("BOX",(0,0),(-1,-1),0.5,_TEAL)]))
    story.append(qt); story.append(Spacer(1,6))

    # Recommandations
    story.append(Paragraph("RECOMMANDATIONS & AVERTISSEMENT",H2))
    story.append(HRFlowable(width=W,thickness=1.5,color=_GREEN,spaceAfter=3))
    rt=Table([
        [Paragraph("<b>Conduite à tenir</b>",BD),Paragraph(profile.get("action",result.get("recommended_action","—")),B)],
        [Paragraph("<b>Références</b>",BD),Paragraph(profile.get("guidelines",result.get("reference_guidelines","—")),SM)],
        [Paragraph("<b>Avertissement IA</b>",BD),
         Paragraph("Rapport généré par IA d'aide au diagnostic. Ne remplace pas l'avis d'un "
                   "gastroentérologue qualifié. Toute décision thérapeutique doit être validée "
                   "par un médecin.",SM)],
        [Paragraph("<b>Traitement</b>",BD),Paragraph(f"{result.get('processing_ms','—')} ms · EfficientNet-B5 ONNX CPU",SM)],
    ],colWidths=[W*0.25,W*0.75])
    rt.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),8.5),("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),("LEFTPADDING",(0,0),(-1,-1),8),
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#ECFDF5")),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#BBF7D0")),
        ("BOX",(0,0),(-1,-1),0.8,_GREEN)]))
    story.append(rt); story.append(Spacer(1,6))

    # Footer
    story.append(HRFlowable(width=W,thickness=0.8,color=_MUTED,spaceAfter=3))
    ft=Table([[
        Paragraph(f"KANEA — GastroAI v1.0 · Knowledge Anthropology & Neural Engine for Africa",
                  _s("F",fontSize=7,textColor=_MUTED,fontName="Helvetica")),
        Paragraph(f"GASTRO-{rid} · {now.strftime('%d/%m/%Y %H:%M')} · Confiance : {conf:.1%}",
                  _s("F2",fontSize=7,textColor=_MUTED,fontName="Helvetica",alignment=TA_RIGHT)),
    ]],colWidths=[W*0.60,W*0.40])
    ft.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
        ("LEFTPADDING",(0,0),(-1,-1),0)]))
    story.append(ft)
    doc.build(story)
    return buf.getvalue()
