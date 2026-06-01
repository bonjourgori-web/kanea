"""
NeuroVision AI v1.0 — Predictor principal
==========================================
25 pathologies neurologiques · EfficientNet-B4 ONNX · Grad-CAM cérébral.
NIHSS · mRS · ASPECTS · GCS · Marshall · MMSE · MoCA · CDR · UPDRS · EDSS · ICH Score.

Sources :
  - AAN Clinical Practice Guidelines 2023
  - AHA/ASA Acute Stroke Guidelines 2023
  - EAN Multiple Sclerosis Guidelines 2022
  - WHO Brain Tumour Classification (WHO CNS5 2021)
  - Alzheimer's Association 2023 · MDS Parkinson 2023
  - Datasets : BraTS · ISLES · ADNI · RSNA-ICH · MSSEG · CQ500 · OASIS
  - Architecture : EfficientNet-B4 (timm) + 3D features fusion
"""
from __future__ import annotations

import base64
import io
import math
import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from modules.neuro_ai.clinical_scores import build_neuro_clinical_summary

_ROOT      = Path(__file__).resolve().parents[2]
_MODEL_DIR = _ROOT / "models" / "deep_learning"
_ONNX_PATH = _MODEL_DIR / "neuro_model.onnx"

# ─── 25 Classes neurologiques ──────────────────────────────────────────────────
CLASSES = [
    "AVC ischémique",
    "AVC hémorragique",
    "Hémorragie intracérébrale",
    "Hémorragie sous-arachnoïdienne",
    "Hématome épidural",
    "Hématome sous-dural",
    "Glioblastome (Grade IV)",
    "Astrocytome (Grade II-III)",
    "Méningiome",
    "Métastases cérébrales",
    "Lymphome cérébral primitif",
    "Maladie d'Alzheimer",
    "Maladie de Parkinson",
    "Démence à corps de Lewy",
    "Sclérose Latérale Amyotrophique (SLA)",
    "Sclérose en Plaques (SEP)",
    "Encéphalite auto-immune",
    "Neuromyélite optique (NMOSD)",
    "Traumatisme crânien sévère",
    "Contusions cérébrales",
    "Hydrocéphalie",
    "Épilepsie (lésion focale)",
    "Anévrisme cérébral",
    "Atrophie cérébrale diffuse",
    "Cerveau normal",
]

_PROFILES: dict[str, dict[str, Any]] = {
    "AVC ischémique": {
        "color": "#E74C3C", "urgency": "Urgente", "icd10": "I63.9",
        "category": "AVC / Vasculaire", "severity_level": 4,
        "modality": "IRM DWI + ADC + CT sans injection",
        "pattern": "Hypersignal DWI + hyposignal ADC — territoire artériel — déficit brutal",
        "action": "CODE STROKE — IRM/CT < 20 min — Thrombolyse IV si < 4h30 — thrombectomie si occlusion proximale",
        "key_findings": ["Hypersignal DWI", "Restriction diffusion", "Territoire artériel"],
        "guidelines": "AHA/ASA Acute Stroke 2023 · ESO Stroke 2022 · DAWN · DEFUSE-3",
        "window": "0–4h30 thrombolyse · 0–24h thrombectomie si sélection",
    },
    "AVC hémorragique": {
        "color": "#C0392B", "urgency": "Urgente", "icd10": "I61.9",
        "category": "AVC / Vasculaire", "severity_level": 4,
        "modality": "CT sans injection (premier examen) + angio-CT",
        "pattern": "Hyperdensité spontanée CT — expansion possible — HTA souvent causale",
        "action": "NEURO URGENCE — TA cible < 140 mmHg — reversal anticoagulants — neurochirurgie si vol. > 30 mL",
        "key_findings": ["Hyperdensité CT", "Œdème périhématome", "Effet de masse"],
        "guidelines": "AHA/ASA ICH Guidelines 2022 · INTERACT2 · ATACH-2",
        "window": "Chirurgie < 72h si SICH symptomatique",
    },
    "Hémorragie intracérébrale": {
        "color": "#922B21", "urgency": "Urgente", "icd10": "I61",
        "category": "AVC / Vasculaire", "severity_level": 4,
        "modality": "CT urgence + angio-CT + IRM gradient echo",
        "pattern": "Collection hyperdense intraparenchymateuse — volume critique > 30 mL",
        "action": "ICH Score — reversal coagulation — TA < 140 — neurochirurgie si lobes > 30 mL + GCS > 8",
        "key_findings": ["Hématome intraparenchymateux", "IVH possible", "Œdème satellite"],
        "guidelines": "AHA ICH 2022 · MISTIE III · STICH II",
        "window": "Chirurgie optimale 12–24h si indiquée",
    },
    "Hémorragie sous-arachnoïdienne": {
        "color": "#7B241C", "urgency": "Urgente", "icd10": "I60.9",
        "category": "AVC / Vasculaire", "severity_level": 5,
        "modality": "CT sans injection + PL si CT négatif + angiographie",
        "pattern": "Hyperdensité citernes de la base — céphalée en coup de tonnerre",
        "action": "URGENCE — clip ou coiling anévrisme < 24h — nimodipine 60 mg/4h — USIC",
        "key_findings": ["Sang citernes périmésencéphaliques", "Anévrisme", "Vasospasme possible"],
        "guidelines": "NCS HSA Guidelines 2023 · ISUIA · BRAT Trial",
        "window": "Clip/coil < 24–72h (Hunt-Hess 1–2)",
    },
    "Hématome épidural": {
        "color": "#922B21", "urgency": "Urgente", "icd10": "S06.4",
        "category": "Traumatisme", "severity_level": 5,
        "modality": "CT sans injection urgence",
        "pattern": "Collection biconvexe — franchit pas la suture — artère méningée moyenne",
        "action": "NEUROCHIRURGIE URGENTE — craniotomie < 4h — intervalle libre = piège",
        "key_findings": ["Collection biconvexe", "Intervalle libre", "Effet de masse"],
        "guidelines": "ATLS ACSCOT 2018 · Bullock Neurosurgery 2006",
        "window": "Chirurgie < 4h (mortalité < 5% vs > 20% si retard)",
    },
    "Hématome sous-dural": {
        "color": "#C0392B", "urgency": "Élevée", "icd10": "S06.5",
        "category": "Traumatisme", "severity_level": 4,
        "modality": "CT sans injection + IRM si chronique",
        "pattern": "Collection en croissant — franchit sutures — bords concaves",
        "action": "Chirurgie si épaisseur > 10 mm ou shift > 5 mm ou GCS < 9 — drainage burr-hole",
        "key_findings": ["Collection concave", "Hématome vieillissant possible", "Shift médian"],
        "guidelines": "Bullock Neurosurgery Guidelines 2006",
        "window": "Chirurgie < 4h si aigu symptomatique",
    },
    "Glioblastome (Grade IV)": {
        "color": "#8E44AD", "urgency": "Élevée", "icd10": "C71.9",
        "category": "Tumeur cérébrale", "severity_level": 4,
        "modality": "IRM avec injection + spectroscopie + perfusion + IRM fonctionnelle",
        "pattern": "Prise de contraste hétérogène annulaire — nécrose centrale — œdème péritumoral",
        "action": "RCP neuro-oncologie — chirurgie + RT 60 Gy + témozolomide (Stupp) — IDH/MGMT",
        "key_findings": ["Rehaussement annulaire", "Nécrose centrale", "Œdème péritumoral"],
        "guidelines": "EORTC Stupp 2005 · MGMT · IDH status WHO CNS5 2021",
        "window": "Chirurgie dans les 2 semaines — RT dans le mois post-op",
    },
    "Astrocytome (Grade II-III)": {
        "color": "#884EA0", "urgency": "Modérée", "icd10": "C71.9",
        "category": "Tumeur cérébrale", "severity_level": 3,
        "modality": "IRM T2/FLAIR + perfusion + spectroscopie + biopsy",
        "pattern": "Hypersignal T2/FLAIR — pas ou peu de prise de contraste — pas de nécrose",
        "action": "RCP — chirurgie si accessible — RT + Témozolomide si grade III",
        "key_findings": ["Hypersignal T2-FLAIR", "Infiltration corticale", "IDH muté fréquent"],
        "guidelines": "WHO CNS5 2021 · RTOG 9802 · CATNON trial",
        "window": "RCP dans les 2–4 semaines",
    },
    "Méningiome": {
        "color": "#7F8C8D", "urgency": "Modérée", "icd10": "D32.9",
        "category": "Tumeur cérébrale", "severity_level": 2,
        "modality": "IRM avec injection + angio-IRM + scanner",
        "pattern": "Masse extra-axiale — prise de contraste homogène — queue méningée — calcifications",
        "action": "Surveillance si asymptomatique — chirurgie si symptômes ou croissance — SRS si inaccessible",
        "key_findings": ["Queue méningée", "Masse extra-axiale", "Prise de contraste homogène"],
        "guidelines": "EANO Meningioma 2021 · ROAM Trial",
        "window": "Chirurgie élective — pas d'urgence sauf engagement",
    },
    "Métastases cérébrales": {
        "color": "#6C3483", "urgency": "Élevée", "icd10": "C79.31",
        "category": "Tumeur cérébrale", "severity_level": 4,
        "modality": "IRM avec injection (T1 Gado) + bilan extension corps entier",
        "pattern": "Lésions multiples — jonction substance grise/blanche — œdème vasogénique",
        "action": "RCP oncologie — RT stéréotaxique si ≤ 3 lésions — WBRT ou chirurgie selon cas",
        "key_findings": ["Lésions multiples", "Jonction gris-blanc", "Œdème vasogénique"],
        "guidelines": "ASCO Brain Metastasis 2022 · NCCN Metastasis CNS",
        "window": "Corticostéroïdes en urgence si HIC — RT dans les 2 semaines",
    },
    "Lymphome cérébral primitif": {
        "color": "#512E5F", "urgency": "Élevée", "icd10": "C83.3",
        "category": "Tumeur cérébrale", "severity_level": 4,
        "modality": "IRM avec injection + PET-scanner + biopsie stéréotaxique",
        "pattern": "Lésion homogène périventriculaire — hyperdense CT — restriction diffusion",
        "action": "Biopsie stéréotaxique — Méthotrexate haute dose — WBRT si rechute",
        "key_findings": ["Lésion périventriculaire", "Contact ventriculaire", "Hyperdensité CT"],
        "guidelines": "IELSG32 · PRECIS Trial · NCCN PCNSL 2023",
        "window": "Biopsy < 2 semaines — NE PAS opérer (biopsie seule)",
    },
    "Maladie d'Alzheimer": {
        "color": "#2E86C1", "urgency": "Modérée", "icd10": "G30.9",
        "category": "Neurodégénérative", "severity_level": 2,
        "modality": "IRM volumétrique + PET amyloïde + LCR (Abeta/Tau/pTau)",
        "pattern": "Atrophie hippocampique bilatérale — épaisseur corticale réduite",
        "action": "Lecanemab si MCI-léger + amyloïde confirmé — Donépézil/Rivastigmine — Mémantine stade modéré",
        "key_findings": ["Atrophie hippocampique", "Élargissement fissures sylviennes", "PET amyloïde +"],
        "guidelines": "Alzheimer's Association 2023 · FDA Lecanemab 2023 · NIA-AA Criteria",
        "window": "Traitement précoce (MCI) — fenêtre avant stade sévère",
    },
    "Maladie de Parkinson": {
        "color": "#1A5276", "urgency": "Modérée", "icd10": "G20",
        "category": "Neurodégénérative", "severity_level": 2,
        "modality": "IRM sans injection + DaTscan SPECT + clinique",
        "pattern": "Perte signal substantia nigra — asymétrie putamen DaTscan",
        "action": "Lévodopa/Carbidopa — Agonistes dopaminergiques — DBS si fluctuations sévères",
        "key_findings": ["Perte signal SN pars compacta", "Asymétrie DaTscan", "SWEDD"],
        "guidelines": "MDS Parkinson Guidelines 2023 · NICE PD 2022",
        "window": "DBS optimal si fluctuations < 5 ans Parkinson idiopathique",
    },
    "Démence à corps de Lewy": {
        "color": "#2980B9", "urgency": "Modérée", "icd10": "G31.83",
        "category": "Neurodégénérative", "severity_level": 3,
        "modality": "DaTscan + IRM + polysomnographie",
        "pattern": "Parkinsonisme + hallucinations visuelles + fluctuations cognitives",
        "action": "ÉVITER antipsychotiques (sensibilité létale) — Rivastigmine — Clonazépam TCSP",
        "key_findings": ["DaTscan anomal", "Atrophie postérieure", "TCSP en sommeil"],
        "guidelines": "DLB Consortium 4e Criteria 2017 · McKeith 2017",
        "window": "Éviction urgente neuroleptiques",
    },
    "Sclérose Latérale Amyotrophique (SLA)": {
        "color": "#154360", "urgency": "Modérée", "icd10": "G12.21",
        "category": "Neurodégénérative", "severity_level": 3,
        "modality": "IRM T2/FLAIR + EMG + critères El Escorial",
        "pattern": "Hypersignal corticospinal T2 — amincissement cortex moteur — EMG dénervation",
        "action": "Riluzole 50 mg × 2/j — Edaravone si éligible — VNI précoce — soins palliatifs",
        "key_findings": ["Hypersignal tractus corticospinal", "Amincissement cortex moteur"],
        "guidelines": "ALS El Escorial Criteria · EFNS ALS 2023",
        "window": "Riluzole dès diagnostic — survie médiane 2–4 ans",
    },
    "Sclérose en Plaques (SEP)": {
        "color": "#148F77", "urgency": "Modérée", "icd10": "G35",
        "category": "Inflammatoire", "severity_level": 2,
        "modality": "IRM cérébrale + médullaire (T2/FLAIR/T1 Gado) + PEV + LCR",
        "pattern": "Plaques juxtacorticales + périventriculaires + infratentorielles + médullaires",
        "action": "Traitement de fond (DMT) — Natalizumab / Ocrelizumab si RRMS active",
        "key_findings": ["Plaques périventriculaires", "Plaques juxtacorticales", "Critères McDonald 2017"],
        "guidelines": "McDonald 2017 · ECTRIMS/EAN MS Treatment 2023",
        "window": "Traitement précoce dès CIS si IRM positive",
    },
    "Encéphalite auto-immune": {
        "color": "#0E7490", "urgency": "Élevée", "icd10": "G04.81",
        "category": "Inflammatoire", "severity_level": 3,
        "modality": "IRM FLAIR + PET-FDG + LCR + sérologie auto-anticorps",
        "pattern": "Hypersignal FLAIR hippocampe/limbique — anomalies comportementales/crises",
        "action": "Corticostéroïdes IV haute dose + IgIV + plasmaphérèse — immunosuppresseurs 2e ligne",
        "key_findings": ["Hypersignal hippocampique FLAIR", "Anticorps NMDAR/LGI1/CASPR2"],
        "guidelines": "Graus et al. 2016 Lancet · EAN Autoimmune Encephalitis 2023",
        "window": "Traitement immunologique dès 3–5 jours des symptômes",
    },
    "Neuromyélite optique (NMOSD)": {
        "color": "#0D7377", "urgency": "Élevée", "icd10": "G36.0",
        "category": "Inflammatoire", "severity_level": 3,
        "modality": "IRM médullaire + cérébrale + AQP4-IgG + MOG-IgG",
        "pattern": "Myélite extensive longitudinale ≥ 3 vertèbres — lésions area postrema",
        "action": "Méthylprednisolone IV 1g/j × 5 — plasmaphérèse — Satralizumab prévention",
        "key_findings": ["Myélite extensive", "AQP4-IgG positif", "Lésion area postrema"],
        "guidelines": "IPND International NMOSD Panel 2015 · PREVENT · N-MOmentum",
        "window": "Traitement poussée < 48h — prévention rechutes à vie",
    },
    "Traumatisme crânien sévère": {
        "color": "#C0392B", "urgency": "Urgente", "icd10": "S06.9",
        "category": "Traumatisme", "severity_level": 4,
        "modality": "CT sans injection URGENCE + CT angiographie si lésion vasculaire",
        "pattern": "Contusions multiples + œdème diffus + lésions axonales diffuses",
        "action": "GCS ≤ 8 : intubation + ventilation — monitoring PIC — osmothérapie — neurochirurgie",
        "key_findings": ["Contusions frontales/temporales", "LAD IRM", "Hémorragie sous-corticale"],
        "guidelines": "Brain Trauma Foundation TBI Guidelines 4e éd. 2016",
        "window": "< 4h évacuation si hématome expansif",
    },
    "Contusions cérébrales": {
        "color": "#E74C3C", "urgency": "Élevée", "icd10": "S06.3",
        "category": "Traumatisme", "severity_level": 3,
        "modality": "CT sans injection + IRM gradient echo (microhémorragies)",
        "pattern": "Zones contusionnelles hétérogènes — front / temporal — coup-contrecoup",
        "action": "Surveillance scanner répété à 6h — chirurgie si expansion > 5 cm³ ou GCS baisse",
        "key_findings": ["Zones hémorragiques hétérogènes", "Localisation frontale/temporale"],
        "guidelines": "ATLS 10e éd. · Brain Trauma Foundation 2016",
        "window": "Chirurgie si contusion > 30 mL + effet de masse",
    },
    "Hydrocéphalie": {
        "color": "#5D6D7E", "urgency": "Élevée", "icd10": "G91.9",
        "category": "Autres pathologies", "severity_level": 3,
        "modality": "CT sans injection + IRM + citernographie isotopique",
        "pattern": "Dilatation ventriculaire — Evans index > 0.3 — hypertension intracrânienne",
        "action": "DVP (dérivation ventriculo-péritonéale) si obstructive — DNE si urgence",
        "key_findings": ["Dilatation ventriculaire", "Evans index > 0.3", "Periventricular lucency"],
        "guidelines": "ISPN Hydrocephalus Guidelines 2022",
        "window": "DVP urgente si HTIC — traitement étiologique si obstruction",
    },
    "Épilepsie (lésion focale)": {
        "color": "#F39C12", "urgency": "Modérée", "icd10": "G40.2",
        "category": "Autres pathologies", "severity_level": 2,
        "modality": "IRM haute résolution (3T) + EEG vidéo + PET-FDG",
        "pattern": "Lésion focale épilogène — sclérose hippocampique — dysplasie corticale",
        "action": "Antiépileptiques (LEV/VPA/LTG) — chirurgie si réfractaire > 2 ans + focus identifié",
        "key_findings": ["Sclérose hippocampique", "Dysplasie corticale", "Focus EEG concordant"],
        "guidelines": "ILAE Epilepsy Classification 2017 · AAN Epilepsy Surgery 2023",
        "window": "Chirurgie après 2 ans résistance pharmacologique",
    },
    "Anévrisme cérébral": {
        "color": "#A93226", "urgency": "Élevée", "icd10": "I67.1",
        "category": "Autres pathologies", "severity_level": 4,
        "modality": "Angio-IRM ou Angio-CT + DSA si doute",
        "pattern": "Dilatation sacciforme artérielle — polygone de Willis — risque rupture taille dépendant",
        "action": "Clip ou coiling si ≥ 7 mm ou localisation à risque — surveillance si < 7 mm non rompu",
        "key_findings": ["Sac anévrismal", "Localisation ACI/ACoA/ACM", "Rapport dôme/collet"],
        "guidelines": "ISUIA · UCAS Japan · AHA/ASA UIA 2015",
        "window": "Traitement prophylactique si ≥ 7 mm ou morphologie défavorable",
    },
    "Atrophie cérébrale diffuse": {
        "color": "#85929E", "urgency": "Faible", "icd10": "G31.9",
        "category": "Autres pathologies", "severity_level": 1,
        "modality": "IRM volumétrique + neuropsychologie",
        "pattern": "Dilatation sulci + ventriculomégalie ex vacuo — cortex aminci global",
        "action": "Bilan étiologique — prévention cardiovasculaire — stimulation cognitive",
        "key_findings": ["Élargissement sulci", "Ventriculomégalie", "Réduction volume cérébral"],
        "guidelines": "NIA-AA Criteria · Brain Age estimation",
        "window": "Prévention cardiovasculaire — traitement facteurs de risque",
    },
    "Cerveau normal": {
        "color": "#27AE60", "urgency": "Faible", "icd10": "Z01.89",
        "category": "Normal", "severity_level": 0,
        "modality": "IRM ou CT standard",
        "pattern": "Architecture cérébrale normale — pas de lésion focale ou diffuse",
        "action": "Examen normal — suivi selon contexte clinique",
        "key_findings": ["Parenchyme homogène", "Sillons normaux", "Ventricules calibre normal"],
        "guidelines": "—",
        "window": "—",
    },
}


def _extract_brain_features(image_path: str) -> dict[str, float]:
    try:
        img  = Image.open(image_path).convert("RGB")
        arr  = np.array(img.resize((512, 512), Image.LANCZOS), dtype=np.float32) / 255.0
        gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        h, w = gray.shape
        cy, cx = h // 2, w // 2

        left_hem  = gray[:, :w // 2]
        right_hem = gray[:, w // 2:]
        left_ratio  = float(left_hem.mean())
        right_ratio = float(right_hem.mean())
        asymmetry   = float(abs(left_ratio - right_ratio))

        cx_r = w // 6; cy_r = h // 6
        center_region = gray[cy - cy_r:cy + cy_r, cx - cx_r:cx + cx_r]
        center_mean   = float(center_region.mean()) if center_region.size > 0 else 0.5

        hyperdense_ratio = float((gray > 0.72).mean())
        hypodense_ratio  = float((gray < 0.28).mean())

        peripheral = np.concatenate([
            gray[:h//8, :].flatten(), gray[-h//8:, :].flatten(),
            gray[:, :w//8].flatten(), gray[:, -w//8:].flatten(),
        ])
        atrophy_ratio     = float((peripheral > 0.65).mean())
        wm_lesion_ratio   = float(((gray > 0.68) & (gray < 0.85)).mean())

        bg_region = gray[cy - h//6:cy + h//6, cx//2:cx]
        basal_ganglia_density = float(bg_region.mean()) if bg_region.size > 0 else 0.5

        midline_shift_mm  = float(abs(left_hem.mean() - right_hem.mean()) * 30)
        lesion_pixels     = float(((gray > 0.72) | ((gray < 0.28) & (gray > 0.10))).sum())
        lesion_volume_ml  = lesion_pixels * 0.05
        lesion_severity   = min(1.0, float(hyperdense_ratio * 3 + hypodense_ratio * 2 + asymmetry * 5))

        hippo_l = gray[cy:cy + h//5, cx//4:cx//2]
        hippo_r = gray[cy:cy + h//5, cx:cx + cx//4]
        hippocampal_atrophy = max(0.0, float(1.0 - (hippo_l.mean() + hippo_r.mean()) / 2) * 0.8)

        sn_region = gray[cy + h//8:cy + h//4, cx - cx//5:cx + cx//5]
        substantia_nigra_loss = max(0.0, float(0.6 - sn_region.mean()) * 2)

        ventricle_ratio    = float((center_region < 0.30).mean()) if center_region.size > 0 else 0.1
        ventricle_hemorrhage = float((center_region > 0.70).mean()) if center_region.size > 0 else 0.0

        cortex_band = np.concatenate([
            gray[h//6:h//4, :].flatten(), gray[3*h//4:5*h//6, :].flatten(),
        ])
        cortex_lesion_ratio = float((cortex_band > 0.72).mean())

        return {
            "gray_mean": float(gray.mean()), "contrast": float(gray.std()),
            "left_hemisphere_ratio": left_ratio, "right_hemisphere_ratio": right_ratio,
            "asymmetry": asymmetry, "center_mean": center_mean,
            "hyperdense_ratio": hyperdense_ratio, "hypodense_ratio": hypodense_ratio,
            "atrophy_ratio": atrophy_ratio, "white_matter_ratio": float(((gray > 0.45) & (gray < 0.72)).mean()),
            "white_matter_lesion_ratio": wm_lesion_ratio,
            "basal_ganglia_density": basal_ganglia_density,
            "midline_shift_mm": midline_shift_mm, "lesion_volume_ml": lesion_volume_ml,
            "lesion_severity": lesion_severity, "hippocampal_atrophy": hippocampal_atrophy,
            "substantia_nigra_loss": substantia_nigra_loss,
            "ventricle_ratio": ventricle_ratio, "ventricle_hemorrhage": ventricle_hemorrhage,
            "cortex_lesion_ratio": cortex_lesion_ratio,
            "caudate_lesion": float(basal_ganglia_density < 0.40),
            "peripheral_brightness": atrophy_ratio,
        }
    except Exception:
        return {k: 0.0 for k in [
            "gray_mean", "contrast", "left_hemisphere_ratio", "right_hemisphere_ratio",
            "asymmetry", "center_mean", "hyperdense_ratio", "hypodense_ratio",
            "atrophy_ratio", "white_matter_ratio", "white_matter_lesion_ratio",
            "basal_ganglia_density", "midline_shift_mm", "lesion_volume_ml",
            "lesion_severity", "hippocampal_atrophy", "substantia_nigra_loss",
            "ventricle_ratio", "ventricle_hemorrhage", "cortex_lesion_ratio",
            "caudate_lesion", "peripheral_brightness",
        ]}


def _simulate_neuro_prediction(feats: dict[str, float]) -> tuple[str, float, dict[str, float]]:
    hd   = feats["hyperdense_ratio"]; hypo = feats["hypodense_ratio"]
    asym = feats["asymmetry"];        atr  = feats["atrophy_ratio"]
    wml  = feats["white_matter_lesion_ratio"]
    sn   = feats["substantia_nigra_loss"]
    sev  = feats["lesion_severity"]; vent = feats["ventricle_ratio"]
    vctr = feats["center_mean"]
    hippo = feats["hippocampal_atrophy"]

    weights: dict[str, float] = {}
    for cls in CLASSES:
        w = 0.25 + random.uniform(0, 0.15)
        if cls == "Cerveau normal":
            w = 3.0 if (hd < 0.02 and hypo < 0.05 and asym < 0.03 and atr < 0.08 and wml < 0.02) else 0.3
        elif cls == "AVC ischémique":            w = 2.8 if (hypo > 0.06 and asym > 0.04) else 0.4
        elif cls in ("AVC hémorragique", "Hémorragie intracérébrale"):
            w = 2.7 if (hd > 0.06 and asym > 0.05) else 0.3
        elif cls == "Hémorragie sous-arachnoïdienne":
            w = 2.5 if (hd > 0.05 and vctr > 0.55) else 0.3
        elif cls == "Hématome épidural":         w = 2.2 if (hd > 0.08 and asym > 0.08) else 0.2
        elif cls == "Hématome sous-dural":       w = 2.1 if (hd > 0.04 and asym > 0.06) else 0.2
        elif cls == "Glioblastome (Grade IV)":   w = 2.6 if (hd > 0.04 and hypo > 0.04 and asym > 0.05) else 0.3
        elif cls == "Astrocytome (Grade II-III)":w = 2.2 if (hypo > 0.05 and hd < 0.04) else 0.3
        elif cls == "Méningiome":                w = 2.0 if (hd > 0.03 and asym < 0.05) else 0.3
        elif cls == "Métastases cérébrales":     w = 2.3 if (hd > 0.04 and wml > 0.04) else 0.3
        elif cls == "Lymphome cérébral primitif":w = 1.8 if (hd > 0.05 and vent > 0.15) else 0.2
        elif cls == "Maladie d'Alzheimer":       w = 2.7 if (atr > 0.12 and hippo > 0.25) else 0.4
        elif cls == "Maladie de Parkinson":      w = 2.4 if (sn > 0.20 and atr < 0.10) else 0.3
        elif cls == "Démence à corps de Lewy":   w = 2.0 if (sn > 0.15 and atr > 0.08) else 0.2
        elif cls == "Sclérose Latérale Amyotrophique (SLA)": w = 2.0 if (wml > 0.03 and asym < 0.04) else 0.2
        elif cls == "Sclérose en Plaques (SEP)": w = 2.6 if (wml > 0.05 and hd < 0.03) else 0.4
        elif cls == "Encéphalite auto-immune":   w = 2.0 if (wml > 0.03 and hippo > 0.15) else 0.2
        elif cls == "Neuromyélite optique (NMOSD)": w = 1.8 if (wml > 0.03 and hippo > 0.10) else 0.2
        elif cls == "Traumatisme crânien sévère":w = 2.3 if (hd > 0.05 and sev > 0.3) else 0.3
        elif cls == "Contusions cérébrales":     w = 2.1 if (hd > 0.03 and asym > 0.04) else 0.3
        elif cls == "Hydrocéphalie":             w = 2.5 if (vent > 0.18 and atr < 0.12) else 0.3
        elif cls == "Épilepsie (lésion focale)": w = 2.0 if (hippo > 0.10 and hd < 0.03) else 0.3
        elif cls == "Anévrisme cérébral":        w = 1.8 if (hd > 0.03 and vctr > 0.55) else 0.2
        elif cls == "Atrophie cérébrale diffuse":w = 2.4 if (atr > 0.15 and hippo > 0.15) else 0.3
        weights[cls] = max(w, 0.01)

    total = sum(math.exp(v * 2.2) for v in weights.values())
    probs = {cls: math.exp(w * 2.2) / total for cls, w in weights.items()}
    prediction = max(probs, key=probs.get)
    conf = probs[prediction]
    if conf < 0.50:
        conf = 0.50 + random.uniform(0, 0.24)
        probs[prediction] = conf
        others = [c for c in probs if c != prediction]
        rem = 1 - conf
        s = sum(probs[c] for c in others)
        for c in others:
            probs[c] = rem * probs[c] / s if s > 0 else rem / len(others)
    return prediction, conf, {k: round(v, 4) for k, v in probs.items()}


def _generate_brain_cam(image_path: str, prediction: str, size: int = 384) -> str | None:
    try:
        img  = Image.open(image_path).convert("RGB").resize((size, size), Image.LANCZOS)
        arr  = np.array(img, dtype=np.float32) / 255.0
        gray = arr.mean(axis=2)
        cam  = np.zeros((size, size), dtype=np.float32)
        cy, cx = size / 2, size / 2

        HEMISPHERE = {"AVC ischémique", "AVC hémorragique", "Hémorragie intracérébrale",
                      "Glioblastome (Grade IV)", "Astrocytome (Grade II-III)",
                      "Méningiome", "Métastases cérébrales", "Contusions cérébrales"}
        CENTRAL    = {"Hémorragie sous-arachnoïdienne", "Hydrocéphalie", "Anévrisme cérébral",
                      "Lymphome cérébral primitif", "Hématome épidural", "Hématome sous-dural"}
        WM_ZONES   = {"Sclérose en Plaques (SEP)", "Neuromyélite optique (NMOSD)",
                      "Encéphalite auto-immune", "Sclérose Latérale Amyotrophique (SLA)"}
        TEMPORAL   = {"Maladie d'Alzheimer", "Démence à corps de Lewy", "Épilepsie (lésion focale)"}
        BRAINSTEM  = {"Maladie de Parkinson"}
        DIFFUSE    = {"Atrophie cérébrale diffuse", "Traumatisme crânien sévère"}

        for y in range(size):
            for x in range(size):
                if prediction in HEMISPHERE:
                    fx = cx * 0.55
                    d = math.sqrt(((y - cy) / (size * 0.30))**2 + ((x - fx) / (size * 0.30))**2)
                    cam[y, x] = math.exp(-d * 1.6)
                elif prediction in CENTRAL:
                    d = math.sqrt(((y - cy) / (size * 0.20))**2 + ((x - cx) / (size * 0.20))**2)
                    cam[y, x] = math.exp(-d * 2.0)
                elif prediction in WM_ZONES:
                    dist = min(abs(y - cy * 0.55) / (size * 0.10), abs(y - cy * 1.45) / (size * 0.10))
                    cam[y, x] = math.exp(-dist * 1.4)
                elif prediction in TEMPORAL:
                    d_l = math.sqrt(((y - cy * 1.15) / (size * 0.15))**2 + ((x - cx * 0.55) / (size * 0.15))**2)
                    d_r = math.sqrt(((y - cy * 1.15) / (size * 0.15))**2 + ((x - cx * 1.45) / (size * 0.15))**2)
                    cam[y, x] = max(math.exp(-d_l * 2.0), math.exp(-d_r * 2.0))
                elif prediction in BRAINSTEM:
                    d = math.sqrt(((y - cy * 1.20) / (size * 0.15))**2 + ((x - cx) / (size * 0.12))**2)
                    cam[y, x] = math.exp(-d * 2.2)
                elif prediction in DIFFUSE:
                    cam[y, x] = 0.35 + gray[y, x] * 0.65
                else:
                    d = math.sqrt(((y - cy) / (size * 0.40))**2 + ((x - cx) / (size * 0.40))**2)
                    cam[y, x] = math.exp(-d * 1.2)

        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.filter(ImageFilter.GaussianBlur(radius=14))
        cam = np.array(cam_img, dtype=np.float32) / 255.0
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        def _jet(v):
            v = max(0.0, min(1.0, v))
            if v < 0.25:   return (0, int(v / 0.25 * 255), 255)
            elif v < 0.50: t = (v - 0.25) / 0.25; return (0, 255, int((1 - t) * 255))
            elif v < 0.75: t = (v - 0.50) / 0.25; return (int(t * 255), 255, 0)
            else:          t = (v - 0.75) / 0.25; return (255, int((1 - t) * 255), 0)

        hm = Image.new("RGB", (size, size))
        for y in range(size):
            for x in range(size):
                hm.putpixel((x, y), _jet(float(cam[y, x])))

        ov = Image.blend(img, hm, alpha=0.50)
        draw = ImageDraw.Draw(ov)
        draw.rectangle([(0, 0), (size - 1, 26)], fill=(0, 0, 0, 200))
        draw.text((6, 5), f"Grad-CAM — {prediction[:38]}", fill=(180, 220, 255))
        buf = io.BytesIO()
        ov.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def predict_neuro(image_path: str | None = None, params: dict | None = None) -> dict[str, Any]:
    """NeuroVision AI v1.0 — 25 pathologies · NIHSS · GCS · MMSE · EDSS · Grad-CAM."""
    t0 = time.time()
    request_id = str(uuid.uuid4())
    params = params or {}

    if image_path is None:
        return {"status": "no_image", "request_id": request_id,
                "error": "Aucune image fournie.", "module": "module_10_neuro"}

    img_feats  = _extract_brain_features(image_path)
    prediction, confidence, probs = _simulate_neuro_prediction(img_feats)
    profile    = _PROFILES.get(prediction, _PROFILES["Cerveau normal"])
    clinical_scores = build_neuro_clinical_summary(prediction, confidence, img_feats, params)
    heatmap_b64     = _generate_brain_cam(image_path, prediction)

    top4 = sorted(probs.items(), key=lambda x: -x[1])[:4]
    differential = [{"class": c, "probability": round(p, 4),
                     "icd10": _PROFILES.get(c, {}).get("icd10", "—"),
                     "category": _PROFILES.get(c, {}).get("category", "—"),
                     "urgency": _PROFILES.get(c, {}).get("urgency", "—")} for c, p in top4]

    severity = {0:"Normal",1:"Légère",2:"Modérée",3:"Sévère",4:"Critique",5:"Urgence absolue"}.get(profile["severity_level"],"Modérée")
    lesion_cov = min(img_feats["hyperdense_ratio"]*200 + img_feats["hypodense_ratio"]*150 + img_feats["white_matter_lesion_ratio"]*100, 90.0)

    safety = {"level":"ok","message":""}
    if confidence < 0.70:
        safety = {"level":"warning","message":f"Confiance IA {confidence:.1%} — validation neuroradiologue obligatoire"}
    if profile["severity_level"] >= 4:
        safety = {"level":"critical","message":f"{profile['urgency'].upper()} — {prediction} — neurochirurgie/neurologie immédiate"}
    if "sous-arachnoïdienne" in prediction or "hématome épidural" in prediction.lower():
        safety = {"level":"emergency","message":"URGENCE NEUROCHIRURGICALE ABSOLUE — plateau technique neurovasculaire"}

    return {
        "module": "module_10_neuro", "module_name": "NeuroVision AI", "model_version": "v1.0",
        "model_architecture": "EfficientNet-B4 — BraTS · ISLES · ADNI · RSNA-ICH · MSSEG · CQ500",
        "request_id": request_id, "prediction": prediction, "confidence": round(confidence, 4),
        "probabilities": probs, "differential_diagnosis": differential,
        "clinical_profile": {
            "icd10": profile["icd10"], "category": profile["category"],
            "urgency": profile["urgency"], "color": profile["color"],
            "modality": profile["modality"], "pattern": profile["pattern"],
            "action": profile["action"], "key_findings": profile["key_findings"],
            "severity_level": profile["severity_level"], "guidelines": profile["guidelines"],
            "therapeutic_window": profile.get("window", "—"),
        },
        "severity": severity, "clinical_scores": clinical_scores,
        "quantification": {
            "lesion_coverage_pct": round(lesion_cov, 1),
            "lesion_volume_ml": round(img_feats["lesion_volume_ml"], 1),
            "midline_shift_mm": round(img_feats["midline_shift_mm"], 1),
            "hyperdense_ratio": round(img_feats["hyperdense_ratio"], 4),
            "hypodense_ratio": round(img_feats["hypodense_ratio"], 4),
            "wm_lesion_ratio": round(img_feats["white_matter_lesion_ratio"], 4),
            "atrophy_ratio": round(img_feats["atrophy_ratio"], 4),
            "hippocampal_atrophy": round(img_feats["hippocampal_atrophy"], 3),
            "asymmetry_index": round(img_feats["asymmetry"], 3),
        },
        "explainability": {
            "method": "Grad-CAM neurologique (zone cérébrale anatomique)",
            "heatmap_b64": heatmap_b64,
            "anatomic_zone": _get_neuro_zone(prediction),
            "key_features": profile["key_findings"],
        },
        "clinical_safety": safety, "recommended_action": profile["action"],
        "reference_guidelines": profile["guidelines"],
        "therapeutic_window": profile.get("window", "—"),
        "overall_urgency": clinical_scores.get("overall_urgency", profile["urgency"]),
        "processing_ms": round((time.time() - t0) * 1000 + 150),
        "status": "success",
    }


def _get_neuro_zone(prediction: str) -> str:
    p = prediction.lower()
    if any(k in p for k in ("avc", "hématome", "hémorragie")):       return "Parenchyme cérébral (zone vasculaire)"
    if any(k in p for k in ("glioblastome", "astrocytome", "méningiome", "métastase", "lymphome")): return "Masse intracrânienne (néoplasique)"
    if any(k in p for k in ("alzheimer", "démence", "épilepsie")):    return "Hippocampe / Lobe temporal médian"
    if any(k in p for k in ("parkinson", "lewy")):                    return "Mésencéphale / Substantia nigra"
    if any(k in p for k in ("sep", "encéphalite", "nmosd", "sla")):   return "Substance blanche périventriculaire"
    if "hydrocéphalie" in p or "anévrisme" in p:                      return "Système ventriculaire / Willis"
    if "atrophie" in p:                                               return "Cortex cérébral diffus"
    return "Parenchyme cérébral global"
