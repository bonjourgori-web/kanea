"""
KANEA — Moteur de scaffold partagé pour les modules IA en cours de développement.
Chaque module définit sa config et appelle `scaffold_predict()`.
"""
from __future__ import annotations

import random
import time
import uuid
from pathlib import Path
from typing import Any


# ── Configs des 12 modules ──────────────────────────────────────────────────

MODULES: dict[str, dict[str, Any]] = {

    "pulmoscan": {
        "icd10":          "J00–J99 (maladies respiratoires), J18 (pneumonie), A15 (tuberculose), U07 (COVID-19)",
        "prevalence":     "Pneumonie : 2,5 millions de décès/an (OMS 2022) · Tuberculose : 10 millions cas/an · "
                          "Cancer pulmonaire : 1er cancer meurtrier mondial (1,8M décès/an)",
        "africa_context": "Tuberculose = 2ème cause de mortalité infectieuse en Afrique subsaharienne. "
                          "Pneumonie = 1ère cause de mortalité infantile (sous 5 ans).",
        "risk_factors":   ["Tabagisme", "Pollution atmosphérique", "VIH/SIDA", "Diabète", "Dénutrition",
                           "Habitat surpeuplé", "Déficit immunitaire", "Expositions professionnelles (amiante)"],
        "diagnosis":      ["Radiographie thoracique", "TDM thoracique", "Bacilloscopie (BAAR)", "PCR GeneXpert",
                           "Bronchoscopie", "Spirométrie (BPCO)", "D-dimères (EP)", "Gazométrie artérielle"],
        "treatment_ref":  "OMS Guidelines 2023 — Protocol DOTS pour tuberculose. GOLD pour BPCO. "
                          "Protocoles nationaux de traitement de la pneumonie.",
        "who_guideline":  "https://www.who.int/publications/i/item/9789240073029",
        "id":          "module_5_pulmoscan",
        "name":        "PulmoScan AI",
        "icon":        "🫁",
        "color":       "#2980B9",
        "specialty":   "Pneumologie · Radiologie thoracique",
        "description": (
            "Analyse automatisée de radiographies thoraciques et scanners pulmonaires. "
            "Détecte pneumonies, tuberculose, COVID-19, cancer pulmonaire et maladies obstructives."
        ),
        "input_type":  "image",
        "formats":     ["PNG", "JPG", "JPEG", "TIFF", "DICOM"],
        "classes":     ["Normal", "Pneumonie", "Tuberculose", "COVID-19", "Cancer pulmonaire", "BPCO"],
        "dataset_ref": "NIH Chest X-ray Dataset (112 120 images) + CheXpert",
        "model_arch":  "DenseNet121 + EfficientNet-B4",
        "metrics_ref": {"accuracy": "94.2%", "auc": "0.97", "sensitivity": "91.8%"},
        "clinical_classes": {
            "Normal":           {"color": "#27AE60", "urgency": "Faible",   "action": "Suivi annuel recommandé"},
            "Pneumonie":        {"color": "#E67E22", "urgency": "Élevée",   "action": "Antibiothérapie ciblée urgente"},
            "Tuberculose":      {"color": "#C0392B", "urgency": "Critique", "action": "Isolement + DOTS 6 mois"},
            "COVID-19":         {"color": "#8E44AD", "urgency": "Élevée",   "action": "Isolement + suivi saturation O2"},
            "Cancer pulmonaire":{"color": "#922B21", "urgency": "Critique", "action": "TDM + PET-scan + biopsie urgents"},
            "BPCO":             {"color": "#F39C12", "urgency": "Modérée",  "action": "Spirométrie + bronchodilatateurs"},
        },
    },

    "derm": {
        "id":          "module_6_derm",
        "name":        "DermAI",
        "icon":        "🩺",
        "color":       "#27AE60",
        "specialty":   "Dermatologie · Oncologie cutanée",
        "description": (
            "Classification automatisée des lésions cutanées. "
            "Détecte mélanome, carcinomes, psoriasis, eczéma et autres dermatoses."
        ),
        "input_type":  "image",
        "formats":     ["PNG", "JPG", "JPEG"],
        "classes":     ["Bénin", "Mélanome", "Carcinome basocellulaire", "Carcinome spinocellulaire",
                        "Kératose actinique", "Nævus", "Psoriasis", "Eczéma"],
        "dataset_ref": "ISIC Archive 2020 (33 000+ images) + HAM10000",
        "model_arch":  "EfficientNet-B4 + Vision Transformer",
        "metrics_ref": {"accuracy": "91.5%", "auc": "0.96", "sensitivity": "89.3%"},
        "clinical_classes": {
            "Bénin":                    {"color": "#27AE60", "urgency": "Faible",   "action": "Surveillance annuelle"},
            "Mélanome":                 {"color": "#922B21", "urgency": "Critique", "action": "Biopsie excisionnelle urgente + staging"},
            "Carcinome basocellulaire": {"color": "#E74C3C", "urgency": "Élevée",   "action": "Exérèse chirurgicale planifiée"},
            "Carcinome spinocellulaire":{"color": "#C0392B", "urgency": "Élevée",   "action": "Biopsie + exérèse + ganglions sentinelles"},
            "Kératose actinique":       {"color": "#F39C12", "urgency": "Modérée",  "action": "Cryothérapie ou crème 5-FU"},
            "Nævus":                    {"color": "#27AE60", "urgency": "Faible",   "action": "Dermoscopie de surveillance"},
            "Psoriasis":                {"color": "#E67E22", "urgency": "Modérée",  "action": "Traitement topique + photothérapie"},
            "Eczéma":                   {"color": "#F39C12", "urgency": "Faible",   "action": "Émollients + corticoïdes topiques"},
        },
    },

    "retina": {
        "id":          "module_7_retina",
        "name":        "RetinaVision AI",
        "icon":        "👁️",
        "color":       "#8E44AD",
        "specialty":   "Ophtalmologie · Rétinologie",
        "description": (
            "Analyse automatisée du fond d'œil. "
            "Détecte rétinopathie diabétique, glaucome, DMLA et décollement de rétine."
        ),
        "input_type":  "image",
        "formats":     ["PNG", "JPG", "JPEG"],
        "classes":     ["Normal", "RD Grade 1", "RD Grade 2", "RD Grade 3", "RD Grade 4",
                        "Glaucome", "DMLA", "Œdème maculaire"],
        "dataset_ref": "EyePACS (88 702 images) + APTOS 2019 + ORIGA",
        "model_arch":  "ResNet50 + Attention U-Net",
        "metrics_ref": {"accuracy": "93.1%", "auc": "0.98", "sensitivity": "95.2%"},
        "clinical_classes": {
            "Normal":          {"color": "#27AE60", "urgency": "Faible",   "action": "Contrôle annuel — glycémie surveillée"},
            "RD Grade 1":      {"color": "#F39C12", "urgency": "Modérée",  "action": "Contrôle 6 mois + optimisation HbA1c"},
            "RD Grade 2":      {"color": "#E67E22", "urgency": "Modérée",  "action": "Consultation ophtalmo urgente"},
            "RD Grade 3":      {"color": "#E74C3C", "urgency": "Élevée",   "action": "Laser photocoagulation"},
            "RD Grade 4":      {"color": "#922B21", "urgency": "Critique", "action": "Vitrectomie — risque cécité"},
            "Glaucome":        {"color": "#8E44AD", "urgency": "Élevée",   "action": "Tonométrie + collyres + suivi IOP"},
            "DMLA":            {"color": "#C0392B", "urgency": "Élevée",   "action": "Injection anti-VEGF intravitréenne"},
            "Œdème maculaire": {"color": "#E74C3C", "urgency": "Élevée",   "action": "Anti-VEGF + corticoïdes intravitréens"},
        },
    },

    "cardio": {
        "id":          "module_8_cardio",
        "name":        "CardioSense AI",
        "icon":        "❤️",
        "color":       "#E74C3C",
        "specialty":   "Cardiologie · Électrocardiographie",
        "description": (
            "Analyse d'ECG et paramètres cardiaques. "
            "Détecte fibrillation auriculaire, infarctus, insuffisance cardiaque et arythmies."
        ),
        "input_type":  "parameters",
        "params": [
            {"key": "age", "label": "Âge (ans)", "type": "number", "min": 0, "max": 120, "default": 55},
            {"key": "heart_rate", "label": "Fréquence cardiaque (bpm)", "type": "number", "min": 20, "max": 300, "default": 75},
            {"key": "systolic_bp", "label": "TA systolique (mmHg)", "type": "number", "min": 50, "max": 250, "default": 130},
            {"key": "diastolic_bp", "label": "TA diastolique (mmHg)", "type": "number", "min": 30, "max": 150, "default": 80},
            {"key": "cholesterol", "label": "Cholestérol total (mmol/L)", "type": "float", "min": 1.0, "max": 15.0, "default": 5.2},
            {"key": "glucose", "label": "Glycémie (mmol/L)", "type": "float", "min": 1.0, "max": 30.0, "default": 5.5},
            {"key": "smoking", "label": "Tabagisme", "type": "select", "options": ["Non", "Ex-fumeur", "Oui"]},
            {"key": "diabetes", "label": "Diabète", "type": "select", "options": ["Non", "Oui"]},
            {"key": "sex", "label": "Sexe", "type": "select", "options": ["M", "F"]},
        ],
        "classes":     ["Risque faible", "Risque modéré", "Risque élevé", "Risque très élevé",
                        "FA probable", "IDM aigu", "Insuffisance cardiaque"],
        "dataset_ref": "Framingham Heart Study + PhysioNet ECG (47 000+ patients)",
        "model_arch":  "XGBoost + LSTM (ECG temporal)",
        "metrics_ref": {"accuracy": "88.7%", "auc": "0.94", "sensitivity": "87.5%"},
        "clinical_classes": {
            "Risque faible":       {"color": "#27AE60", "urgency": "Faible",   "action": "Mode de vie sain — contrôle annuel"},
            "Risque modéré":       {"color": "#F39C12", "urgency": "Modérée",  "action": "Bilan lipidique + ECG de stress"},
            "Risque élevé":        {"color": "#E67E22", "urgency": "Élevée",   "action": "Statines + aspirine + coronarographie"},
            "Risque très élevé":   {"color": "#E74C3C", "urgency": "Élevée",   "action": "Hospitalisation — urgence cardiologique"},
            "FA probable":         {"color": "#8E44AD", "urgency": "Élevée",   "action": "Anticoagulation + cardioversion"},
            "IDM aigu":            {"color": "#922B21", "urgency": "Critique", "action": "SAMU 15 — angioplastie primaire urgente"},
            "Insuffisance cardiaque":{"color": "#C0392B","urgency": "Élevée",  "action": "Diurétiques + IEC + bêtabloquants"},
        },
    },

    "neuro": {
        "id":          "module_9_neuro",
        "name":        "NeuroVision AI",
        "icon":        "🧠",
        "color":       "#E67E22",
        "specialty":   "Neurologie · Neuroradiologie",
        "description": (
            "Analyse d'IRM et scanners cérébraux. "
            "Détecte tumeurs cérébrales, AVC, Alzheimer, épilepsie et sclérose en plaques."
        ),
        "input_type":  "image",
        "formats":     ["PNG", "JPG", "JPEG", "TIFF", "DICOM"],
        "classes":     ["Normal", "Tumeur bénigne", "Tumeur maligne", "AVC ischémique",
                        "Hémorragie", "Alzheimer", "Sclérose en plaques", "Épilepsie"],
        "dataset_ref": "BraTS 2023 + ADNI + OASIS-3",
        "model_arch":  "3D U-Net + Swin Transformer",
        "metrics_ref": {"accuracy": "92.4%", "auc": "0.96", "sensitivity": "91.0%"},
        "clinical_classes": {
            "Normal":           {"color": "#27AE60", "urgency": "Faible",   "action": "Suivi neurologique standard"},
            "Tumeur bénigne":   {"color": "#F39C12", "urgency": "Élevée",   "action": "Neurochirurgie planifiée + suivi IRM"},
            "Tumeur maligne":   {"color": "#922B21", "urgency": "Critique", "action": "RCP neuro-oncologique urgente + biopsie"},
            "AVC ischémique":   {"color": "#E74C3C", "urgency": "Critique", "action": "Thrombolyse IV < 4.5h ou thrombectomie"},
            "Hémorragie":       {"color": "#C0392B", "urgency": "Critique", "action": "Neurochirurgie urgente — SAMU"},
            "Alzheimer":        {"color": "#8E44AD", "urgency": "Modérée",  "action": "Bilan cognitif + inhibiteurs cholinestérase"},
            "Sclérose en plaques":{"color":"#2980B9","urgency": "Élevée",   "action": "Interféron bêta + suivi IRM 6 mois"},
            "Épilepsie":        {"color": "#E67E22", "urgency": "Modérée",  "action": "EEG + antiépileptiques adaptés"},
        },
    },

    "gastro": {
        "id":          "module_10_gastro",
        "name":        "GastroAI",
        "icon":        "🔬",
        "color":       "#16A085",
        "specialty":   "Gastro-entérologie · Endoscopie IA",
        "description": (
            "Analyse automatisée d'images endoscopiques. "
            "Détecte polypes, ulcères, cancers colorectaux, gastrite et œsophage de Barrett."
        ),
        "input_type":  "image",
        "formats":     ["PNG", "JPG", "JPEG"],
        "classes":     ["Normal", "Polype bénin", "Polype malin", "Ulcère gastrique",
                        "Cancer colorectal", "Gastrite", "Barrett", "Maladie de Crohn"],
        "dataset_ref": "Kvasir-SEG + CVC-ClinicDB + SUN-SEG (50 000+ images)",
        "model_arch":  "YOLOv8 + EfficientNet-B3",
        "metrics_ref": {"accuracy": "95.3%", "auc": "0.97", "sensitivity": "94.1%"},
        "clinical_classes": {
            "Normal":            {"color": "#27AE60", "urgency": "Faible",   "action": "Coloscopie de surveillance 5 ans"},
            "Polype bénin":      {"color": "#F39C12", "urgency": "Modérée",  "action": "Polypectomie endoscopique"},
            "Polype malin":      {"color": "#E74C3C", "urgency": "Élevée",   "action": "Biopsie + résection chirurgicale + staging"},
            "Ulcère gastrique":  {"color": "#E67E22", "urgency": "Modérée",  "action": "IPP + triple thérapie H. pylori"},
            "Cancer colorectal": {"color": "#922B21", "urgency": "Critique", "action": "RCP oncologique + chirurgie +/- chimio"},
            "Gastrite":          {"color": "#F39C12", "urgency": "Faible",   "action": "IPP + régime + sérologie H. pylori"},
            "Barrett":           {"color": "#C0392B", "urgency": "Élevée",   "action": "Surveillance 3 mois + ablation par radiofréquence"},
            "Maladie de Crohn":  {"color": "#8E44AD", "urgency": "Modérée",  "action": "Biothérapies (anti-TNF) + corticoïdes"},
        },
    },

    "histopath": {
        "id":          "module_11_histopath",
        "name":        "HistoPath AI",
        "icon":        "🧫",
        "color":       "#9B59B6",
        "specialty":   "Anatomopathologie · Histologie numérique",
        "description": (
            "Classification automatisée de lames histologiques. "
            "Analyse tissu tumoral, grade, invasion vasculaire et marges chirurgicales."
        ),
        "input_type":  "image",
        "formats":     ["PNG", "JPG", "JPEG", "TIFF", "SVS"],
        "classes":     ["Tissu sain", "Grade 1", "Grade 2", "Grade 3",
                        "Carcinome in situ", "Carcinome invasif", "Invasion vasculaire", "Marge atteinte"],
        "dataset_ref": "TCGA PanCancer + PCam + Camelyon17 (400 000+ patches)",
        "model_arch":  "ViT + DenseNet121 + Attention MIL",
        "metrics_ref": {"accuracy": "96.1%", "auc": "0.98", "sensitivity": "95.4%"},
        "clinical_classes": {
            "Tissu sain":        {"color": "#27AE60", "urgency": "Faible",   "action": "Pas d'intervention requise"},
            "Grade 1":           {"color": "#F39C12", "urgency": "Modérée",  "action": "Surveillance rapprochée"},
            "Grade 2":           {"color": "#E67E22", "urgency": "Élevée",   "action": "Traitement adjuvant à discuter"},
            "Grade 3":           {"color": "#E74C3C", "urgency": "Critique", "action": "Chimiothérapie adjuvante urgente"},
            "Carcinome in situ": {"color": "#C0392B", "urgency": "Élevée",   "action": "Exérèse large + marges saines"},
            "Carcinome invasif": {"color": "#922B21", "urgency": "Critique", "action": "RCP multidisciplinaire urgent"},
            "Invasion vasculaire":{"color":"#8E44AD", "urgency": "Critique", "action": "Staging ganglionnaire + curage"},
            "Marge atteinte":    {"color": "#E74C3C", "urgency": "Élevée",   "action": "Ré-excision chirurgicale"},
        },
    },

    "osteo": {
        "id":          "module_12_osteo",
        "name":        "OsteoDetect AI",
        "icon":        "🦴",
        "color":       "#795548",
        "specialty":   "Rhumatologie · Radiologie musculo-squelettique",
        "description": (
            "Analyse de radiographies osseuses. "
            "Détecte fractures, ostéoporose, arthrose, ostéosarcome et nécrose osseuse."
        ),
        "input_type":  "image",
        "formats":     ["PNG", "JPG", "JPEG", "TIFF", "DICOM"],
        "classes":     ["Normal", "Fracture", "Ostéoporose légère", "Ostéoporose sévère",
                        "Arthrose", "Ostéosarcome", "Nécrose avasculaire", "Métastase osseuse"],
        "dataset_ref": "MURA Stanford + RSNA Bone Age (50 000+ radiographies)",
        "model_arch":  "ResNet50 + DenseNet169",
        "metrics_ref": {"accuracy": "90.8%", "auc": "0.95", "sensitivity": "91.3%"},
        "clinical_classes": {
            "Normal":              {"color": "#27AE60", "urgency": "Faible",   "action": "Bilan osseux annuel si > 50 ans"},
            "Fracture":            {"color": "#E74C3C", "urgency": "Élevée",   "action": "Immobilisation + plâtre/chirurgie selon type"},
            "Ostéoporose légère":  {"color": "#F39C12", "urgency": "Modérée",  "action": "Calcium + Vit D + bisphosphonates"},
            "Ostéoporose sévère":  {"color": "#C0392B", "urgency": "Élevée",   "action": "Dénosumab + prévention chutes"},
            "Arthrose":            {"color": "#E67E22", "urgency": "Modérée",  "action": "AINS + kiné + prothèse si sévère"},
            "Ostéosarcome":        {"color": "#922B21", "urgency": "Critique", "action": "Biopsie urgente + centre spécialisé sarcomes"},
            "Nécrose avasculaire": {"color": "#8E44AD", "urgency": "Élevée",   "action": "Décharge + bisphosphonates + prothèse"},
            "Métastase osseuse":   {"color": "#E74C3C", "urgency": "Critique", "action": "Radio-oncologie + biphosphonates IV"},
        },
    },

    "sepsis": {
        "id":          "module_13_sepsis",
        "name":        "SepsisPredict AI",
        "icon":        "🚨",
        "color":       "#C0392B",
        "specialty":   "Réanimation · Médecine d'urgence",
        "description": (
            "Prédiction précoce du sepsis par analyse des paramètres vitaux et biologiques. "
            "Basé sur le score NEWS2 + algorithme ML sur données MIMIC-IV."
        ),
        "input_type":  "parameters",
        "params": [
            {"key": "temperature", "label": "Température (°C)", "type": "float", "min": 33.0, "max": 43.0, "default": 38.2},
            {"key": "heart_rate",  "label": "Fréquence cardiaque (bpm)", "type": "number", "min": 20, "max": 250, "default": 110},
            {"key": "resp_rate",   "label": "Fréquence respiratoire (/min)", "type": "number", "min": 5, "max": 60, "default": 22},
            {"key": "spo2",        "label": "SpO2 (%)", "type": "number", "min": 50, "max": 100, "default": 95},
            {"key": "systolic_bp", "label": "TA systolique (mmHg)", "type": "number", "min": 50, "max": 250, "default": 90},
            {"key": "gcs",         "label": "Score GCS", "type": "number", "min": 3, "max": 15, "default": 13},
            {"key": "wbc",         "label": "Leucocytes (G/L)", "type": "float", "min": 0.5, "max": 50.0, "default": 14.5},
            {"key": "lactate",     "label": "Lactate (mmol/L)", "type": "float", "min": 0.1, "max": 20.0, "default": 2.8},
            {"key": "creatinine",  "label": "Créatinine (µmol/L)", "type": "number", "min": 20, "max": 1500, "default": 110},
        ],
        "classes":     ["Risque faible", "SIRS", "Sepsis", "Sepsis sévère", "Choc septique"],
        "dataset_ref": "MIMIC-IV (40 000+ patients) + eICU CRD",
        "model_arch":  "XGBoost + LightGBM + LSTM temporal",
        "metrics_ref": {"accuracy": "89.4%", "auc": "0.93", "sensitivity": "91.2%", "specificity": "87.6%"},
        "clinical_classes": {
            "Risque faible":  {"color": "#27AE60", "urgency": "Faible",   "action": "Surveillance horaire — protocole standard"},
            "SIRS":           {"color": "#F39C12", "urgency": "Modérée",  "action": "Remplissage + prélèvements bactério + antibio"},
            "Sepsis":         {"color": "#E67E22", "urgency": "Élevée",   "action": "Antibio IV < 1h + 30 mL/kg cristalloïdes"},
            "Sepsis sévère":  {"color": "#E74C3C", "urgency": "Critique", "action": "Réanimation + vasopresseurs + transfusion si Hb < 7"},
            "Choc septique":  {"color": "#922B21", "urgency": "EXTRÊME",  "action": "RÉANIMATION URGENTE — noradrénaline + hydrocortisone"},
        },
    },

    "hepato": {
        "id":          "module_14_hepato",
        "name":        "HepatoScan AI",
        "icon":        "🫀",
        "color":       "#E67E22",
        "specialty":   "Hépatologie · Imagerie hépatique",
        "description": (
            "Analyse d'échographies et IRM hépatiques. "
            "Détecte cirrhose, hépatites, carcinome hépatocellulaire et stéatose."
        ),
        "input_type":  "image",
        "formats":     ["PNG", "JPG", "JPEG", "TIFF", "DICOM"],
        "classes":     ["Normal", "Stéatose légère", "Stéatose sévère", "Fibrose", "Cirrhose",
                        "Hépatite virale", "Carcinome hépatocellulaire", "Métastase hépatique"],
        "dataset_ref": "TCGA-LIHC + LiTS 2017 + CHAOS (8 000+ images)",
        "model_arch":  "ResNet50 + 3D CNN",
        "metrics_ref": {"accuracy": "89.6%", "auc": "0.94", "sensitivity": "88.4%"},
        "clinical_classes": {
            "Normal":                   {"color": "#27AE60", "urgency": "Faible",   "action": "Bilan hépatique annuel"},
            "Stéatose légère":          {"color": "#F39C12", "urgency": "Faible",   "action": "Régime + arrêt alcool + exercice"},
            "Stéatose sévère":          {"color": "#E67E22", "urgency": "Modérée",  "action": "Bilan hépatique + élastographie"},
            "Fibrose":                  {"color": "#E74C3C", "urgency": "Modérée",  "action": "Traitement étiologique + suivi Fibroscan"},
            "Cirrhose":                 {"color": "#C0392B", "urgency": "Élevée",   "action": "Prévention CHC + TIPS si HTP"},
            "Hépatite virale":          {"color": "#8E44AD", "urgency": "Élevée",   "action": "Antiviraux directs (VHC) ou ténofovir (VHB)"},
            "Carcinome hépatocellulaire":{"color":"#922B21","urgency": "Critique",  "action": "RCP oncologie + résection/RFA/transplantation"},
            "Métastase hépatique":      {"color": "#E74C3C", "urgency": "Critique", "action": "Staging + chimiothérapie systémique"},
        },
    },

    "nephro": {
        "id":          "module_15_nephro",
        "name":        "NephroAI",
        "icon":        "💧",
        "color":       "#2980B9",
        "specialty":   "Néphrologie · Médecine interne",
        "description": (
            "Prédiction et stadification de la maladie rénale chronique. "
            "Analyse DFG, créatinine, protéinurie et paramètres cliniques selon KDIGO."
        ),
        "input_type":  "parameters",
        "params": [
            {"key": "age",         "label": "Âge (ans)", "type": "number", "min": 0, "max": 120, "default": 60},
            {"key": "sex",         "label": "Sexe", "type": "select", "options": ["M", "F"]},
            {"key": "creatinine",  "label": "Créatinine (µmol/L)", "type": "number", "min": 30, "max": 2000, "default": 150},
            {"key": "urea",        "label": "Urée (mmol/L)", "type": "float", "min": 1.0, "max": 50.0, "default": 12.5},
            {"key": "proteinuria", "label": "Protéinurie (g/24h)", "type": "float", "min": 0.0, "max": 20.0, "default": 0.8},
            {"key": "hemoglobin",  "label": "Hémoglobine (g/dL)", "type": "float", "min": 4.0, "max": 20.0, "default": 11.5},
            {"key": "potassium",   "label": "Potassium (mmol/L)", "type": "float", "min": 2.0, "max": 8.0, "default": 4.8},
            {"key": "phosphate",   "label": "Phosphate (mmol/L)", "type": "float", "min": 0.2, "max": 4.0, "default": 1.4},
            {"key": "diabetes",    "label": "Diabète", "type": "select", "options": ["Non", "Oui"]},
            {"key": "hypertension","label": "HTA", "type": "select", "options": ["Non", "Oui"]},
        ],
        "classes":     ["MRC Stade 1", "MRC Stade 2", "MRC Stade 3a", "MRC Stade 3b", "MRC Stade 4", "MRC Stade 5 — IRC terminale"],
        "dataset_ref": "NHANES + CKD-EPI equations + KDIGO Guidelines",
        "model_arch":  "XGBoost + Régression logistique ordinal",
        "metrics_ref": {"accuracy": "91.2%", "auc": "0.96", "sensitivity": "90.5%"},
        "clinical_classes": {
            "MRC Stade 1":            {"color": "#27AE60", "urgency": "Faible",   "action": "DFG > 90 — néphroprotection + suivi annuel"},
            "MRC Stade 2":            {"color": "#2ECC71", "urgency": "Faible",   "action": "DFG 60-89 — IEC/ARA2 + contrôle PA"},
            "MRC Stade 3a":           {"color": "#F39C12", "urgency": "Modérée",  "action": "DFG 45-59 — néphrologue tous les 6 mois"},
            "MRC Stade 3b":           {"color": "#E67E22", "urgency": "Modérée",  "action": "DFG 30-44 — préparation épuration extrarénale"},
            "MRC Stade 4":            {"color": "#E74C3C", "urgency": "Élevée",   "action": "DFG 15-29 — éducation dialyse + FAV"},
            "MRC Stade 5 — IRC terminale":{"color":"#922B21","urgency":"Critique","action": "DFG < 15 — dialyse ou transplantation rénale"},
        },
    },

    "hemato": {
        "id":          "module_16_hemato",
        "name":        "HematoVision AI",
        "icon":        "🩸",
        "color":       "#C0392B",
        "specialty":   "Hématologie · Cytologie sanguine",
        "description": (
            "Analyse automatisée de frottis sanguins et biopsies médullaires. "
            "Détecte anémies, leucémies, lymphomes et thrombocytopénies."
        ),
        "input_type":  "image",
        "formats":     ["PNG", "JPG", "JPEG"],
        "classes":     ["Normal", "Anémie ferriprive", "Anémie hémolytique", "LLA",
                        "LMC", "LNH", "Myélome multiple", "Thrombocytopénie"],
        "dataset_ref": "Blood Cell Count Dataset + ALL-IDB + CellaVision (18 000+ images)",
        "model_arch":  "EfficientNet-B0 + Faster R-CNN",
        "metrics_ref": {"accuracy": "93.7%", "auc": "0.97", "sensitivity": "92.8%"},
        "clinical_classes": {
            "Normal":              {"color": "#27AE60", "urgency": "Faible",   "action": "NFS annuelle si facteurs de risque"},
            "Anémie ferriprive":   {"color": "#F39C12", "urgency": "Modérée",  "action": "Fer oral + recherche étiologique"},
            "Anémie hémolytique":  {"color": "#E67E22", "urgency": "Modérée",  "action": "Bilan auto-immun + transfusion si Hb < 7"},
            "LLA":                 {"color": "#922B21", "urgency": "Critique", "action": "Hématologie urgente + chimio d'induction"},
            "LMC":                 {"color": "#C0392B", "urgency": "Élevée",   "action": "Imatinib (ITK) + cytogénétique"},
            "LNH":                 {"color": "#8E44AD", "urgency": "Élevée",   "action": "Biopsie ganglionnaire + PET-TDM + staging"},
            "Myélome multiple":    {"color": "#E74C3C", "urgency": "Élevée",   "action": "Protéinurie Bence Jones + bortézomib"},
            "Thrombocytopénie":    {"color": "#E67E22", "urgency": "Modérée",  "action": "Éviter AINS + transfusion si < 10 G/L"},
        },
    },

    "gyno": {
        "id":          "module_17_gyno",
        "name":        "GynoCare AI",
        "icon":        "🎗️",
        "color":       "#D91E7A",
        "specialty":   "Gynécologie · Oncologie gynécologique",
        "description": (
            "Analyse d'images gynécologiques et échographies pelviennes. "
            "Détecte cancer du col, ovaires polykystiques, fibromes, endométriose et grossesse ectopique."
        ),
        "input_type":  "image",
        "formats":     ["PNG", "JPG", "JPEG", "TIFF"],
        "classes":     ["Normal", "Cancer du col", "SOPK", "Fibrome utérin",
                        "Endométriose", "Kyste ovarien", "Grossesse ectopique", "Cancer de l'ovaire"],
        "dataset_ref": "GynNet + SIPaKMeD (cervical) + TVUS Dataset (5 000+ images)",
        "model_arch":  "EfficientNet-B3 + U-Net segmentation",
        "metrics_ref": {"accuracy": "91.8%", "auc": "0.95", "sensitivity": "90.2%"},
        "clinical_classes": {
            "Normal":              {"color": "#27AE60", "urgency": "Faible",   "action": "Frottis cervico-vaginal tous les 3 ans"},
            "Cancer du col":       {"color": "#922B21", "urgency": "Critique", "action": "Biopsie urgente + conisation + staging"},
            "SOPK":                {"color": "#F39C12", "urgency": "Modérée",  "action": "Metformine + pilule + suivi hormonal"},
            "Fibrome utérin":      {"color": "#E67E22", "urgency": "Modérée",  "action": "Ulipristal ou myomectomie si symptomatique"},
            "Endométriose":        {"color": "#8E44AD", "urgency": "Modérée",  "action": "Progestérone + laparoscopie diagnostique"},
            "Kyste ovarien":       {"color": "#E74C3C", "urgency": "Élevée",   "action": "Surveillance 3 mois ou kystectomie si > 5 cm"},
            "Grossesse ectopique": {"color": "#922B21", "urgency": "Critique", "action": "URGENCE CHIRURGICALE — salpingotomie"},
            "Cancer de l'ovaire":  {"color": "#C0392B", "urgency": "Critique", "action": "CA-125 + chirurgie cytoréductrice + chimio"},
        },
    },
}


# ── Informations cliniques étendues par module ───────────────────────────────

CLINICAL_EXTENDED: dict[str, dict[str, Any]] = {
    "pulmoscan": {
        "icd10":          "J18 (Pneumonie) · A15 (Tuberculose) · U07 (COVID-19) · C34 (Cancer broncho-pulmonaire) · J44 (BPCO)",
        "prevalence":     "Pneumonie : 2,5M décès/an · Tuberculose : 10M cas/an · Cancer pulmonaire : 1,8M décès/an (OMS 2022)",
        "africa_context": "Tuberculose = 2ème cause de mortalité infectieuse en Afrique subsaharienne. Pneumonie = 1ère cause décès < 5 ans.",
        "risk_factors":   ["Tabagisme (80% cancers pulmonaires)", "VIH/SIDA", "Pollution atmosphérique",
                           "Exposition amiante/silice", "Dénutrition", "Habitat surpeuplé"],
        "diagnosis":      ["Radiographie thoracique", "TDM thoracique HR", "Bacilloscopie + PCR GeneXpert",
                           "Bronchoscopie + BAL", "Spirométrie (BPCO)", "Gazométrie artérielle"],
        "treatment_ref":  "OMS Protocol DOTS (tuberculose 6 mois) · GOLD Guidelines BPCO · Protocoles nationaux pneumonie",
    },
    "derm": {
        "icd10":          "C43 (Mélanome) · C44 (Carcinomes cutanés) · L40 (Psoriasis) · L20 (Eczéma) · D22 (Nævus)",
        "prevalence":     "Mélanome : 325 000 nouveaux cas/an · Carcinome basocellulaire : cancer le plus fréquent de l'homme",
        "africa_context": "Carcinomes cutanés rares sur peau noire mais souvent diagnostiqués tard. Mycosis fongoïde plus fréquent.",
        "risk_factors":   ["Exposition UV/soleil", "Phototype clair (Fitzpatrick I-II)", "Antécédents familiaux mélanome",
                           "Immunodépression", "Expositions chimiques", "Cicatrices chroniques (Marjolin)"],
        "diagnosis":      ["Dermatoscopie", "Biopsie excisionnelle", "Examen anatomopathologique",
                           "Ganglion sentinelle (mélanome)", "IHC (marqueurs tumoraux)"],
        "treatment_ref":  "ESMO Guidelines mélanome 2022 · Règle ABCDE (Asymétrie/Bords/Couleur/Diamètre/Évolution)",
    },
    "retina": {
        "icd10":          "H36.0 (Rétinopathie diabétique) · H40 (Glaucome) · H35.3 (DMLA) · H33 (Décollement rétine)",
        "prevalence":     "Rétinopathie diabétique : 103M personnes touchées (2020) · Glaucome : 80M (2020) · DMLA : 196M (2020)",
        "africa_context": "Cataracte = 1ère cause de cécité en Afrique. Rétinopathie diabétique sous-diagnostiquée (diabète méconnu).",
        "risk_factors":   ["Diabète non contrôlé (HbA1c > 8%)", "HTA", "Obésité", "Âge > 50 ans",
                           "Antécédents familiaux glaucome", "Myopie forte"],
        "diagnosis":      ["Fond d'œil avec rétinographie", "OCT (tomographie cohérence optique)",
                           "Angiographie fluorescéinique", "Champ visuel (périmétrie)", "Tonométrie (PIO)"],
        "treatment_ref":  "OMS Prevention of Blindness · AAO Diabetic Retinopathy Guidelines 2022 · ETDRS protocol",
    },
    "cardio": {
        "icd10":          "I21 (IDM) · I48 (FA) · I50 (Insuffisance cardiaque) · I10 (HTA) · I25 (Coronaropathie)",
        "prevalence":     "Maladies cardiovasculaires = 1ère cause de mortalité mondiale (17,9M décès/an · OMS 2022)",
        "africa_context": "HTA sous-traitée en Afrique (< 30% contrôlée). IDM souvent tardif. Cardiomyopathie dilatée fréquente.",
        "risk_factors":   ["HTA", "Dyslipidémie", "Diabète", "Tabagisme", "Obésité abdominale",
                           "Sédentarité", "Antécédents familiaux coronariens", "Stress chronique"],
        "diagnosis":      ["ECG 12 dérivations", "Écho-Doppler cardiaque", "Troponine I/T + BNP",
                           "Coronarographie", "Holter ECG 24h", "Score de Framingham"],
        "treatment_ref":  "ESC Guidelines 2022 (SCA, HTA, FA) · Score GRACE (SCA) · Score CHA2DS2-VASc (FA)",
    },
    "neuro": {
        "icd10":          "I63 (AVC ischémique) · I61 (Hémorragie cérébrale) · C71 (Tumeur cérébrale) · G30 (Alzheimer) · G35 (SEP)",
        "prevalence":     "AVC : 12,2M nouveaux cas/an · Alzheimer : 50M personnes · Tumeurs cérébrales : 308 000/an",
        "africa_context": "AVC = 1ère cause de handicap acquis en Afrique. Accès limité à l'IRM. Traitement thrombolytique rare.",
        "risk_factors":   ["HTA (principal FDR AVC)", "FA", "Diabète", "Tabagisme", "Dyslipidémie",
                           "Traumatismes crâniens (tumeurs)", "Prédispositions génétiques (Alzheimer APOE4)"],
        "diagnosis":      ["IRM cérébrale (FLAIR, DWI, T1/T2)", "TDM cérébrale en urgence (AVC)",
                           "PET-scan (Alzheimer)", "EEG (épilepsie)", "Biopsie stéréotaxique (tumeur)"],
        "treatment_ref":  "ESO Stroke Guidelines 2021 · ESMO CNS Tumors 2022 · EAN Alzheimer guidelines",
    },
    "gastro": {
        "icd10":          "K25 (Ulcère gastrique) · C18 (Cancer côlon) · K57 (Diverticulose) · K50 (Crohn) · K22.1 (Barrett)",
        "prevalence":     "Cancer colorectal : 1,9M nouveaux cas/an (3ème cancer mondial) · H. pylori : 50% population mondiale",
        "africa_context": "Cancer colorectal en hausse en Afrique subsaharienne. Coloscopie peu disponible. Parasitoses digestives fréquentes.",
        "risk_factors":   ["Alimentation pauvre en fibres", "Obésité", "Alcool", "Tabac",
                           "H. pylori (ulcère + cancer gastrique)", "Antécédents familiaux polypes/CCR", "MICI"],
        "diagnosis":      ["Coloscopie + biopsie", "Endoscopie haute (FOGD)", "TDM abdominale",
                           "ACE + CA 19-9 (marqueurs)", "Coproculture + PCR H. pylori", "Coloscopie virtuelle"],
        "treatment_ref":  "ESMO Colorectal Cancer 2022 · ECCO Guidelines Crohn 2022 · ACG H. pylori guidelines",
    },
    "histopath": {
        "icd10":          "C00-C97 (Tumeurs malignes) · D00-D09 (Carcinomes in situ) · D10-D36 (Tumeurs bénignes)",
        "prevalence":     "18,1M nouveaux cancers diagnostiqués/an (GLOBOCAN 2022) · 50% diagnostics via histopathologie",
        "africa_context": "Pathologie numérique peu déployée. Délais diagnostics longs (4-8 semaines). Manque de pathologistes (1/500 000).",
        "risk_factors":   ["Exposition carcinogènes", "Infections oncogènes (HPV, VHB, VHC, EBV, H. pylori)",
                           "Prédispositions génétiques", "Tabac/alcool", "Rayonnements ionisants", "Inflammation chronique"],
        "diagnosis":      ["Biopsie (ponction, chirurgicale)", "IHC (immunohistochimie)", "FISH (cytogénétique)",
                           "NGS (biologie moléculaire)", "PCR", "Cytologie exfoliative"],
        "treatment_ref":  "OMS Classification of Tumours 2022 · IARC Guidelines · CAP Protocols (College of American Pathologists)",
    },
    "osteo": {
        "icd10":          "M80-M82 (Ostéoporose) · S72 (Fracture fémur) · M15-M19 (Arthrose) · C40-C41 (Ostéosarcome)",
        "prevalence":     "Ostéoporose : 200M femmes touchées · Fracture hanche : 1,5M/an · Arthrose : 250M personnes",
        "africa_context": "Fractures ostéoporotiques sous-diagnostiquées. Ostéosarcome fréquent chez l'enfant/adolescent en Afrique.",
        "risk_factors":   ["Âge > 65 ans", "Ménopause précoce", "Carence calcium/Vit D", "Immobilisation",
                           "Corticoïdes au long cours", "Alcool", "Faible IMC", "Antécédents fractures"],
        "diagnosis":      ["Radiographie standard", "Ostéodensitométrie (DXA)", "Score T (ostéoporose)",
                           "IRM (nécrose, tumeurs)", "TDM (fractures complexes)", "Biopsie osseuse (sarcome)", "Bilan phosphocalcique"],
        "treatment_ref":  "IOF Clinical Guidelines 2022 · FRAX Risk Assessment · WHO Fracture Liaison Service",
    },
    "sepsis": {
        "icd10":          "A41 (Sepsis) · R65.2 (Sepsis sévère) · R57.2 (Choc septique) · R65.1 (SIRS)",
        "prevalence":     "Sepsis : 48,9M cas/an · 11M décès/an (20% mortalité mondiale) · 85% dans pays à faibles revenus",
        "africa_context": "Sepsis = urgence vitale sous-reconnue. Hémocultures souvent indisponibles. Mortalité > 40% en Afrique.",
        "risk_factors":   ["Immunodépression (VIH, chimio)", "Diabète", "Chirurgie récente", "Cathéters/sondes",
                           "Grand âge", "Dénutrition", "Antibiorésistance", "Accouchement (sepsis puerpéral)"],
        "diagnosis":      ["Hémocultures ×2 (avant antibio)", "NFS + PCT + CRP + lactate", "Bilan rénal + hépatique",
                           "Gaz du sang (lactate > 2 mmol/L = sepsis)", "ECBU + ECBC", "Score SOFA + qSOFA"],
        "treatment_ref":  "Surviving Sepsis Campaign 2021 · Bundle 1h (antibio < 1h, 30mL/kg cristalloïdes, cultures)",
    },
    "hepato": {
        "icd10":          "K70 (Hépatopathie alcoolique) · K74 (Fibrose/Cirrhose) · C22 (CHC) · B18 (Hépatite chronique)",
        "prevalence":     "Hépatite B : 296M porteurs · Hépatite C : 58M porteurs · CHC : 2ème cancer le plus meurtrier",
        "africa_context": "VHB endémique en Afrique (> 8% portage). CHC = 1ère cause de décès par cancer en Afrique subsaharienne.",
        "risk_factors":   ["VHB/VHC chronique", "Alcool", "Stéatohépatite (NASH)", "Aflatoxines (arachides moisies)",
                           "Hémochromatose", "Wilson", "Cirrhose (quelle qu'en soit la cause)"],
        "diagnosis":      ["Échographie hépatique + Fibroscan", "Bilan hépatique (ASAT/ALAT/GGT/PAL/bilirubine)",
                           "AFP (CHC)", "Biopsie hépatique", "IRM hépatique (CHC)", "Virologie (AgHBs, ARN VHC)"],
        "treatment_ref":  "EASL Clinical Guidelines 2022 (VHB, VHC, NASH, CHC) · BCLC Staging (CHC)",
    },
    "nephro": {
        "icd10":          "N18 (MRC) · N17 (IRA) · N20 (Lithiase) · N04 (Syndrome néphrotique) · E11.22 (Néphropathie diabétique)",
        "prevalence":     "MRC : 850M personnes touchées · 2ème cause de décès prématuré dans les pays à faibles revenus",
        "africa_context": "HTA et diabète = principales causes MRC en Afrique. Dialyse peu accessible (< 10% des patients). Néphropathies endémiques (drépanocytose).",
        "risk_factors":   ["Diabète", "HTA", "Glomérulonéphrites", "Drépanocytose", "Néphrotoxiques (AINS, aminosides)",
                           "Obésité", "Antécédents familiaux", "VIH"],
        "diagnosis":      ["Créatinine + DFG (CKD-EPI)", "Rapport albumine/créatinine urinaire", "ECBU + protéinurie 24h",
                           "Échographie rénale", "Biopsie rénale (glomérulonéphrites)", "Ionogramme + bicarbonates"],
        "treatment_ref":  "KDIGO CKD Guidelines 2022 · K/DOQI · ISPD (dialyse péritonéale) · UNOS (transplantation)",
    },
    "hemato": {
        "icd10":          "D50 (Anémie ferriprive) · C91 (LLA) · C92 (LMC) · C85 (LNH) · C90 (Myélome) · D69.6 (Thrombocytopénie)",
        "prevalence":     "Anémie : 1,6 milliard personnes (OMS) · LLA : 1ère leucémie de l'enfant · Drépanocytose : 300 000/an",
        "africa_context": "Anémie falciforme (drépanocytose) = maladie génétique la plus fréquente en Afrique. Paludisme = 1ère cause anémie infantile.",
        "risk_factors":   ["Carence en fer/B12/folates", "Paludisme chronique", "Drépanocytose (génétique)",
                           "Radiations ionisantes (leucémies)", "Exposition benzène", "VIH/EBV/HTLV-1", "Âge > 65 ans (myélome)"],
        "diagnosis":      ["NFS + frottis sanguin + réticulocytes", "Ferritine + transferrine", "Myélogramme + biopsie médullaire",
                           "Caryotype (LMC: t9;22 Philadelphia)", "Immunophénotypage (flow cytometry)", "Hémoglobine électrophorèse (drépanocytose)"],
        "treatment_ref":  "EHA Guidelines 2022 · WHO Anemia Protocols · ESMO Lymphoma/Leukemia Guidelines",
    },
    "gyno": {
        "icd10":          "C53 (Cancer col utérin) · N80 (Endométriose) · D25 (Fibrome utérin) · E28.2 (SOPK) · O00 (Grossesse ectopique)",
        "prevalence":     "Cancer col : 604 000 nouveaux cas/an (2ème cancer femme dans PED) · SOPK : 10% femmes en âge de procréer",
        "africa_context": "Cancer du col = 1er cancer de la femme en Afrique subsaharienne (faible couverture dépistage et vaccination HPV).",
        "risk_factors":   ["HPV (types 16/18 dans 70% cancers col)", "Multiparité", "Tabagisme", "Immunodépression",
                           "Obésité (cancer endomètre)", "Antécédents ATCD familiaux", "Absence de dépistage"],
        "diagnosis":      ["Frottis cervico-vaginal (PCR HPV)", "Colposcopie + biopsie dirigée", "Échographie pelvienne",
                           "IRM pelvienne", "CA-125 (cancer ovaire)", "Test de grossesse + écho (GEU)"],
        "treatment_ref":  "OMS HPV Vaccination Guidelines 2022 · FIGO Staging (cancers gynéco) · ESGO Guidelines 2022",
    },
}

# Fusion des infos étendues dans MODULES
for _k, _ext in CLINICAL_EXTENDED.items():
    if _k in MODULES:
        MODULES[_k].update(_ext)


# ── Logique de prédiction scaffold ──────────────────────────────────────────

def scaffold_predict(
    module_key: str,
    image_path: str | None = None,
    params: dict | None = None,
) -> dict[str, Any]:
    """
    Retourne une prédiction scaffold pour les modules sans modèle entraîné.
    Simule une réponse clinique réaliste avec la classe la plus probable.
    """
    t0 = time.time()
    cfg = MODULES.get(module_key)
    if not cfg:
        return {"status": "unknown_module", "error": f"Module '{module_key}' inconnu"}

    classes = cfg["classes"]
    rng     = random.Random(42)

    # Génère des probabilités réalistes (une classe dominante + distribution)
    dominant_idx = rng.randint(0, len(classes) - 1)
    raw = [rng.uniform(0.02, 0.15) for _ in classes]
    raw[dominant_idx] = rng.uniform(0.55, 0.85)
    total  = sum(raw)
    probs  = {cls: round(p / total, 4) for cls, p in zip(classes, raw)}
    pred   = max(probs, key=probs.get)
    conf   = probs[pred]

    clinical = cfg["clinical_classes"].get(pred, {})

    return {
        "module":          cfg["id"],
        "module_name":     cfg["name"],
        "task":            f"{cfg['specialty']} — analyse automatisée",
        "model_arch":      cfg["model_arch"],
        "model_version":   "v1.0-scaffold",
        "request_id":      str(uuid.uuid4()),
        "input_image":     Path(image_path).name if image_path else None,
        "input_params":    params,

        "prediction":      pred,
        "confidence":      conf,
        "probabilities":   probs,
        "risk_level":      clinical.get("urgency", "—"),
        "recommended_action": clinical.get("action", "—"),

        "clinical_info": {
            "classes":     classes,
            "urgency":     clinical.get("urgency", "—"),
            "action":      clinical.get("action", "—"),
            "color":       clinical.get("color", "#5E7A8A"),
            "dataset_ref": cfg["dataset_ref"],
            "metrics_ref": cfg["metrics_ref"],
        },

        "explainability":  {"method": "Grad-CAM", "status": "not_generated"},
        "deployment_mode": "scaffold — model training in progress",
        "processing_ms":   round((time.time() - t0) * 1000 + 150),
        "status":          "scaffold_ready",
    }
