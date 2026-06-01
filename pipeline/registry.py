"""
KANÉA — Universal Medical Data Sources Registry
================================================
Registre central de toutes les sources de données médicales pour les 15 modules IA.
Phase 1 du Universal Medical Data Acquisition & Auto-Training Pipeline.

Sources : NIH · TCIA · PhysioNet · Kaggle · Zenodo · ISIC · MIMIC · OpenNeuro · TCGA
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataSource:
    name: str
    url: str
    api_type: str          # kaggle | physionet | tcia | zenodo | direct | custom
    license: str
    requires_auth: bool
    formats: list[str]
    size_estimate_gb: float
    description: str
    api_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleRegistry:
    module_id: str
    name: str
    modalities: list[str]  # image | tabular | ecg | wsi | dicom | cytology
    sources: list[DataSource]
    classes: list[str]
    task_type: str         # classification | detection | segmentation | regression
    framework: str         # pytorch | sklearn | monai | fastai
    architecture: str
    input_shape: tuple[int, ...]
    target_metrics: dict[str, float]


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRE DES 15 MODULES
# ═══════════════════════════════════════════════════════════════════════════════

REGISTRY: dict[str, ModuleRegistry] = {

    # ── MODULE 1 — MALARIASCAN AI ─────────────────────────────────────────────
    "malaria": ModuleRegistry(
        module_id="module_01_malaria",
        name="MalariaScan AI",
        modalities=["image"],
        sources=[
            DataSource(
                name="NIH Malaria Cell Images",
                url="https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria",
                api_type="kaggle",
                license="CC0",
                requires_auth=True,
                formats=["PNG"],
                size_estimate_gb=0.35,
                description="27 558 images microscopiques (Parasitised/Uninfected)",
                api_params={"dataset": "iarunava/cell-images-for-detecting-malaria"},
            ),
            DataSource(
                name="Malaria Cell Images Dataset — Kaggle",
                url="https://www.kaggle.com/datasets/miracle9to9/files1",
                api_type="kaggle",
                license="CC0",
                requires_auth=True,
                formats=["PNG", "JPG"],
                size_estimate_gb=0.5,
                description="Jeu de données frottis sanguins paludisme",
                api_params={"dataset": "miracle9to9/files1"},
            ),
            DataSource(
                name="Zenodo Malaria Collections",
                url="https://zenodo.org/search?q=malaria+blood+smear",
                api_type="zenodo",
                license="CC BY 4.0",
                requires_auth=False,
                formats=["PNG", "TIFF"],
                size_estimate_gb=2.0,
                description="Collections Zenodo frottis paludisme",
                api_params={"q": "malaria blood smear", "type": "dataset"},
            ),
        ],
        classes=["Parasitised", "Uninfected"],
        task_type="classification",
        framework="fastai",
        architecture="ResNet34",
        input_shape=(3, 224, 224),
        target_metrics={"accuracy": 0.96, "f1": 0.958, "auc": 0.97},
    ),

    # ── MODULE 2 — NUTRITRACK AI ─────────────────────────────────────────────
    "nutrition": ModuleRegistry(
        module_id="module_02_nutrition",
        name="NutriTrack AI",
        modalities=["tabular"],
        sources=[
            DataSource(
                name="WHO/UNICEF SMART Survey Data",
                url="https://www.who.int/data/nutrition/nlis",
                api_type="direct",
                license="CC BY-NC 4.0",
                requires_auth=False,
                formats=["CSV", "XLSX"],
                size_estimate_gb=0.05,
                description="Données anthropométriques nutrition Afrique subsaharienne",
                api_params={},
            ),
            DataSource(
                name="DHS Program Nutrition Surveys",
                url="https://dhsprogram.com/data/available-datasets.cfm",
                api_type="custom",
                license="DHS",
                requires_auth=True,
                formats=["CSV", "STATA"],
                size_estimate_gb=0.5,
                description="DHS nutritional surveys — Côte d'Ivoire / Afrique",
                api_params={"country": "CI", "survey_type": "DHS"},
            ),
            DataSource(
                name="UNICEF MICS Dataset",
                url="https://mics.unicef.org/surveys",
                api_type="custom",
                license="UNICEF Open",
                requires_auth=True,
                formats=["SPSS", "CSV"],
                size_estimate_gb=0.3,
                description="MICS — indicateurs nutritionnels enfants",
                api_params={},
            ),
        ],
        classes=["Normal", "Stunted", "Wasted", "Overweight", "Underweight"],
        task_type="classification",
        framework="sklearn",
        architecture="XGBoost + RandomForest",
        input_shape=(1, 10),
        target_metrics={"accuracy": 0.91, "f1": 0.905, "recall": 0.897},
    ),

    # ── MODULE 3 — BIOID AI ──────────────────────────────────────────────────
    "bioid": ModuleRegistry(
        module_id="module_03_bioid",
        name="BioID AI",
        modalities=["tabular"],
        sources=[
            DataSource(
                name="FORDISC Forensic Dataset",
                url="https://fac.utk.edu/fordisc-3-1-personal-computer-forensic-discriminant-functions/",
                api_type="custom",
                license="Academic",
                requires_auth=True,
                formats=["CSV"],
                size_estimate_gb=0.01,
                description="Mesures ostéométriques craniennes — estimation profil médico-légal",
                api_params={},
            ),
            DataSource(
                name="Howells Craniometric Data",
                url="https://web.utk.edu/~auerbach/HOWL.htm",
                api_type="direct",
                license="Academic Open",
                requires_auth=False,
                formats=["CSV"],
                size_estimate_gb=0.005,
                description="2 500+ crânes — 82 mesures — 28 populations mondiales",
                api_params={},
            ),
        ],
        classes=["Male", "Female", "Young Adult", "Middle Age", "Old Adult"],
        task_type="classification",
        framework="sklearn",
        architecture="SVM + PCA + Random Forest",
        input_shape=(1, 82),
        target_metrics={"accuracy": 0.88, "f1": 0.875},
    ),

    # ── MODULE 4 — BREAST CANCER AI ──────────────────────────────────────────
    "breast_cancer": ModuleRegistry(
        module_id="module_04_breast_cancer",
        name="Breast Cancer AI",
        modalities=["image"],
        sources=[
            DataSource(
                name="CBIS-DDSM — TCIA",
                url="https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM",
                api_type="tcia",
                license="CC BY 3.0",
                requires_auth=False,
                formats=["DICOM"],
                size_estimate_gb=163.6,
                description="Curated Breast Imaging Subset of DDSM — 2 620 études",
                api_params={"collection": "CBIS-DDSM"},
            ),
            DataSource(
                name="VinDr-Mammo",
                url="https://physionet.org/content/vindr-mammo/1.0.0/",
                api_type="physionet",
                license="PhysioNet Credentialed",
                requires_auth=True,
                formats=["DICOM"],
                size_estimate_gb=180.0,
                description="20 000 mammographies — 5 grades BI-RADS",
                api_params={"project": "vindr-mammo", "version": "1.0.0"},
            ),
            DataSource(
                name="RSNA Breast Imaging Challenge",
                url="https://www.kaggle.com/competitions/rsna-breast-cancer-detection",
                api_type="kaggle",
                license="Competition",
                requires_auth=True,
                formats=["DICOM"],
                size_estimate_gb=314.0,
                description="RSNA 2022 — 54 706 patients — détection cancer sein",
                api_params={"competition": "rsna-breast-cancer-detection"},
            ),
        ],
        classes=["Normal", "Benign", "Malignant"],
        task_type="classification",
        framework="pytorch",
        architecture="EfficientNet-B0",
        input_shape=(3, 512, 512),
        target_metrics={"accuracy": 0.94, "auc": 0.96, "f1": 0.93},
    ),

    # ── MODULE 5 — PULMOSCAN AI ───────────────────────────────────────────────
    "pulmoscan": ModuleRegistry(
        module_id="module_05_pulmoscan",
        name="PulmoScan AI",
        modalities=["image", "dicom"],
        sources=[
            DataSource(
                name="NIH ChestX-ray14",
                url="https://www.kaggle.com/datasets/nih-chest-xrays/data",
                api_type="kaggle",
                license="CC0",
                requires_auth=True,
                formats=["PNG"],
                size_estimate_gb=42.0,
                description="112 120 radiographies thoraciques — 14 classes pathologiques",
                api_params={"dataset": "nih-chest-xrays/data"},
            ),
            DataSource(
                name="CheXpert — Stanford",
                url="https://stanfordmlgroup.github.io/competitions/chexpert/",
                api_type="custom",
                license="Stanford Academic",
                requires_auth=True,
                formats=["JPG"],
                size_estimate_gb=439.0,
                description="224 316 radiographies thoraciques — 14 observations",
                api_params={},
            ),
            DataSource(
                name="MIMIC-CXR-JPG",
                url="https://physionet.org/content/mimic-cxr-jpg/2.0.0/",
                api_type="physionet",
                license="PhysioNet Credentialed",
                requires_auth=True,
                formats=["JPG"],
                size_estimate_gb=160.0,
                description="227 827 études radiologiques avec rapports",
                api_params={"project": "mimic-cxr-jpg", "version": "2.0.0"},
            ),
            DataSource(
                name="RSNA Pneumonia Detection",
                url="https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge",
                api_type="kaggle",
                license="Competition",
                requires_auth=True,
                formats=["DICOM"],
                size_estimate_gb=6.5,
                description="30 000 images — détection pneumonie",
                api_params={"competition": "rsna-pneumonia-detection-challenge"},
            ),
        ],
        classes=["Normal", "Pneumonia", "Tuberculosis", "Nodule", "Effusion",
                 "Cardiomegaly", "Atelectasis", "Infiltration"],
        task_type="classification",
        framework="pytorch",
        architecture="DenseNet121",
        input_shape=(1, 224, 224),
        target_metrics={"auc": 0.85, "accuracy": 0.87},
    ),

    # ── MODULE 6 — DERMAI ──────────────────────────────────────────────────────
    "derm": ModuleRegistry(
        module_id="module_06_derm",
        name="DermAI",
        modalities=["image"],
        sources=[
            DataSource(
                name="ISIC 2020 Challenge",
                url="https://www.kaggle.com/competitions/siim-isic-melanoma-classification",
                api_type="kaggle",
                license="CC BY-NC 4.0",
                requires_auth=True,
                formats=["JPG", "DICOM"],
                size_estimate_gb=110.0,
                description="33 126 images dermoscopiques — mélanome",
                api_params={"competition": "siim-isic-melanoma-classification"},
            ),
            DataSource(
                name="HAM10000",
                url="https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection",
                api_type="kaggle",
                license="CC BY-NC 4.0",
                requires_auth=True,
                formats=["JPG"],
                size_estimate_gb=2.7,
                description="10 015 images dermoscopiques — 7 classes",
                api_params={"dataset": "kmader/skin-lesion-analysis-toward-melanoma-detection"},
            ),
            DataSource(
                name="PAD-UFES-20",
                url="https://www.kaggle.com/datasets/mahdavi1202/skin-disease",
                api_type="kaggle",
                license="CC BY 4.0",
                requires_auth=True,
                formats=["PNG"],
                size_estimate_gb=0.45,
                description="2 298 images lésions cutanées — données cliniques",
                api_params={"dataset": "mahdavi1202/skin-disease"},
            ),
        ],
        classes=["Melanoma", "Nevus", "BCC", "AK", "BKL", "DF", "VASC"],
        task_type="classification",
        framework="pytorch",
        architecture="EfficientNet-B4",
        input_shape=(3, 512, 512),
        target_metrics={"auc": 0.93, "accuracy": 0.90},
    ),

    # ── MODULE 7 — RETINAVISION AI ────────────────────────────────────────────
    "retina": ModuleRegistry(
        module_id="module_07_retina",
        name="RetinaVision AI",
        modalities=["image"],
        sources=[
            DataSource(
                name="APTOS 2019 Blindness Detection",
                url="https://www.kaggle.com/competitions/aptos2019-blindness-detection",
                api_type="kaggle",
                license="Competition",
                requires_auth=True,
                formats=["PNG"],
                size_estimate_gb=9.1,
                description="3 662 images fond d'œil — rétinopathie diabétique 0-4",
                api_params={"competition": "aptos2019-blindness-detection"},
            ),
            DataSource(
                name="EyePACS",
                url="https://www.kaggle.com/competitions/diabetic-retinopathy-detection",
                api_type="kaggle",
                license="Competition",
                requires_auth=True,
                formats=["JPG"],
                size_estimate_gb=33.0,
                description="88 702 images rétiniennes — rétinopathie diabétique",
                api_params={"competition": "diabetic-retinopathy-detection"},
            ),
            DataSource(
                name="REFUGE Glaucoma",
                url="https://refuge.grand-challenge.org/",
                api_type="custom",
                license="Academic",
                requires_auth=True,
                formats=["JPG"],
                size_estimate_gb=2.0,
                description="1 200 images fond d'œil — glaucome — segmentation",
                api_params={},
            ),
            DataSource(
                name="IDRiD",
                url="https://www.kaggle.com/datasets/mariaherrerot/idrid",
                api_type="kaggle",
                license="CC BY 4.0",
                requires_auth=True,
                formats=["JPG"],
                size_estimate_gb=1.1,
                description="516 images — rétinopathie + maculopathie — segmentation",
                api_params={"dataset": "mariaherrerot/idrid"},
            ),
        ],
        classes=["Normal", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR",
                 "Glaucoma", "AMD"],
        task_type="classification",
        framework="pytorch",
        architecture="EfficientNet-B3",
        input_shape=(3, 512, 512),
        target_metrics={"kappa": 0.87, "auc": 0.95},
    ),

    # ── MODULE 8 — CARDIOSENSE AI ────────────────────────────────────────────
    "cardio": ModuleRegistry(
        module_id="module_08_cardio",
        name="CardioSense AI",
        modalities=["tabular", "ecg"],
        sources=[
            DataSource(
                name="PTB-XL ECG Dataset",
                url="https://physionet.org/content/ptb-xl/1.0.3/",
                api_type="physionet",
                license="CC BY 4.0",
                requires_auth=False,
                formats=["WFDB", "CSV"],
                size_estimate_gb=2.6,
                description="21 837 ECG 12 dérivations — 71 classes — PTB",
                api_params={"project": "ptb-xl", "version": "1.0.3"},
            ),
            DataSource(
                name="PhysioNet Computing in Cardiology 2017",
                url="https://physionet.org/content/challenge-2017/1.0.0/",
                api_type="physionet",
                license="CC BY 4.0",
                requires_auth=False,
                formats=["WFDB"],
                size_estimate_gb=0.5,
                description="8 528 ECG courtes durées — FA — Normal — Bruit",
                api_params={"project": "challenge-2017", "version": "1.0.0"},
            ),
            DataSource(
                name="MIMIC-IV ECG",
                url="https://physionet.org/content/mimic-iv-ecg/1.0/",
                api_type="physionet",
                license="PhysioNet Credentialed",
                requires_auth=True,
                formats=["WFDB"],
                size_estimate_gb=15.0,
                description="800 000 ECG 12 dérivations — MIMIC-IV",
                api_params={"project": "mimic-iv-ecg", "version": "1.0"},
            ),
        ],
        classes=["Normal", "AF", "MI", "LBBB", "RBBB", "ST-Elevation", "HF"],
        task_type="classification",
        framework="pytorch",
        architecture="ResNet1D + Transformer",
        input_shape=(12, 5000),
        target_metrics={"auc": 0.96, "f1": 0.91},
    ),

    # ── MODULE 9 — NEUROVISION AI ─────────────────────────────────────────────
    "neuro": ModuleRegistry(
        module_id="module_09_neuro",
        name="NeuroVision AI",
        modalities=["image", "dicom"],
        sources=[
            DataSource(
                name="BraTS 2021",
                url="https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1",
                api_type="kaggle",
                license="CC BY 4.0",
                requires_auth=True,
                formats=["NIfTI"],
                size_estimate_gb=108.0,
                description="1 251 patients — tumeurs cérébrales — segmentation 4 régions",
                api_params={"dataset": "dschettler8845/brats-2021-task1"},
            ),
            DataSource(
                name="RSNA Intracranial Hemorrhage",
                url="https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection",
                api_type="kaggle",
                license="Competition",
                requires_auth=True,
                formats=["DICOM"],
                size_estimate_gb=90.0,
                description="674 258 examens — 5 types hémorragie intracrânienne",
                api_params={"competition": "rsna-intracranial-hemorrhage-detection"},
            ),
            DataSource(
                name="ADNI Alzheimer",
                url="https://adni.loni.usc.edu/",
                api_type="custom",
                license="ADNI Data Use Agreement",
                requires_auth=True,
                formats=["DICOM", "NIfTI"],
                size_estimate_gb=200.0,
                description="IRM cérébrales — progression Alzheimer",
                api_params={},
            ),
        ],
        classes=["Normal", "Glioma", "Meningioma", "Metastasis",
                 "Stroke/AVC", "Hemorrhage", "Alzheimer"],
        task_type="segmentation",
        framework="monai",
        architecture="SegResNet / UNETR",
        input_shape=(4, 240, 240, 155),
        target_metrics={"dice": 0.87, "auc": 0.94},
    ),

    # ── MODULE 10 — GASTROAI ──────────────────────────────────────────────────
    "gastro": ModuleRegistry(
        module_id="module_10_gastro",
        name="GastroAI",
        modalities=["image"],
        sources=[
            DataSource(
                name="HyperKvasir",
                url="https://datasets.simula.no/hyper-kvasir/",
                api_type="direct",
                license="CC BY 4.0",
                requires_auth=False,
                formats=["JPG"],
                size_estimate_gb=12.5,
                description="110 079 images endoscopiques — 23 classes — 374 vidéos",
                api_params={"url": "https://datasets.simula.no/hyper-kvasir/hyper-kvasir.zip"},
            ),
            DataSource(
                name="Kvasir-SEG Polyps",
                url="https://datasets.simula.no/kvasir-seg/",
                api_type="direct",
                license="CC BY 4.0",
                requires_auth=False,
                formats=["JPG", "PNG"],
                size_estimate_gb=0.05,
                description="1 000 images polypes avec segmentation",
                api_params={"url": "https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip"},
            ),
            DataSource(
                name="CVC-ClinicDB",
                url="https://polyp.grand-challenge.org/CVCClinicDB/",
                api_type="custom",
                license="Academic",
                requires_auth=True,
                formats=["PNG"],
                size_estimate_gb=0.15,
                description="612 images coloscopie — polyps",
                api_params={},
            ),
        ],
        classes=["Normal", "Polyp", "Ulcer", "Esophagitis", "Barrett",
                 "Hemorrhoids", "Colorectal Cancer"],
        task_type="detection",
        framework="pytorch",
        architecture="YOLOv8 + PVT",
        input_shape=(3, 512, 512),
        target_metrics={"dice": 0.90, "map50": 0.88},
    ),

    # ── MODULE 11 — HISTOPATH AI ─────────────────────────────────────────────
    "histopath": ModuleRegistry(
        module_id="module_11_histopath",
        name="HistoPath AI",
        modalities=["image", "wsi"],
        sources=[
            DataSource(
                name="CAMELYON16",
                url="https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images",
                api_type="kaggle",
                license="CC BY 4.0",
                requires_auth=True,
                formats=["TIF"],
                size_estimate_gb=28.7,
                description="400 WSI seins — métastases ganglions sentinelles",
                api_params={"dataset": "paultimothymooney/breast-histopathology-images"},
            ),
            DataSource(
                name="PANDA Prostate",
                url="https://www.kaggle.com/competitions/prostate-cancer-grade-assessment",
                api_type="kaggle",
                license="Competition",
                requires_auth=True,
                formats=["TIF"],
                size_estimate_gb=44.0,
                description="10 616 biopsies prostate — score Gleason",
                api_params={"competition": "prostate-cancer-grade-assessment"},
            ),
            DataSource(
                name="TCGA Histopathology",
                url="https://portal.gdc.cancer.gov/",
                api_type="tcia",
                license="NIH Open",
                requires_auth=False,
                formats=["SVS", "TIF"],
                size_estimate_gb=500.0,
                description="TCGA — 33 types cancer — WSI + données cliniques",
                api_params={"project": "TCGA"},
            ),
        ],
        classes=["Normal", "Benign", "Grade_1", "Grade_2", "Grade_3",
                 "Malignant", "Metastasis"],
        task_type="classification",
        framework="pytorch",
        architecture="CONCH / UNI / ViT-L",
        input_shape=(3, 224, 224),
        target_metrics={"kappa": 0.88, "auc": 0.97},
    ),

    # ── MODULE 12 — OSTEODETECT AI ────────────────────────────────────────────
    "osteo": ModuleRegistry(
        module_id="module_12_osteo",
        name="OsteoDetect AI",
        modalities=["image", "dicom"],
        sources=[
            DataSource(
                name="MURA Bone Radiographs",
                url="https://stanfordmlgroup.github.io/competitions/mura/",
                api_type="custom",
                license="Stanford Academic",
                requires_auth=True,
                formats=["PNG"],
                size_estimate_gb=8.1,
                description="40 561 images radiographiques — 7 régions anatomiques",
                api_params={},
            ),
            DataSource(
                name="RSNA Bone Age",
                url="https://www.kaggle.com/competitions/rsna-bone-age",
                api_type="kaggle",
                license="Competition",
                requires_auth=True,
                formats=["PNG"],
                size_estimate_gb=1.0,
                description="12 611 radiographies main — estimation âge osseux",
                api_params={"competition": "rsna-bone-age"},
            ),
            DataSource(
                name="RSNA Spine Fracture 2022",
                url="https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection",
                api_type="kaggle",
                license="Competition",
                requires_auth=True,
                formats=["DICOM"],
                size_estimate_gb=130.0,
                description="2 019 patients — fractures rachis cervical",
                api_params={"competition": "rsna-2022-cervical-spine-fracture-detection"},
            ),
        ],
        classes=["Normal", "Fracture", "Osteoarthritis", "Osteoporosis",
                 "Tumor", "Bone Age"],
        task_type="detection",
        framework="pytorch",
        architecture="EfficientDet-D3",
        input_shape=(3, 512, 512),
        target_metrics={"map50": 0.86, "accuracy": 0.90},
    ),

    # ── MODULE 13 — SEPSISPREDICT AI ──────────────────────────────────────────
    "sepsis": ModuleRegistry(
        module_id="module_13_sepsis",
        name="SepsisPredict AI",
        modalities=["tabular"],
        sources=[
            DataSource(
                name="PhysioNet Sepsis Challenge 2019",
                url="https://physionet.org/content/challenge-2019/1.0.0/",
                api_type="physionet",
                license="CC BY 4.0",
                requires_auth=False,
                formats=["PSV"],
                size_estimate_gb=0.2,
                description="40 336 patients — 40 variables — prédiction sepsis",
                api_params={"project": "challenge-2019", "version": "1.0.0"},
            ),
            DataSource(
                name="MIMIC-IV ICU",
                url="https://physionet.org/content/mimiciv/2.2/",
                api_type="physionet",
                license="PhysioNet Credentialed",
                requires_auth=True,
                formats=["CSV"],
                size_estimate_gb=6.9,
                description="431 231 admissions ICU — MIMIC-IV complet",
                api_params={"project": "mimiciv", "version": "2.2"},
            ),
            DataSource(
                name="eICU Collaborative Research Database",
                url="https://physionet.org/content/eicu-crd/2.0/",
                api_type="physionet",
                license="PhysioNet Credentialed",
                requires_auth=True,
                formats=["CSV"],
                size_estimate_gb=5.9,
                description="200 000+ admissions ICU — 208 hôpitaux US",
                api_params={"project": "eicu-crd", "version": "2.0"},
            ),
        ],
        classes=["Normal", "SIRS", "Sepsis", "Severe Sepsis", "Septic Shock"],
        task_type="classification",
        framework="sklearn",
        architecture="XGBoost + LSTM",
        input_shape=(1, 40),
        target_metrics={"auroc": 0.85, "f1": 0.80},
    ),

    # ── MODULE 14 — HEPATOSCAN AI ─────────────────────────────────────────────
    "hepato": ModuleRegistry(
        module_id="module_14_hepato",
        name="HepatoScan AI",
        modalities=["image", "dicom"],
        sources=[
            DataSource(
                name="LiTS — Liver Tumor Segmentation",
                url="https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation",
                api_type="kaggle",
                license="CC BY 4.0",
                requires_auth=True,
                formats=["NIfTI"],
                size_estimate_gb=30.6,
                description="131 scanners + annotations tumeurs hépatiques",
                api_params={"dataset": "andrewmvd/liver-tumor-segmentation"},
            ),
            DataSource(
                name="CHAOS Liver Multi-modal",
                url="https://zenodo.org/record/3431873",
                api_type="zenodo",
                license="CC BY-NC 4.0",
                requires_auth=False,
                formats=["DICOM"],
                size_estimate_gb=0.8,
                description="40 cas — CT + IRM foie — segmentation multi-organes",
                api_params={"record_id": "3431873"},
            ),
            DataSource(
                name="TCIA HCC-TACE-Seg",
                url="https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230229",
                api_type="tcia",
                license="CC BY 4.0",
                requires_auth=False,
                formats=["DICOM"],
                size_estimate_gb=70.0,
                description="105 CHC — pré/post TACE — CT multiparamétriques",
                api_params={"collection": "HCC-TACE-Seg"},
            ),
        ],
        classes=["Normal", "Steatosis", "Cirrhosis", "HCC", "Metastasis",
                 "Cholangiocarcinoma"],
        task_type="segmentation",
        framework="monai",
        architecture="nnU-Net",
        input_shape=(1, 512, 512, 64),
        target_metrics={"dice": 0.88, "auc": 0.93},
    ),

    # ── MODULE 15 — NEPHROAI ──────────────────────────────────────────────────
    "nephro": ModuleRegistry(
        module_id="module_15_nephro",
        name="NephroAI",
        modalities=["tabular", "image"],
        sources=[
            DataSource(
                name="CKD Dataset — UCI ML Repository",
                url="https://www.kaggle.com/datasets/mansoordaku/ckdisease",
                api_type="kaggle",
                license="CC0",
                requires_auth=True,
                formats=["CSV"],
                size_estimate_gb=0.001,
                description="400 patients — 25 variables biologiques — IRC stades",
                api_params={"dataset": "mansoordaku/ckdisease"},
            ),
            DataSource(
                name="MIMIC-IV Nephrology",
                url="https://physionet.org/content/mimiciv/2.2/",
                api_type="physionet",
                license="PhysioNet Credentialed",
                requires_auth=True,
                formats=["CSV"],
                size_estimate_gb=6.9,
                description="Données biologiques rénales — MIMIC-IV",
                api_params={"project": "mimiciv", "version": "2.2"},
            ),
            DataSource(
                name="KiTS23 — Kidney Tumor Segmentation",
                url="https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection",
                api_type="kaggle",
                license="Academic",
                requires_auth=True,
                formats=["NIfTI"],
                size_estimate_gb=50.0,
                description="CT abdominal — rein + tumeurs rénales",
                api_params={"competition": "rsna-2023-abdominal-trauma-detection"},
            ),
        ],
        classes=["Normal", "CKD Stage 1", "CKD Stage 2", "CKD Stage 3",
                 "CKD Stage 4", "CKD Stage 5"],
        task_type="classification",
        framework="sklearn",
        architecture="XGBoost + DFG Calculator",
        input_shape=(1, 25),
        target_metrics={"accuracy": 0.92, "auc": 0.95},
    ),

    # ── MODULE 16 — HEMATOVISION AI ───────────────────────────────────────────
    "hemato": ModuleRegistry(
        module_id="module_16_hemato",
        name="HematoVision AI",
        modalities=["image", "tabular"],
        sources=[
            DataSource(
                name="Raabin-WBC White Blood Cell",
                url="https://www.kaggle.com/datasets/amirreza99/raabin-wbc",
                api_type="kaggle",
                license="CC BY 4.0",
                requires_auth=True,
                formats=["JPG"],
                size_estimate_gb=1.2,
                description="17 965 images leucocytes — 5 classes — frottis sanguins",
                api_params={"dataset": "amirreza99/raabin-wbc"},
            ),
            DataSource(
                name="ALL-IDB Leukemia",
                url="https://homes.di.unimi.it/scotti/all/",
                api_type="direct",
                license="Academic",
                requires_auth=True,
                formats=["JPG"],
                size_estimate_gb=0.09,
                description="ALL-IDB1 + ALL-IDB2 — leucémie lymphoïde aiguë",
                api_params={},
            ),
            DataSource(
                name="BCCD Blood Cell Count",
                url="https://www.kaggle.com/datasets/draaslan/blood-cell-count-and-detection",
                api_type="kaggle",
                license="MIT",
                requires_auth=True,
                formats=["JPG"],
                size_estimate_gb=0.08,
                description="364 images — detection GB/GR/Plt — annotations XML",
                api_params={"dataset": "draaslan/blood-cell-count-and-detection"},
            ),
            DataSource(
                name="PBC Blood Cell Dataset",
                url="https://www.kaggle.com/datasets/pauloaraujo/peripheral-blood-cells",
                api_type="kaggle",
                license="CC BY 4.0",
                requires_auth=True,
                formats=["JPG"],
                size_estimate_gb=1.0,
                description="17 092 images — 8 types cellules sanguines périphériques",
                api_params={"dataset": "pauloaraujo/peripheral-blood-cells"},
            ),
        ],
        classes=["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil",
                 "Basophil", "Blast", "Normal RBC", "Platelet"],
        task_type="classification",
        framework="pytorch",
        architecture="EfficientNet-B2 + YOLOv8",
        input_shape=(3, 224, 224),
        target_metrics={"accuracy": 0.95, "f1": 0.94},
    ),

    # ── MODULE 17 — GYNOCARE AI ───────────────────────────────────────────────
    "gyno": ModuleRegistry(
        module_id="module_17_gyno",
        name="GynoCare AI",
        modalities=["image"],
        sources=[
            DataSource(
                name="SIPaKMeD Cervical Cells",
                url="https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed",
                api_type="kaggle",
                license="CC BY 4.0",
                requires_auth=True,
                formats=["BMP"],
                size_estimate_gb=0.17,
                description="4 049 images cellules cervicales — 5 classes (Bethesda)",
                api_params={"dataset": "prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed"},
            ),
            DataSource(
                name="Herlev Cervical Cytology",
                url="https://mde-lab.aegean.gr/index.php/downloads",
                api_type="direct",
                license="Academic",
                requires_auth=True,
                formats=["BMP"],
                size_estimate_gb=0.03,
                description="917 cellules cervicales — normal vs anormal — 7 classes",
                api_params={},
            ),
            DataSource(
                name="MobileODT Cervical Screening",
                url="https://www.kaggle.com/competitions/intel-mobileodt-cervical-cancer-screening",
                api_type="kaggle",
                license="Competition",
                requires_auth=True,
                formats=["JPG"],
                size_estimate_gb=1.7,
                description="11 514 images col utérin — Type 1/2/3 — transformation zone",
                api_params={"competition": "intel-mobileodt-cervical-cancer-screening"},
            ),
            DataSource(
                name="TCGA Cervical + Ovarian",
                url="https://portal.gdc.cancer.gov/",
                api_type="tcia",
                license="NIH Open",
                requires_auth=False,
                formats=["SVS", "JPG"],
                size_estimate_gb=50.0,
                description="TCGA-CESC + TCGA-OV — histopathologie gynécologique",
                api_params={"project": "TCGA-CESC"},
            ),
        ],
        classes=["Normal", "LSIL", "HSIL", "SCC", "Adenocarcinoma",
                 "Koilocyte", "Dyskeratocyte"],
        task_type="classification",
        framework="pytorch",
        architecture="EfficientNet-B0",
        input_shape=(3, 224, 224),
        target_metrics={"accuracy": 0.93, "f1": 0.91},
    ),
}


def get_module(module_key: str) -> ModuleRegistry | None:
    return REGISTRY.get(module_key)


def list_modules() -> list[str]:
    return list(REGISTRY.keys())


def get_all_sources() -> list[tuple[str, str, DataSource]]:
    """Retourne toutes les sources : (module_key, source_name, DataSource)."""
    out = []
    for key, mod in REGISTRY.items():
        for src in mod.sources:
            out.append((key, src.name, src))
    return out


def get_sources_by_api_type(api_type: str) -> list[tuple[str, DataSource]]:
    return [
        (key, src)
        for key, mod in REGISTRY.items()
        for src in mod.sources
        if src.api_type == api_type
    ]
