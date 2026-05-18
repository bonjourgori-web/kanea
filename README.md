# KANEA

KANEA signifie **Knowledge Anthropology & Neural Engine for Africa**.

KANEA reprend l'architecture de la documentation technique source en conservant le meme socle scientifique et logiciel, avec un nouveau nom d'application. La plateforme est concue comme un outil open-source d'aide a la decision biomedicale et medico-legale pour les contextes africains a ressources limitees.

## Resume executif

KANEA integre quatre modules specialises :

1. `MalariaScan AI`
   - detection automatique du paludisme a partir d'images microscopiques
   - approche : **deep learning**
   - stack documentee : `fast.ai v2 + PyTorch`, `ResNet34`
   - interpretabilite : `Grad-CAM`

2. `NutriTrack AI`
   - analyse biometrique et prediction de l'etat nutritionnel
   - approche : **machine learning tabulaire**
   - stack documentee : `RandomForest` + `XGBoost`
   - calculs derives : `IMC`, `WAZ`, `HAZ`, `WHZ`, `MUAC`

3. `BioID AI`
   - estimation du profil biologique medico-legal
   - approche : **machine learning classique**
   - sorties : sexe biologique, age au deces, ascendance, stature
   - methodes documentees : `scikit-learn`, `PCA`, regressions osteometriques

4. `Breast Cancer AI`
   - detection du cancer du sein sur mammographies
   - approche : **deep learning**
   - stack cible : `PyTorch`, `EfficientNet` ou `ResNet`
   - classes : `Normal`, `Benign`, `Malignant`

Le tout est orchestre par une fonction centrale `multibio_predict()`, expose via un dashboard `Streamlit` et, dans la feuille de route, une API `FastAPI`.

## Architecture fidele au document

- `Model Layer`
  - Module 1 - Malaria : `fast.ai / ResNet34`
  - Module 2 - Biometry : `XGBoost / RandomForest`
  - Module 3 - Forensic : `scikit-learn / ensemble`
  - Module 4 - Breast Cancer : `PyTorch / EfficientNet`
- `Integration Layer`
  - `multibio_predict()`
- `Presentation Layer`
  - `dashboard/app.py` avec `Streamlit`
  - `api/main.py` pour l'exposition REST des modules

## Jeux de donnees references

- `data/malaria/`
  - NIH Cell Image Library
  - 27 558 images
  - classes : `Parasitised` / `Uninfected`
- `data/biometry/`
  - donnees SMART Survey / DHS
  - colonnes attendues : `age_months`, `weight_kg`, `height_cm`, `sex`, `MUAC`
- `data/forensic/`
  - donnees craniometriques et post-craniennes
  - source type : FORDISC ou equivalent africain
- `data/breast_cancer/`
  - mammographies
  - classes attendues : `Normal`, `Benign`, `Malignant`

## Performances documentees

- Module 1 - Malaria
  - accuracy : `96.4 %`
  - F1-score : `95.8 %`
- Module 2 - Biometry
  - accuracy : `91.3 %`
  - recall : `89.7 %`
  - F1-score : `90.5 %`

## Structure du dossier

- `assets/branding/` : logo KANEA et elements visuels
- `api/` : API FastAPI alignee avec la roadmap du projet
- `dashboard/` : application Streamlit
- `data/` : `malaria`, `biometry`, `forensic`
- `docs/` : documentation technique et vues d'ensemble
- `models/deep_learning/` : artefacts deep learning exportes
- `models/machine_learning/` : artefacts machine learning exportes
- `modules/malaria_dl/` : inference du Module 1
- `modules/nutrition_ml/` : inference du Module 2
- `modules/bioid_ml/` : inference du Module 3
- `modules/breast_cancer_dl/` : inference du Module 4
- `modules/integration/` : orchestration multimodale
- `notebooks/` : execution dans l'ordre `01` a `04`
- `tests/` : tests unitaires et tests fumee

## Branding

Le logo fourni doit etre place dans `assets/branding/kanea_logo.png`.

## Lancement local

- guide rapide : `docs/quickstart.md`
- checklist de validation : `docs/local_validation_checklist.md`
- strategie de donnees : `docs/data_sources_strategy.md`
- workflow malaria reel : `docs/real_malaria_workflow.md`
- module cancer du sein : `docs/breast_cancer_module.md`
- prompt Codex cancer du sein : `docs/prompt_codex_kanea_breast_cancer.txt`
- lancement dashboard : `python scripts/run_dashboard.py`
- lancement API : `python scripts/run_api.py`
