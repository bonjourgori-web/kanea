# Scripts KANEA

Scripts disponibles :

- `train_malaria_model.py`
  - entrainement du Module 1 avec `fast.ai v2 + PyTorch`
  - export attendu : `models/deep_learning/malaria_model.pkl`
- `train_biometry_model.py`
  - entrainement du Module 2 avec `RandomForest + XGBoost`
  - export attendu : `models/machine_learning/nutrition_model.pkl`
- `train_forensic_model.py`
  - entrainement du Module 3 medico-legal avec `scikit-learn`, `PCA` et modeles separes
  - export attendu : `models/machine_learning/forensic_model.pkl`
- `run_dashboard.py`
  - lancement du dashboard Streamlit
- `run_api.py`
  - lancement de l'API FastAPI avec Uvicorn en mode stable
- `setup_windows.ps1`
  - creation du venv et installation des dependances sous Windows
- `prepare_datasets.py`
  - preparation des dossiers de donnees
  - generation optionnelle d'un dataset medico-legal simule
- `generate_biometry_demo.py`
  - generation d'un dataset biometrie de demonstration pour le Module 2
- `train_breast_cancer_model.py`
  - entrainement du module cancer du sein sur mammographies
  - export attendu : `models/deep_learning/breast_cancer_model.pth`
- `prepare_malaria_dataset.py`
  - preparation d'un dataset malaria reel avec split `train/valid`
  - generation d'un manifeste `dataset_manifest.csv`

Exemple :

```bash
python scripts/train_malaria_model.py --dataset-dir data/malaria
```

```bash
python scripts/train_biometry_model.py --csv-path data/biometry/biometry_dataset.csv
```

```bash
python scripts/train_forensic_model.py --csv-path data/forensic/forensic_dataset.csv
```

```powershell
python scripts/run_dashboard.py
python scripts/run_api.py
```

```powershell
python scripts/prepare_datasets.py --generate-forensic-demo --forensic-rows 500
```

```powershell
python scripts/generate_biometry_demo.py --rows 600
```

```powershell
python scripts/prepare_malaria_dataset.py --source-dir "C:\chemin\vers\dataset"
```
