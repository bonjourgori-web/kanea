# Demarrage rapide KANEA

## 1. Preparer l'environnement

Sous Windows PowerShell :

```powershell
cd C:\Users\HP\gori\KANEA
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

Installation manuelle possible :

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Verifier les assets et les donnees

- placer le logo dans `assets/branding/kanea_logo.png`
- verifier les dossiers :
  - `data/malaria/`
  - `data/biometry/`
  - `data/forensic/`
- utiliser les gabarits :
  - `data/biometry/biometry_dataset_template.csv`
  - `data/forensic/forensic_dataset_template.csv`
- si besoin, preparer l'arborescence et un jeu medico-legal simule :

```powershell
python scripts/prepare_datasets.py --generate-forensic-demo --forensic-rows 500
```

## 3. Lancer le dashboard

```powershell
.\venv\Scripts\Activate.ps1
python scripts/run_dashboard.py
```

Acces local :

- `http://localhost:8501`

## 4. Lancer l'API

Dans un autre terminal :

```powershell
.\venv\Scripts\Activate.ps1
python scripts/run_api.py
```

Acces local :

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

## 5. Entrainer les modeles

Module 1 :

```powershell
python scripts/train_malaria_model.py --dataset-dir data/malaria
```

Module 2 :

```powershell
python scripts/train_biometry_model.py --csv-path data/biometry/biometry_dataset.csv
```

Module 3 :

```powershell
python scripts/train_forensic_model.py --csv-path data/forensic/forensic_dataset.csv
```

## 6. Validation rapide

- ouvrir le dashboard et verifier l'affichage du hero KANEA
- verifier que l'API repond sur `/health`
- tester `POST /predict/biometry`
- tester `POST /predict/forensic`
- tester `POST /predict/malaria/upload` quand un modele est disponible
