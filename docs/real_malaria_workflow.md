# Workflow dataset malaria reel

## 1. Telecharger un dataset

Sources recommandees :

- NIH Malaria Dataset
- VGG Malaria Dataset
- Kaggle Malaria Dataset

Le plus pratique pour KANEA reste un dossier source contenant :

```text
source_dataset/
|-- Parasitised/
`-- Uninfected/
```

## 2. Preparer le dataset pour KANEA

Commande :

```powershell
python scripts/prepare_malaria_dataset.py --source-dir "C:\chemin\vers\dataset"
```

Cette commande :

- verifie les classes `Parasitised` et `Uninfected`
- cree une structure `train/valid`
- copie les images dans `data/malaria/`
- genere `data/malaria/dataset_manifest.csv`

## 3. Entrainer le modele

```powershell
python scripts/train_malaria_model.py --dataset-dir data/malaria
```

## 4. Sortie attendue

- `models/deep_learning/malaria_model.pkl`

## 5. Test dans KANEA

- dashboard Streamlit : onglet `MalariaScan AI`
- API :
  - `POST /predict/malaria`
  - `POST /predict/malaria/upload`
