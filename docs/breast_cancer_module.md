# Module Breast Cancer AI

Ce module ajoute a KANEA une detection du cancer du sein a partir de mammographies.

## Classes

- `Normal`
- `Benign`
- `Malignant`

## Fichiers principaux

- `modules/breast_cancer_dl/predictor.py`
- `scripts/train_breast_cancer_model.py`
- `data/breast_cancer/README.md`

## Entrainement

```powershell
python scripts/train_breast_cancer_model.py --dataset-dir data/breast_cancer
```

## Sortie

- `models/deep_learning/breast_cancer_model.pth`
