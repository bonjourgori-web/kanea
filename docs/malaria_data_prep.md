# Preparation des donnees malaria

## Source de reference

Le document KANEA fait reference au dataset NIH Cell Image Library avec deux classes :

- `Parasitised`
- `Uninfected`

## Formats acceptes

Le script d'entrainement malaria fonctionne avec :

1. un dataset classe par dossier :

```text
data/malaria/Parasitised/
data/malaria/Uninfected/
```

2. ou une structure `train/valid` :

```text
data/malaria/train/Parasitised/
data/malaria/train/Uninfected/
data/malaria/valid/Parasitised/
data/malaria/valid/Uninfected/
```

## Manifest facultatif

Un gabarit est fourni ici :

- `data/malaria/dataset_manifest_template.csv`

Ce fichier ne pilote pas encore directement l'entrainement, mais il est utile pour :

- tracer les sources
- documenter la repartition train/valid
- suivre la qualite des images

## Checklist avant entrainement

- dossiers `Parasitised` et `Uninfected` correctement orthographies
- images lisibles par Python
- pas de fichiers corrompus
- jeu de validation distinct si possible
- volume suffisant par classe

## Commande d'entrainement

```powershell
python scripts/train_malaria_model.py --dataset-dir data/malaria
```

## Preparation automatisee conseillee

Si tu recuperes un dataset brut organise avec :

```text
Parasitised/
Uninfected/
```

tu peux preparer le split KANEA avec :

```powershell
python scripts/prepare_malaria_dataset.py --source-dir "C:\chemin\vers\dataset"
```

## Sortie

- `models/deep_learning/malaria_model.pkl`
