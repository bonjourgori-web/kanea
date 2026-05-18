# Strategie de donnees KANEA

Cette strategie suit la logique que tu as definie pour KANEA :

- **Module 1 - Paludisme** : utiliser prioritairement le dataset NIH
- **Module 2 - Nutrition** : utiliser DHS / SMART / OMS
- **Module 3 - Medico-legal** : commencer avec des donnees simulees realistes, puis remplacer par des donnees de reference si disponibles

## 1. Module 1 - Donnees paludisme

### Dataset principal recommande

- **NIH Malaria Dataset**
- environ `27 500` images
- classes : `Parasitised` / `Uninfected`
- usage : classification binaire sur images microscopiques

Lien de reference :

- `https://lhncbc.nlm.nih.gov/publications/datasets/malaria-datasets.html`

### Datasets complementaires utiles

- **VGG Malaria Dataset**
  - `https://www.robots.ox.ac.uk/~vgg/data/malaria/`
- **Kaggle Malaria Dataset**
  - `https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria`

### Format attendu dans KANEA

```text
data/malaria/
|-- Parasitised/
`-- Uninfected/
```

ou

```text
data/malaria/
|-- train/
|   |-- Parasitised/
|   `-- Uninfected/
`-- valid/
    |-- Parasitised/
    `-- Uninfected/
```

## 2. Module 2 - Donnees nutritionnelles

### Sources recommandees

- **DHS - Demographic and Health Surveys**
  - `https://dhsprogram.com/`
- **SMART Survey / UNICEF**
  - `https://smartmethodology.org/`
- **WHO Child Growth Standards**
  - `https://www.who.int/tools/child-growth-standards`

### Variables de base recommandees

- `age_months`
- `weight_kg`
- `height_cm`
- `sex`
- `muac_cm`
- `waz`
- `haz`
- `whz`

### Exemple minimal

```csv
age_months,weight_kg,height_cm,sex,muac_cm
24,10.5,82,M,12.0
36,12.0,90,F,13.5
```

## 3. Module 3 - Donnees medico-legales

### Sources possibles

- FORDISC / jeux craniometriques specialises
- jeux de mesures craniennes ou squelettiques issus de publications, GitHub ou Kaggle

### Contrainte importante

Les donnees publiques medico-legales robustes sont plus rares. Pour un demarrage realiste de KANEA, il est acceptable de :

1. commencer avec des donnees simulees plausibles
2. valider le pipeline technique
3. remplacer ensuite par des donnees de reference mieux documentees

### Strategie recommandee pour KANEA

- phase 1 : donnees simulees realistes pour `BioID AI`
- phase 2 : calibration sur jeux africains ou sous-representes si disponibles
- phase 3 : evaluation scientifique et verification des biais

## 4. Bonnes pratiques

- nettoyer les donnees avant entrainement
- equilibrer les classes si possible
- documenter les sources et les limites
- utiliser augmentation d'image pour le module malaria
- utiliser validation croisee et evaluation hors echantillon pour les modules tabulaires
- garder `multibio_predict()` comme couche d'integration finale

## 5. Strategie synthetique retenue

Pour KANEA, la ligne directrice la plus pragmatique est :

- `NIH` pour le paludisme
- `DHS` et `SMART` pour la nutrition
- `donnees simulees` pour BioID au depart

Ensuite :

1. entrainer les trois modules separement
2. exporter les modeles
3. integrer les sorties avec `multibio_predict()`
