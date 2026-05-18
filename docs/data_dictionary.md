# Dictionnaire de donnees KANEA

## Module 2 - Biometry

Fichier recommande :

- `data/biometry/biometry_dataset.csv`

Gabarit fourni :

- `data/biometry/biometry_dataset_template.csv`

Colonnes attendues :

- `age_months`
  - entier
  - age en mois
- `weight_kg`
  - numerique
  - poids en kilogrammes
- `height_cm`
  - numerique
  - taille ou longueur en centimetres
- `sex`
  - categoriel
  - `M` ou `F`
- `muac_cm`
  - numerique
  - perimetre brachial en centimetres
- `waz`
  - numerique
  - z-score poids-pour-age
- `haz`
  - numerique
  - z-score taille-pour-age
- `whz`
  - numerique
  - z-score poids-pour-taille
- `nutrition_status`
  - cible
  - classes conseillees :
    - `severe_undernutrition`
    - `moderate_undernutrition`
    - `normal`
    - `overweight`
    - `obesity`

Notes :

- le script calcule aussi automatiquement `bmi`
- si certains z-scores manquent, le pipeline peut imputer numeriquement, mais il vaut mieux fournir des valeurs fiables

## Module 3 - Forensic

Fichier recommande :

- `data/forensic/forensic_dataset.csv`

Gabarit fourni :

- `data/forensic/forensic_dataset_template.csv`

Colonnes de features attendues :

- `max_cranial_length_mm`
- `max_cranial_breadth_mm`
- `bizygomatic_breadth_mm`
- `nasal_height_mm`
- `nasal_breadth_mm`
- `basion_nasion_length_mm`
- `femur_length_cm`
- `tibia_length_cm`
- `humerus_length_cm`
- `radius_length_cm`
- `AIM_PC1`
- `AIM_PC2`
- `AIM_PC3`

Colonnes cibles attendues :

- `biological_sex`
  - ex. `M`, `F`
- `age_at_death`
  - numerique
  - en annees
- `ancestry`
  - categoriel
  - ex. `West_African`, `Central_African`, `East_African`
- `stature_cm`
  - numerique
  - stature estimee ou observee en centimetres

Notes :

- les noms de colonnes doivent correspondre exactement au script
- `AIM_PC1`, `AIM_PC2`, `AIM_PC3` peuvent provenir d'une PCA effectuee en amont sur les marqueurs AIMs
- pour un usage scientifique, les categories d'ascendance doivent etre definies avec prudence et justifiees par la base de reference
