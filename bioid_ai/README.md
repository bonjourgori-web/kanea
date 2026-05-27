# BioID AI — Module d'Estimation du Profil Biologique Forensique

**Plateforme KANÉA — Intelligence Artificielle Médicale — Côte d'Ivoire**

---

## Vue d'ensemble

Le module **BioID AI** est le moteur d'analyse forensique de la plateforme KANÉA. Il estime automatiquement le **profil biologique** d'un individu à partir de mesures anthropométriques :

| Paramètre | Méthode |
|-----------|---------|
| Sexe biologique | VotingClassifier (Random Forest + Gradient Boosting) |
| Âge au décès | VotingRegressor (RF + GB + ElasticNet) |
| Ascendance biogéographique | Random Forest + PCA sur marqueurs AIMs |
| Stature | Régression linéaire + formule Trotter & Gleser (1958) |

---

## Structure du module

```
bioid_ai/
├── __init__.py                    # Exports principaux
├── api/
│   ├── __init__.py
│   ├── main.py                    # Application FastAPI
│   ├── security.py                # JWT, OAuth2, AES-256, sanitisation
│   ├── routes/
│   │   ├── __init__.py
│   │   └── bioid.py               # Endpoints REST
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── bioid_schemas.py       # Modèles Pydantic v2
│   └── services/
│       ├── __init__.py
│       └── bioid_service.py       # Orchestration des prédictions
├── utils/
│   ├── __init__.py
│   ├── feature_engineering.py     # Construction des features
│   ├── pca_analysis.py            # Analyse PCA (AIMs)
│   ├── pdf_generator.py           # Rapports PDF médico-légaux
│   └── visualizations.py         # Graphiques professionnels
├── training/
│   ├── __init__.py
│   └── train.py                   # Script d'entraînement
├── visualizations/                # Graphiques générés (auto-créé)
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Prérequis

- Python 3.10+
- Environnement virtuel recommandé

```bash
cd KANEA
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 2. Installation des dépendances

```bash
pip install -r bioid_ai/requirements.txt
```

---

## Utilisation

### Étape 1 : Entraîner les modèles

```bash
cd KANEA
python bioid_ai/training/train.py
```

Le script :
- Génère 500 échantillons synthétiques réalistes (basés sur FORDISC)
- Entraîne les 4 modèles avec cross-validation 5-fold
- Sauvegarde le bundle dans `KANEA/models/machine_learning/bioid_bundle.pkl`
- Affiche les métriques (accuracy, F1, MAE, R²)

Si un fichier `KANEA/data/bioid_dataset.csv` existe, il sera utilisé à la place des données synthétiques.

### Étape 2 : Lancer l'API

```bash
cd KANEA
uvicorn bioid_ai.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Ou directement :

```bash
python bioid_ai/api/main.py
```

L'API sera accessible sur :
- Documentation interactive : http://localhost:8000/docs
- ReDoc : http://localhost:8000/redoc
- Health check : http://localhost:8000/api/v2/bioid/health

### Étape 3 : Exemple d'appel API

#### Analyse complète (Python)

```python
import requests

payload = {
    "case_id": "CASE-2024-001",
    "examiner": "Dr. Koné Mamadou",
    "cranial": {
        "GOL": 182.0,
        "XCB": 139.0,
        "BBH": 135.0,
        "ZYB": 128.0,
        "NLH": 52.0,
        "NLB": 26.5,
        "OBH": 34.0,
        "OBB": 41.0
    },
    "postcranial": {
        "femur_max_length": 460.0,
        "femur_bicondylar": 455.0,
        "femur_head_diam": 46.0
    }
}

response = requests.post(
    "http://localhost:8000/api/v2/bioid/predict/bioid",
    json=payload
)
result = response.json()
print(result)
```

#### Réponse attendue

```json
{
  "case_id": "CASE-2024-001",
  "timestamp": "2024-01-15T10:30:00Z",
  "predictions": {
    "biological_sex": "Male",
    "sex_confidence": 0.892,
    "age_at_death": 45.3,
    "age_range": "37–53 ans",
    "ancestry": "Africaine",
    "ancestry_confidence": 0.754,
    "stature_cm": 172.4,
    "stature_method": "Trotter & Gleser (1958)"
  },
  "status": "success"
}
```

#### Génération d'un rapport PDF

```python
response = requests.post(
    "http://localhost:8000/api/v2/bioid/predict/bioid?generate_report=true",
    json=payload
)
result = response.json()
print("Rapport PDF :", result["report_path"])
```

---

## Variables mesurées

### Mesures crâniennes (15 variables FORDISC)

| Variable | Description | Unité |
|----------|-------------|-------|
| GOL | Glabello-occipital length | mm |
| XCB | Maximum cranial breadth | mm |
| BBH | Basion-bregma height | mm |
| ZYB | Bizygomatic breadth | mm |
| AUB | Biauricular breadth | mm |
| ASB | Biasterionic breadth | mm |
| BNL | Basion-nasion length | mm |
| BPL | Basion-prosthion length | mm |
| NLH | Nasal height | mm |
| NLB | Nasal breadth | mm |
| OBH | Orbital height | mm |
| OBB | Orbital breadth | mm |
| MAB | Maxillary breadth | mm |
| FOL | Foramen magnum length | mm |
| FOB | Foramen magnum breadth | mm |

### Mesures post-crâniennes (8 variables)

| Variable | Description | Unité |
|----------|-------------|-------|
| femur_max_length | Longueur max fémur | mm |
| femur_bicondylar | Longueur bicondylaire fémur | mm |
| tibia_length | Longueur tibia | mm |
| humerus_max_length | Longueur max humérus | mm |
| radius_max_length | Longueur max radius | mm |
| fibula_max_length | Longueur max fibula | mm |
| femur_head_diam | Diamètre tête fémur | mm |
| humerus_head_diam | Diamètre tête humérus | mm |

### Indices dérivés (calculés automatiquement)

| Indice | Formule |
|--------|---------|
| cephalic_idx | (XCB / GOL) × 100 |
| nasal_idx | (NLB / NLH) × 100 |
| orbital_idx | (OBH / OBB) × 100 |
| femur_ratio | femur_max_length / femur_head_diam |

---

## Sécurité

- **JWT** : Tokens signés HS256 (python-jose)
- **Chiffrement** : AES-256 via Fernet (cryptography)
- **Rate limiting** : slowapi (100 req/min par défaut)
- **Sanitisation** : Validation anti-injection SQL/XSS/path traversal
- **Audit logs** : Toutes les opérations loggées dans `kanea.audit`

### Variables d'environnement

```bash
KANEA_SECRET_KEY=your-super-secret-key-here
KANEA_DEV_MODE=false          # true = accès sans token (dev uniquement)
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

---

## Rapports PDF

Le rapport médico-légal généré contient :
- En-tête KANÉA avec identification du dossier
- Résumé médico-légal encadré
- Tableau des prédictions avec probabilités
- Graphiques intégrés (radar, résumé, importance des features)
- QR code de vérification
- Signature numérique SHA-256
- Disclaimer médical et pagination

---

## Références scientifiques

- **Ousley, S.D. & Jantz, R.L.** (1996). FORDISC 3.0: Personal Computer Forensic Discriminant Functions
- **Trotter, M. & Gleser, G.C.** (1958). A re-evaluation of estimation of stature based on measurements of stature taken during life and of long bones after death
- **Giles, E. & Elliot, O.** (1963). Sex determination by discriminant function analysis of crania

---

## Licence

Propriétaire — KANÉA — Plateforme IA Médicale — Côte d'Ivoire

---

*BioID AI v2.0.0 — Ce module est destiné à un usage professionnel en anthropologie forensique. Les estimations ne constituent pas un diagnostic médical définitif.*
