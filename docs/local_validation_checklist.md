# Checklist de validation locale

## Environnement

- Python installe
- environnement virtuel `venv` cree
- dependances installees depuis `requirements.txt`

## Branding

- logo present dans `assets/branding/kanea_logo.png`
- dashboard affiche correctement le branding KANEA

## Dashboard

- `streamlit run dashboard/app.py` demarre sans erreur
- onglet `MalariaScan AI` visible
- onglet `Biometry AI` visible
- onglet `BioID AI` visible
- onglet `Integration` visible

## API

 - `uvicorn api.main:app` demarre sans erreur
- `GET /health` retourne `ok`
- `GET /info` retourne la description des modules
- `POST /predict/biometry` retourne une structure JSON valide
- `POST /predict/forensic` retourne une structure JSON valide

## Modeles

- `models/deep_learning/malaria_model.pkl` genere apres entrainement
- `models/machine_learning/nutrition_model.pkl` genere apres entrainement
- `models/machine_learning/forensic_model.pkl` genere apres entrainement

## Donnees

- dossier malaria pret dans `data/malaria/`
- dataset biometrie present dans `data/biometry/`
- dataset forensic present dans `data/forensic/`
