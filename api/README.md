# API KANEA

L'API FastAPI expose les modules principaux de la plateforme :

- `GET /`
  - statut de base de l'application
- `GET /health`
  - verification de sante simple
- `GET /info`
  - resume des modules et des technologies
- `POST /predict/malaria`
  - inference Module 1 via chemin d'image
- `POST /predict/malaria/upload`
  - inference Module 1 via upload de fichier
- `POST /predict/biometry`
  - inference Module 2
- `POST /predict/forensic`
  - inference Module 3
- `POST /predict/breast-cancer`
  - inference Module 4 via chemin local
- `POST /predict/breast-cancer/upload`
  - inference Module 4 via upload
- `POST /predict`
  - orchestration multimodale via `multibio_predict()`

Lancement local :

```bash
uvicorn api.main:app
```
