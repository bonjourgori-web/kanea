# Vue d'ensemble KANEA

KANEA est la declinaison nominale de la plateforme decrite dans la documentation technique source. Cette base de code cherche a rester fidele a cette documentation sur quatre axes : architecture, modules, dependances et mode d'execution.

## Architecture globale

La plateforme suit une architecture a trois couches :

- `Model Layer`
  - Module 1 - Malaria : `fast.ai / ResNet34`
  - Module 2 - Biometry : `XGBoost / RandomForest`
  - Module 3 - Forensic : `scikit-learn / ensemble`
- `Integration Layer`
  - `multibio_predict()`
- `Presentation Layer`
  - dashboard `Streamlit`
  - API `FastAPI` en extension de la feuille de route

## Positionnement methodologique

- images microscopiques : **deep learning**
- donnees anthropometriques tabulaires : **machine learning**
- donnees medico-legales morphometriques et AIMs : **machine learning classique + PCA + regressions**

## Flux de donnees

- entree 1 : images microscopiques `JPEG/PNG`
- entree 2 : donnees anthropometriques `CSV` ou saisie directe
- entree 3 : mesures morpho-genetiques
- sortie : dictionnaire Python structure via `multibio_predict()`

## Contraintes de conception

- contexte principal : Afrique subsaharienne
- interet fort pour le deploiement hors ligne
- besoin de solutions robustes sur materiel modeste
- l'application reste un outil d'aide a la decision et non un remplacement du clinicien ou de l'expert
