# Projet Apprentissage Non Supervisé

Projet de groupe sur l'apprentissage non supervisé comprenant K-Means, PCA, et les AutoEncoders.

## Installation

Ce projet utilise le gestionnaire de paquets `uv` pour installer les dépendances et gérer l'environnement virtuel.

```bash
# Installer les dépendances et créer l'environnement virtuel
uv sync
```

## Utilisation

Les notebooks de travail se trouvent dans le dossier `notebooks/`. Pour les lancer :

```bash
uv run jupyter lab
```

## Fichiers de travail par algorithme

Chaque membre du groupe travaille sur son algorithme dans ses fichiers dédiés :

* **K-Means**
  * Algorithme : [src/kmeans.py](src/kmeans.py)
  * Expérimentations : [notebooks/01_kmeans.ipynb](notebooks/01_kmeans.ipynb)

* **PCA**
  * Algorithme : [src/pca.py](src/pca.py)
  * Expérimentations : [notebooks/02_pca.ipynb](notebooks/02_pca.ipynb)

* **AutoEncoder**
  * Algorithme : [src/autoencoder.py](src/autoencoder.py)
  * Expérimentations : [notebooks/03_autoencoder.ipynb](notebooks/03_autoencoder.ipynb)

Les fonctions de chargement de données communes sont regroupées dans [src/dataset.py](src/dataset.py) et les utilitaires dans [src/helper.py](src/helper.py).
