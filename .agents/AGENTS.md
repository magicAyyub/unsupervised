# Directives de Développement et Contexte du Projet

Ce document sert de référence pour tout développeur (humain ou agent IA) travaillant sur ce projet d'apprentissage non supervisé. Il définit les normes de conception, les structures de données communes et l'architecture logicielle du projet.

## Architecture et Contrat d'Interface

Tous les algorithmes du projet (K-Means, PCA, AutoEncoder) doivent hériter de la classe de base abstraite `BaseModel` définie dans [src/base.py](../src/base.py) et respecter strictement ses signatures.

### Contrat d'API

Les modèles doivent implémenter les quatre méthodes suivantes :
1. `fit(self, X: np.ndarray) -> BaseModel` : Entraîne le modèle à partir d'une matrice NumPy de dimension 2D.
2. `encode(self, X: np.ndarray) -> Latent` : Projette les données dans l'espace latent et retourne un objet `Latent`.
3. `decode(self, latent: Latent) -> np.ndarray` : Reconstruit les données originales à partir de l'objet `Latent`.
4. `get_codebook(self) -> Codebook` : Retourne un objet `Codebook` contenant les paramètres de reconstruction du modèle.

### Gestion de la frontière NumPy / PyTorch

Pour l'implémentation de l'AutoEncoder (qui utilise PyTorch) :
* Toute la logique interne (tenseurs, gradients, transfert sur GPU/MPS/CPU, boucles d'entraînement) doit être confinée à l'intérieur de la classe du modèle.
* Les méthodes publiques (`fit`, `encode`, `decode`) ne doivent accepter et retourner que des types NumPy standards ou les dataclasses communes.
* Les conversions NumPy vers PyTorch (ex: `torch.from_numpy()`) et PyTorch vers NumPy (ex: `.detach().cpu().numpy()`) doivent être gérées de façon transparente en interne.

---

## Structures de Données de Compression

Les classes d'évaluation sont définies dans [src/metrics.py](../src/metrics.py) :

### Codebook

La dataclass `Codebook` modélise les paramètres fixes partagés nécessaires à la décompression :
* Attribut : `arrays: list[np.ndarray]` (ex: les centroïdes pour K-Means, les composantes principales pour la PCA, ou les poids du décodeur pour l'AutoEncoder).
* Propriété : `n_bytes` retourne la taille cumulée en octets de tous les tableaux.

### Latent

La dataclass `Latent` modélise les représentations compressées des données :
* Attributs :
  * `array: np.ndarray` (le tenseur latent).
  * `nature: str` (`"discrete"` pour K-Means, `"continuous"` pour PCA et AutoEncoder).
* Propriété : `n_bytes` retourne la taille de stockage optimale estimée. Pour un espace discret (K-Means), elle simule un codage optimal (ex: `uint8` si $K \le 256$) plutôt que d'utiliser le type d'entier brute de NumPy.

---

## Directives de Code et Normes Linguistiques

* **Langue du code** : Tout le code (noms de variables, fonctions, classes, scripts) doit être rédigé en anglais.
* **Docstrings** : La documentation de l'API (docstrings de classes et de fonctions) doit être écrite en anglais pour conserver une cohérence professionnelle.
* **Commentaires** : Les commentaires explicatifs internes (inline comments) peuvent être rédigés en français.
* **Format des images** : Les jeux de données doivent être chargés sous forme de tenseurs d'images 4D standards `(batch_size, channels, height, width)`. Tout aplatissement pour les modèles linéaires (K-Means, PCA, Linear AutoEncoder) doit s'effectuer à la volée avec la méthode `.flatten(start_dim=1)` afin de préserver la compatibilité avec de futures architectures convolutionnelles.
* **Évaluation** : Aucune formule de calcul de poids ou de métriques de compression ne doit être dupliquée dans les classes des modèles. Tout calcul de performance de compression doit faire appel à la fonction centralisée `compression_report` dans [src/metrics.py](../src/metrics.py).
