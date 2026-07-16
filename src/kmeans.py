from __future__ import annotations

import numpy as np

from src.base import BaseModel
from src.metrics import Codebook, Latent
"""
Approche général
fit()                    -> boucle sur n_init, garde le meilleur run
 └─ _run_single()        -> UN run complet de Lloyd (la boucle itérative)
     ├─ _init_centroids()      → _kmeans_plus_plus()  
     ├─ _assign_clusters()                            
     ├─ _update_centroids()                           
     └─ test de convergence (shift <= tol)          
"""

class KMeans(BaseModel):
    # K-Means (Lloyd) from scratch, interface de compression BaseModel.
    # Quantification vectorielle : chaque point -> indice du centroïde le plus proche.

    def __init__(
        self,
        n_clusters: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        init: str = "k-means++",
        random_state: int | None = None,
    ):
        if n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++' or 'random'.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter          # itérations de Lloyd max par run
        self.tol = tol                    # seuil d'arrêt sur le déplacement des centroïdes
        self.n_init = n_init              # départs indépendants, on garde le meilleur
        self.init = init                  # "k-means++" ou "random"
        self.random_state = random_state

        # Attributs appris (fit) :
        # - centroids_ : centres (n_clusters, n_features)
        # - labels_    : cluster de chaque point d'entraînement
        # - inertia_   : somme des distances² au centroïde, du meilleur run
        # - n_iter_    : itérations du meilleur run
        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0

    # API (contrat BaseModel)

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples must be at least n_clusters.")

        rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0

        # n_init départs indépendants -> on retient la plus basse inertie (évite un minimum local)
        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._run_single(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter

        # float32 : format de stockage réaliste du codebook, comparaison de compression équitable
        self.centroids_ = best_centroids.astype(np.float32)
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = best_n_iter
        return self

    def encode(self, X: np.ndarray) -> Latent:
        # Compression : code = indice du centroïde le plus proche
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        labels, _ = self._assign_clusters(X, self.centroids_)
        return Latent(array=labels.astype(np.int64), nature="discrete")

    def decode(self, latent: Latent) -> np.ndarray:
        # Décompression : chaque code redevient son centroïde
        self._check_fitted()
        return self.centroids_[latent.array]

    def get_codebook(self) -> Codebook:
        # Dictionnaire partagé = les K centroïdes
        self._check_fitted()
        return Codebook(arrays=[self.centroids_])

    # Coeur de l'algorithme de Lloyd

    def _run_single(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        # Un run depuis une init indépendante :
        # 1. initialiser les centroïdes
        # 2. répéter : assigner les points, recalculer les centres
        # 3. stop quand les centres ne bougent presque plus (< tol)
        centroids = self._init_centroids(X, rng)

        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            labels, _ = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels, rng)

            shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
            centroids = new_centroids
            if shift <= self.tol:
                break

        # Assignation finale + inertie avec les centres convergés
        labels, min_sq = self._assign_clusters(X, centroids)
        inertia = float(min_sq.sum())
        return centroids, labels, inertia, n_iter

    @staticmethod
    def _assign_clusters(
        X: np.ndarray, centroids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Chaque point -> centroïde le plus proche.
        # Distances vectorisées via ||x - c||² = ||x||² - 2 x·c + ||c||²
        x_sq = np.sum(X ** 2, axis=1)[:, None]
        c_sq = np.sum(centroids ** 2, axis=1)[None, :]
        sq_dists = x_sq - 2.0 * (X @ centroids.T) + c_sq

        # Rattrape les valeurs légèrement négatives dues aux arrondis
        np.maximum(sq_dists, 0, out=sq_dists)

        labels = np.argmin(sq_dists, axis=1)
        min_sq = sq_dists[np.arange(X.shape[0]), labels]
        return labels, min_sq

    def _update_centroids(
        self, X: np.ndarray, labels: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        # Nouveau centre = moyenne des points assignés
        n_features = X.shape[1]
        new_centroids = np.empty((self.n_clusters, n_features), dtype=X.dtype)
        for k in range(self.n_clusters):
            members = X[labels == k]
            if members.shape[0] == 0:
                # Cluster vide : relancé sur un point tiré au hasard
                new_centroids[k] = X[rng.integers(X.shape[0])]
            else:
                new_centroids[k] = members.mean(axis=0)
        return new_centroids

    def _init_centroids(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        if self.init == "random":
            # Tirage uniforme de K points distincts
            idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            return X[idx].copy()
        return self._kmeans_plus_plus(X, rng)

    def _kmeans_plus_plus(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        # Seeding k-means++ : centres étalés, tirés un par un.
        # - 1er centre : uniforme
        # - suivant : proba proportionnelle à la distance² au centre le plus proche
        #   -> favorise les points lointains, mais reste aléatoire
        n_samples, n_features = X.shape
        centroids = np.empty((self.n_clusters, n_features), dtype=X.dtype)

        centroids[0] = X[rng.integers(n_samples)]
        closest_sq = np.sum((X - centroids[0]) ** 2, axis=1)

        for k in range(1, self.n_clusters):
            total = closest_sq.sum()
            if total == 0:
                # Tous les points confondus avec un centre -> tirage uniforme
                next_idx = rng.integers(n_samples)
            else:
                next_idx = rng.choice(n_samples, p=closest_sq / total)
            centroids[k] = X[next_idx]
            # Distance au centre le plus proche, réactualisée avec le nouveau centre
            closest_sq = np.minimum(closest_sq, np.sum((X - centroids[k]) ** 2, axis=1))

        return centroids

    def _check_fitted(self) -> None:
        if self.centroids_ is None:
            raise RuntimeError("KMeans must be fitted before calling this method.")
