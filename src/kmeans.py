from __future__ import annotations

import numpy as np

from src.base import BaseModel
from src.metrics import Codebook, Latent


class KMeans(BaseModel):
    """
    From-scratch K-Means clustering (Lloyd's algorithm), implementing the
    project's BaseModel compression interface.

    In the compression setting, K-Means performs vector quantization: every
    sample is encoded by the index of its nearest centroid (a discrete latent
    code) and decoded back to the centroid vector itself. The K centroids form
    the shared codebook required for reconstruction.

    Attributes set after fit:
        centroids_: cluster centers of shape (n_clusters, n_features).
        labels_: cluster assignment of each training sample.
        inertia_: sum of squared distances of samples to their centroid.
        n_iter_: number of Lloyd iterations of the best run.
    """

    def __init__(
        self,
        n_clusters: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        init: str = "k-means++",
        random_state: int | None = None,
    ):
        """
        Args:
            n_clusters: number of clusters K.
            max_iter: maximum number of Lloyd iterations per run.
            tol: convergence threshold on the centroid shift (Frobenius norm).
            n_init: number of independent runs; the lowest-inertia one is kept.
            init: centroid initialization, "k-means++" or "random".
            random_state: seed for reproducible initializations.
        """
        if n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++' or 'random'.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.random_state = random_state

        # Attributs appris, renseignés par fit
        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0

    # API publique (contrat BaseModel)

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples must be at least n_clusters.")

        rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0

        # Plusieurs initialisations pour limiter le risque de minimum local
        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._run_single(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter

        # Les centroïdes sont stockés en float32 : c'est le format de stockage
        # réaliste du codebook et cela garde la comparaison de compression équitable.
        self.centroids_ = best_centroids.astype(np.float32)
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = best_n_iter
        return self

    def encode(self, X: np.ndarray) -> Latent:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        labels, _ = self._assign_clusters(X, self.centroids_)
        return Latent(array=labels.astype(np.int64), nature="discrete")

    def decode(self, latent: Latent) -> np.ndarray:
        self._check_fitted()
        # Décompression : chaque code entier est remplacé par son centroïde
        return self.centroids_[latent.array]

    def get_codebook(self) -> Codebook:
        self._check_fitted()
        return Codebook(arrays=[self.centroids_])

    # Coeur de l'algorithme de Lloyd

    def _run_single(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Runs a single K-Means from an independent initialization."""
        centroids = self._init_centroids(X, rng)

        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            labels, _ = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels, rng)

            # Critère d'arrêt : déplacement global des centroïdes sous le seuil
            shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
            centroids = new_centroids
            if shift <= self.tol:
                break

        # Assignation finale avec les centroïdes convergés
        labels, min_sq = self._assign_clusters(X, centroids)
        inertia = float(min_sq.sum())
        return centroids, labels, inertia, n_iter

    @staticmethod
    def _assign_clusters(
        X: np.ndarray, centroids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Assigns each sample to its nearest centroid.

        Returns the cluster labels and the squared distance to the chosen
        centroid. Distances use the expansion
        ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 to stay fully vectorized.
        """
        x_sq = np.sum(X ** 2, axis=1)[:, None]
        c_sq = np.sum(centroids ** 2, axis=1)[None, :]
        sq_dists = x_sq - 2.0 * (X @ centroids.T) + c_sq

        # Corrige les valeurs légèrement négatives dues aux erreurs d'arrondi
        np.maximum(sq_dists, 0, out=sq_dists)

        labels = np.argmin(sq_dists, axis=1)
        min_sq = sq_dists[np.arange(X.shape[0]), labels]
        return labels, min_sq

    def _update_centroids(
        self, X: np.ndarray, labels: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Recomputes each centroid as the mean of its assigned samples."""
        n_features = X.shape[1]
        new_centroids = np.empty((self.n_clusters, n_features), dtype=X.dtype)
        for k in range(self.n_clusters):
            members = X[labels == k]
            if members.shape[0] == 0:
                # Cluster vide : on le relance sur un point tiré au hasard
                new_centroids[k] = X[rng.integers(X.shape[0])]
            else:
                new_centroids[k] = members.mean(axis=0)
        return new_centroids

    def _init_centroids(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        if self.init == "random":
            idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            return X[idx].copy()
        return self._kmeans_plus_plus(X, rng)

    def _kmeans_plus_plus(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """
        k-means++ seeding: spreads the initial centroids by sampling each new
        one with probability proportional to its squared distance to the
        closest already chosen centroid.
        """
        n_samples, n_features = X.shape
        centroids = np.empty((self.n_clusters, n_features), dtype=X.dtype)

        # Premier centre tiré uniformément
        centroids[0] = X[rng.integers(n_samples)]
        closest_sq = np.sum((X - centroids[0]) ** 2, axis=1)

        for k in range(1, self.n_clusters):
            total = closest_sq.sum()
            if total == 0:
                # Tous les points coïncident déjà avec un centre : tirage uniforme
                next_idx = rng.integers(n_samples)
            else:
                next_idx = rng.choice(n_samples, p=closest_sq / total)
            centroids[k] = X[next_idx]
            # Met à jour la distance au centre le plus proche
            closest_sq = np.minimum(closest_sq, np.sum((X - centroids[k]) ** 2, axis=1))

        return centroids

    def _check_fitted(self) -> None:
        if self.centroids_ is None:
            raise RuntimeError("KMeans must be fitted before calling this method.")
