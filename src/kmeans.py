import numpy as np
from src.base import BaseModel
from src.metrics import Codebook, Latent


class KMeans(BaseModel):
    """
    Custom implementation of K-means.

    Latent space: discrete, one cluster index (0 to K-1) per sample.
    Codebook: positions of the K centroids (shared, independent of the number of samples).
    """

    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, random_state: int | None = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids: np.ndarray | None = None  # shape (n_clusters, n_features) once fitted

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Learns the centroids from the data X.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            self
        """
        rng = np.random.default_rng(self.random_state)

        # Initialisation : Choisir K points aléatoires distincts parmi les données de X
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        self.centroids = X[indices].copy()

        for iteration in range(self.max_iter):
            # Assigner chaque point au centroïde le plus proche
            distances = np.empty((X.shape[0], self.n_clusters))
            for k in range(self.n_clusters):
                distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)
            labels = np.argmin(distances, axis=1)

            # Recalculer les centroïdes
            new_centroids = np.empty_like(self.centroids)
            for k in range(self.n_clusters):
                points_in_cluster = X[labels == k]
                if len(points_in_cluster) > 0:
                    new_centroids[k] = points_in_cluster.mean(axis=0)
                else:
                    # Si un cluster est vide, on le réinitialise sur un point aléatoire de X
                    new_centroids[k] = X[rng.choice(X.shape[0])]

            # Condition d'arrêt : on s'arrête si le déplacement total des centroïdes est inférieur à tol
            drift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            
            if drift < self.tol:
                break

        return self

    def encode(self, X: np.ndarray) -> Latent:
        """
        Projects X into the latent space (cluster indices).

        Args:
            X: shape (n_samples, n_features)

        Returns:
            latent: A Latent object containing the cluster indices.
        """
        if self.centroids is None:
            raise ValueError("The model must be fitted before encoding.")

        distances = np.empty((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)
        labels = np.argmin(distances, axis=1)

        return Latent(array=labels, nature="discrete")

    def decode(self, latent: Latent) -> np.ndarray:
        """
        Reconstructs an approximation of the data from the cluster indices.

        Args:
            latent: A Latent object.

        Returns:
            X_reconstructed: shape (n_samples, n_features)
        """
        if self.centroids is None:
            raise ValueError("The model must be fitted before decoding.")

        # Chaque indice est remplacé par les coordonnées du centroïde associé
        return self.centroids[latent.array]

    def get_codebook(self) -> Codebook:
        """
        Returns the standardized codebook of the model (the centroids).

        Returns:
            codebook: A Codebook object.
        """
        if self.centroids is None:
            raise ValueError("The model must be fitted before retrieving the codebook.")
        return Codebook(arrays=[self.centroids])