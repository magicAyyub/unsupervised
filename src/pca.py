import numpy as np
from src.base import BaseModel
from src.metrics import Codebook, Latent


class PCA(BaseModel):
    """
    Custom implementation of PCA.

    Latent space: continuous, a vector of n_components real coordinates per sample.
    Codebook: the retained eigenvectors (components) and the mean used for centering.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components: np.ndarray | None = None  # shape (n_features, n_components) once fitted
        self.mean: np.ndarray | None = None         # shape (n_features,) once fitted
        self.explained_variance: np.ndarray | None = None  # eigenvalues retained, useful for reports

    def fit(self, X: np.ndarray) -> "PCA":
        """
        Learns the principal components from the data X.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            self
        """
        # Calculer et stocker la moyenne par colonne, puis centrer les données
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Calculer la matrice de covariance (rowvar=False car les features sont en colonnes)
        # On conserve la précision float64 ici pour la stabilité numérique des calculs
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Calculer les valeurs et vecteurs propres
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Trier par ordre décroissant de valeur propre (car eigh trie par ordre croissant)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Conserver les n_components premiers composants et convertir physiquement
        # en float32 (précision standard en ML pour économiser la mémoire et éviter l'upcasting)
        self.components = eigenvectors[:, :self.n_components].astype(np.float32)
        self.explained_variance = eigenvalues[:self.n_components].astype(np.float32)
        self.mean = self.mean.astype(np.float32)

        return self

    def encode(self, X: np.ndarray) -> Latent:
        """
        Projects X into the latent space.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            latent: A Latent object containing the projected coordinates.
        """
        if self.components is None or self.mean is None:
            raise ValueError("The model must be fitted before encoding.")

        # La projection produit naturellement un tableau float32 puisque self.mean et self.components le sont
        X_centered = X - self.mean
        latent_coords = np.dot(X_centered, self.components)

        return Latent(array=latent_coords, nature="continuous")

    def decode(self, latent: Latent) -> np.ndarray:
        """
        Reconstructs an approximation of the data from the latent coordinates.

        Args:
            latent: A Latent object.

        Returns:
            X_reconstructed: shape (n_samples, n_features)
        """
        if self.components is None or self.mean is None:
            raise ValueError("The model must be fitted before decoding.")

        # Pas d'upcasting silencieux en float64 car latent.array, self.components et self.mean sont tous en float32
        return np.dot(latent.array, self.components.T) + self.mean

    def get_codebook(self) -> Codebook:
        """
        Returns the standardized codebook of the model (components and mean).

        Returns:
            codebook: A Codebook object.
        """
        if self.components is None or self.mean is None:
            raise ValueError("The model must be fitted before retrieving the codebook.")

        return Codebook(arrays=[self.components, self.mean])