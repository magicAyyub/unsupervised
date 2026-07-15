from abc import ABC, abstractmethod
import numpy as np
from src.metrics import Codebook, Latent

class BaseModel(ABC):
    """
    Common interface for the project algorithms (K-Means, PCA, AutoEncoder).
    Each algorithm must implement these methods to ensure that compression
    metrics calculations are uniform.
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseModel":
        """
        Fits the model parameters from data X.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            self
        """
        pass

    @abstractmethod
    def encode(self, X: np.ndarray) -> Latent:
        """
        Projects data X into the latent space.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            latent: A Latent object representing the encoded data.
        """
        pass

    @abstractmethod
    def decode(self, latent: Latent) -> np.ndarray:
        """
        Reconstructs the original data from the latent space.

        Args:
            latent: A Latent object.

        Returns:
            X_reconstructed: shape (n_samples, n_features)
        """
        pass

    @abstractmethod
    def get_codebook(self) -> Codebook:
        """
        Returns the standardized codebook of the model containing its reconstruction parameters.

        Returns:
            codebook: A Codebook object.
        """
        pass
