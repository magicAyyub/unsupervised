import numpy as np
from src.base import BaseModel
from src.metrics import Codebook, Latent


class SOM(BaseModel):
    """
    Custom implementation of a Self-Organizing Map (Kohonen Map).

    Latent space: discrete, the index of the BMU (Best Matching Unit) per sample
    - conceptually identical to K-means, but neurons also have a fixed position
      on a grid (grid_shape), which allows for topological visualization.

    Codebook: the feature vectors W of all neurons (their position in the data space).
    Grid coordinates (C) are not needed for decode() - they are only used for training
    and visualization, so they remain a separate attribute, not in the Codebook.
    """

    def __init__(self, grid_shape: tuple[int, int], alpha: float = 0.5,
                 gamma: float = 1.0, n_iterations: int = 5000,
                 decay: str = "linear",
                 random_state: int | None = None):
        self.grid_shape = grid_shape  # ex: (8, 8) -> 64 neurons
        self.alpha = alpha
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.decay = decay  # "linear" (classique Kohonen) ou "none" (taux fixes)
        self.random_state = random_state

        self.n_neurons = grid_shape[0] * grid_shape[1]
        self.W: np.ndarray | None = None       # shape (n_neurons, n_features) once fitted
        self.C: np.ndarray | None = None        # shape (n_neurons, 2), fixed grid coordinates
        self.history_: list[float] = []          # quantization error sampled during training

    def _build_grid_coordinates(self) -> np.ndarray:
        """
        Builds C: the fixed (x, y) coordinates of each neuron on the grid.

        Returns:
            coords: shape (n_neurons, 2)
        """
        # Double boucle pour générer les paires (row, col) fixes de chaque neurone
        coords = []
        for r in range(self.grid_shape[0]):
            for c in range(self.grid_shape[1]):
                coords.append([r, c])
        return np.array(coords, dtype=np.float32)

    def _find_bmu(self, x: np.ndarray) -> int:
        """
        Finds the index of the neuron whose feature vector W is closest to x.

        Args:
            x: shape (n_features,) - a single sample

        Returns:
            bmu_idx: index of the winning neuron (0 to n_neurons - 1)
        """
        # Distance euclidienne entre le point x et tous les poids des neurones self.W
        distances = np.linalg.norm(self.W - x, axis=1)
        return int(np.argmin(distances))

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Computes pairwise distances between all samples in X and all neurons.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            distances: shape (n_samples, n_neurons)
        """
        n_samples = X.shape[0]
        distances = np.empty((n_samples, self.n_neurons), dtype=np.float32)
        for i in range(self.n_neurons):
            distances[:, i] = np.linalg.norm(X - self.W[i], axis=1)
        return distances

    def fit(self, X: np.ndarray) -> "SOM":
        """
        Trains the SOM on the data X.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            self
        """
        rng = np.random.default_rng(self.random_state)

        self.C = self._build_grid_coordinates()
        self.history_ = []

        # Initialisation : Choisir n_neurons exemples distincts au hasard dans X
        indices = rng.choice(X.shape[0], size=self.n_neurons, replace=False)
        self.W = X[indices].copy()

        # Fréquence d'échantillonnage pour le suivi de la loss
        log_interval = max(1, self.n_iterations // 20)

        for iteration in range(self.n_iterations):
            # Calcul du taux courant selon le mode de décroissance
            if self.decay == "linear":
                progress = iteration / self.n_iterations
                current_alpha = self.alpha * (1.0 - progress)
                current_gamma = max(1e-4, self.gamma * (1.0 - progress))
            else:
                current_alpha = self.alpha
                current_gamma = self.gamma

            # Choisir un exemple au hasard S_j dans X
            idx = rng.choice(X.shape[0])
            S_j = X[idx]

            # Trouver le BMU (Best Matching Unit)
            k = self._find_bmu(S_j)

            # Calculer le terme de voisinage pour TOUS les neurones à la fois (vectorisé)
            # exp(-||C_i - C_k||^2 / (2 * gamma))
            diff_C = self.C - self.C[k]  # shape (n_neurons, 2)
            dists_sq = np.sum(diff_C ** 2, axis=1)  # shape (n_neurons,)
            neighborhood = np.exp(-dists_sq / (2.0 * current_gamma))  # shape (n_neurons,)

            # Mise à jour vectorisée de tous les vecteurs de poids W_i à la fois
            # W_i = W_i + alpha * neighborhood_i * (S_j - W_i)
            self.W += current_alpha * neighborhood[:, np.newaxis] * (S_j - self.W)

            # Échantillonnage périodique de la quantization error pour les courbes de convergence
            if iteration % log_interval == 0:
                sample_idx = rng.choice(X.shape[0], size=min(500, X.shape[0]), replace=False)
                qe = self.quantization_error(X[sample_idx])
                self.history_.append(qe)

        return self

    def quantization_error(self, X: np.ndarray) -> float:
        """
        Mean Euclidean distance between each sample and its BMU.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            Mean quantization error (scalar).
        """
        if self.W is None:
            raise ValueError("The model must be fitted before computing quantization error.")

        distances = self._compute_distances(X)
        return float(np.mean(np.min(distances, axis=1)))

    def topographic_error(self, X: np.ndarray) -> float:
        """
        Proportion of samples whose second-closest neuron is NOT a direct
        neighbor of the first-closest on the grid.

        A lower value indicates better topological preservation.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            Topographic error in [0, 1].
        """
        if self.W is None or self.C is None:
            raise ValueError("The model must be fitted before computing topographic error.")

        distances = self._compute_distances(X)

        # Trouver les indices du 1er et du 2e BMU pour chaque échantillon
        sorted_idx = np.argpartition(distances, kth=2, axis=1)[:, :2]
        bmu1 = sorted_idx[:, 0]
        bmu2 = sorted_idx[:, 1]

        # Un neurone est "voisin direct" s'il est adjacent sur la grille (distance de Manhattan = 1)
        grid_dist = np.sum(np.abs(self.C[bmu1] - self.C[bmu2]), axis=1)
        n_errors = np.sum(grid_dist > 1.0 + 1e-6)

        return float(n_errors / len(X))

    def encode(self, X: np.ndarray) -> Latent:
        """
        Projects X into the latent space (BMU index per sample).

        Args:
            X: shape (n_samples, n_features)

        Returns:
            latent: A Latent object containing the BMU indices.
        """
        if self.W is None:
            raise ValueError("The model must be fitted before encoding.")

        distances = self._compute_distances(X)
        labels = np.argmin(distances, axis=1)
        return Latent(array=labels, nature="discrete")

    def decode(self, latent: Latent) -> np.ndarray:
        """
        Reconstructs an approximation of the data from the BMU indices.

        Args:
            latent: A Latent object.

        Returns:
            X_reconstructed: shape (n_samples, n_features)
        """
        if self.W is None:
            raise ValueError("The model must be fitted before decoding.")

        # Chaque échantillon est reconstruit par les coordonnées W du neurone gagnant associé
        return self.W[latent.array]

    def get_codebook(self) -> Codebook:
        """
        Returns the standardized codebook of the model (the neurons' feature vectors).

        Returns:
            codebook: A Codebook object.
        """
        if self.W is None:
            raise ValueError("The model must be fitted before retrieving the codebook.")

        # Cohérence de type float32 pour l'estimation de stockage
        return Codebook(arrays=[self.W.astype(np.float32)])